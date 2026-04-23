"""
eval_loss.py - Compute held-out loss / perplexity for vanilla vs fine-tuned.

Loads the test split (or eval split) and scores each example by computing the
LM loss on the full chat-formatted sequence, masked so loss only counts
assistant tokens. Returns mean loss + perplexity for both vanilla base and
base + adapter.

This is the quantitative truth-test of "did the fine-tune actually learn
anything beyond a stylistic nudge?"

Run:
    python src/IV_inference/eval_loss.py
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
from pathlib import Path
from datetime import datetime

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.IV_inference.gemma4_integration import MODELS, resolve_model_id, _detect_device, _detect_dtype

CYAN = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; MAGENTA = "\033[95m"
RED = "\033[91m"; GREY = "\033[90m"; BOLD = "\033[1m"; RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _load_base(model_id: str, hf_token: str | None):
    """Load just the base model + processor for scoring."""
    from transformers import AutoProcessor

    dev = _detect_device()
    dtype = _detect_dtype(dev)
    info = MODELS.get(next((k for k, v in MODELS.items() if v["id"] == model_id), ""), {})
    is_mm = info.get("multimodal", False)

    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
    load_kwargs = {"dtype": dtype, "token": hf_token}

    if is_mm:
        from transformers import AutoModelForMultimodalLM
        base = AutoModelForMultimodalLM.from_pretrained(model_id, **load_kwargs)
    else:
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    if dev in ("cuda", "mps"):
        base = base.to(dev)
    base.eval()
    return processor, base, dev


def _attach_adapter(base, adapter_path: str):
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, adapter_path, low_cpu_mem_usage=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Loss scoring
# ---------------------------------------------------------------------------

def _build_prompt_and_full(processor, messages: list[dict], thinking: bool = False) -> tuple[str, str]:
    """
    Returns:
      prompt_text: chat template applied to messages WITHOUT the assistant turn
                   (so we can locate where assistant tokens begin).
      full_text:   chat template applied to all messages (no generation prompt).
    """
    user_only = [m for m in messages if m["role"] != "assistant"]
    prompt_text = processor.apply_chat_template(
        user_only,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    full_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=thinking,
    )
    return prompt_text, full_text


@torch.no_grad()
def _score_example(model, processor, messages: list[dict], device: str, max_len: int = 4096) -> dict:
    """Compute mean per-token cross-entropy on the assistant tokens only."""
    prompt_text, full_text = _build_prompt_and_full(processor, messages)

    # Tokenize full and prompt-only to find the assistant span.
    full = processor(text=full_text, return_tensors="pt").to(device)
    prompt = processor(text=prompt_text, return_tensors="pt")

    input_ids = full["input_ids"]
    if input_ids.shape[-1] > max_len:
        input_ids = input_ids[:, :max_len]
        full["input_ids"] = input_ids
        full["attention_mask"] = full["attention_mask"][:, :max_len]

    prompt_len = prompt["input_ids"].shape[-1]
    total_len = input_ids.shape[-1]
    assistant_len = max(0, total_len - prompt_len)
    if assistant_len < 2:
        return {"loss": float("nan"), "n_tokens": 0, "skipped": True}

    labels = input_ids.clone()
    # Mask everything before the assistant span so loss is only on the response
    labels[:, :prompt_len] = -100

    out = model(**full, labels=labels)
    loss = float(out.loss.item())
    return {"loss": loss, "n_tokens": int(assistant_len), "skipped": False}


def score_dataset(model, processor, rows: list[dict], device: str, label: str) -> dict:
    losses = []
    tok_counts = []
    skipped = 0
    print(f"\n{CYAN}Scoring {label} on {len(rows)} examples...{RESET}")
    for i, row in enumerate(rows, 1):
        msgs = row.get("messages") or []
        if not any(m.get("role") == "assistant" for m in msgs):
            skipped += 1
            continue
        try:
            res = _score_example(model, processor, msgs, device)
        except Exception as e:
            print(f"  {RED}[{i}] error: {e}{RESET}")
            skipped += 1
            continue
        if res["skipped"]:
            skipped += 1
            continue
        losses.append(res["loss"])
        tok_counts.append(res["n_tokens"])
        ppl = math.exp(min(res["loss"], 20))
        print(f"  [{i:>3}/{len(rows)}] loss={res['loss']:.4f}  ppl={ppl:.2f}  tokens={res['n_tokens']}")
    if not losses:
        return {"label": label, "n": 0, "mean_loss": float("nan"), "perplexity": float("nan"), "skipped": skipped}
    mean_loss = sum(losses) / len(losses)
    return {
        "label": label,
        "n": len(losses),
        "mean_loss": round(mean_loss, 4),
        "perplexity": round(math.exp(min(mean_loss, 20)), 3),
        "skipped": skipped,
        "per_example_loss": [round(x, 4) for x in losses],
    }


# ---------------------------------------------------------------------------
# TUI
# ---------------------------------------------------------------------------

def _pick_split() -> Path:
    options = []
    for name in ("test", "eval"):
        p = REPO_ROOT / "data" / "input" / name
        if p.exists():
            for f in sorted(p.glob("*.jsonl")):
                options.append(f)
    if not options:
        raise SystemExit("No test/eval splits found in data/input/")
    print(f"\n{YELLOW}Pick held-out split:{RESET}")
    for i, p in enumerate(options, 1):
        print(f"  [{i}] {p.parent.name}/{p.name}")
    raw = input(f"\n  Choice [1]: ").strip() or "1"
    idx = int(raw) - 1 if raw.isdigit() and 1 <= int(raw) <= len(options) else 0
    return options[idx]


def _pick_adapter() -> Path | None:
    models_dir = REPO_ROOT / "data" / "output" / "models"
    if not models_dir.exists():
        return None
    adapters = sorted([p for p in models_dir.glob("*") if p.is_dir()])
    if not adapters:
        return None
    print(f"\n{YELLOW}Pick adapter:{RESET}")
    for i, p in enumerate(adapters, 1):
        print(f"  [{i}] {p.name}")
    raw = input(f"\n  Choice [1]: ").strip() or "1"
    idx = int(raw) - 1 if raw.isdigit() and 1 <= int(raw) <= len(adapters) else 0
    return adapters[idx]


def _detect_alias(name: str) -> str:
    name = name.lower()
    for alias in ("e4b", "e2b", "26b", "31b"):
        if alias in name:
            return alias
    return "e2b"


def main():
    print(f"\n{BOLD}{MAGENTA}{'=' * 60}{RESET}")
    print(f"  {BOLD}Held-out Loss / Perplexity{RESET}")
    print(f"  Vanilla base vs base + adapter")
    print(f"{BOLD}{MAGENTA}{'=' * 60}{RESET}")

    split_path = _pick_split()
    adapter_path = _pick_adapter()
    if not adapter_path:
        print(f"{YELLOW}No adapter found. Aborting.{RESET}")
        return
    alias = _detect_alias(adapter_path.name)
    print(f"\n  Split:   {split_path.parent.name}/{split_path.name}")
    print(f"  Adapter: {adapter_path.name}")
    print(f"  Base:    {alias}")

    raw = input(f"\n{YELLOW}Limit examples? [enter for all]:{RESET}\n  > ").strip()
    limit = int(raw) if raw.isdigit() else None
    rows = _load_jsonl(split_path, limit=limit)
    print(f"  Loaded {len(rows)} examples.")

    hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN") or os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)

    model_id = resolve_model_id(alias)

    # Vanilla
    print(f"\n{CYAN}Loading vanilla base...{RESET}")
    processor, base, device = _load_base(model_id, hf_token)
    vanilla_result = score_dataset(base, processor, rows, device, label="vanilla")

    # Tuned
    print(f"\n{MAGENTA}Attaching adapter...{RESET}")
    tuned = _attach_adapter(base, str(adapter_path))
    tuned_result = score_dataset(tuned, processor, rows, device, label="tuned")

    # Cleanup
    del tuned, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Compare
    print(f"\n{BOLD}{YELLOW}=== Held-out comparison ==={RESET}")
    print(f"  vanilla:  loss={vanilla_result['mean_loss']}  ppl={vanilla_result['perplexity']}  n={vanilla_result['n']}")
    print(f"  tuned:    loss={tuned_result['mean_loss']}  ppl={tuned_result['perplexity']}  n={tuned_result['n']}")
    delta = round(vanilla_result["mean_loss"] - tuned_result["mean_loss"], 4)
    print(f"  delta:    {delta}  (+ = tuned is better)")

    if delta > 0.5:
        print(f"  {GREEN}Strong improvement. Adapter clearly internalized something.{RESET}")
    elif delta > 0.1:
        print(f"  {GREEN}Modest improvement. Real but not dramatic.{RESET}")
    elif delta > -0.1:
        print(f"  {YELLOW}No meaningful difference. Adapter is essentially neutral on held-out.{RESET}")
    else:
        print(f"  {RED}Tuned is WORSE on held-out. Likely overfit to format on train only.{RESET}")

    out = {
        "split": str(split_path.relative_to(REPO_ROOT)),
        "adapter": adapter_path.name,
        "base": alias,
        "n_examples": len(rows),
        "vanilla": vanilla_result,
        "tuned": tuned_result,
        "delta_loss_vanilla_minus_tuned": delta,
        "timestamp": datetime.now().isoformat(),
    }
    out_dir = REPO_ROOT / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_loss_{adapter_path.name}_{split_path.parent.name}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{GREEN}Saved:{RESET} {out_path}")


if __name__ == "__main__":
    main()
