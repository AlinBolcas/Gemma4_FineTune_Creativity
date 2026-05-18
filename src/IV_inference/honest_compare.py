"""
honest_compare.py - Vanilla vs Tuned, interactive loop.

Loads BOTH models once at startup. Then loops forever asking:
  - new prompt? new scale? new temperature?
  - generate vanilla + tuned with current settings
  - save .json + .md
  - go again

Quit with `q` at any prompt.

Run:
    python src/IV_inference/honest_compare.py
"""

from __future__ import annotations
import os, sys, gc, json, copy
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from src.IV_inference.gemma4_integration import resolve_model_id, _detect_device, _detect_dtype, MODELS

CYAN = "\033[96m"; GREEN = "\033[92m"; MAGENTA = "\033[95m"
YELLOW = "\033[93m"; GREY = "\033[90m"; RED = "\033[91m"; BOLD = "\033[1m"; RESET = "\033[0m"

SYSTEM_PROMPT = "You are a creative reasoning assistant."

DEFAULT_PROMPTS = [
    "Invent an educational product that becomes more useful when bandwidth gets worse, not better.",
    "Design a creative mentorship system for students in under-resourced schools that works offline.",
    "Reframe climate adaptation for coastal cities in a way that changes what gets built first.",
    "For the Gemma 4 Good Hackathon I need project ideas that feel 'wow' to judges.",
    "Find a name for my sausage dog that feels distinctive, a little unexpected, and still affectionate.",
]

TRACE_MARKERS = ["## Iteration", "### Curiosity", "Branch seeds:", "### Creativity", "## Final Output"]

DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMP = 0.5
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64

SCALE_OPTIONS = {"1": 1.0, "2": 2.0, "3": 4.0, "4": 8.0, "5": 0.5}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_adapter() -> Path | None:
    models_dir = REPO_ROOT / "data" / "output" / "models"
    adapters = sorted([p for p in models_dir.glob("*") if p.is_dir()]) if models_dir.exists() else []
    if not adapters:
        print(f"{YELLOW}No adapters found.{RESET}")
        return None
    print(f"\n{YELLOW}Adapter:{RESET}")
    for i, p in enumerate(adapters, 1):
        print(f"  [{i}] {p.name}")
    raw = input(f"\n  Choice [1]: ").strip() or "1"
    idx = int(raw) - 1 if raw.isdigit() and 1 <= int(raw) <= len(adapters) else 0
    return adapters[idx]


def _detect_alias(name: str) -> str:
    for alias in ("e4b", "e2b", "26b", "31b"):
        if alias in name.lower():
            return alias
    return "e2b"


def _scan_markers(text: str) -> list[str]:
    return [m for m in TRACE_MARKERS if m in text]


def _print_block(label: str, color: str, text: str, max_lines: int = 30):
    print(f"\n{BOLD}{color}{'='*60}{RESET}")
    print(f"{BOLD}{color}  {label}{RESET}")
    print(f"{color}{'='*60}{RESET}")
    lines = text.strip().split("\n")
    for line in lines[:max_lines]:
        print(f"  {line}")
    if len(lines) > max_lines:
        print(f"  {GREY}... ({len(lines) - max_lines} more lines){RESET}")


# ---------------------------------------------------------------------------
# LoRA scale handling
# ---------------------------------------------------------------------------

def _capture_original_scaling(model) -> dict:
    """Snapshot every LoRA layer's scaling dict so we can reset later."""
    snapshot = {}
    for name, module in model.named_modules():
        if hasattr(module, "scaling") and isinstance(module.scaling, dict):
            snapshot[name] = copy.deepcopy(module.scaling)
    return snapshot


def _apply_scale(model, original: dict, factor: float) -> int:
    """Reset all LoRA layers to original × factor."""
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, "scaling") and isinstance(module.scaling, dict):
            orig = original.get(name, {})
            for key in module.scaling:
                if key in orig:
                    module.scaling[key] = orig[key] * factor
                    count += 1
    return count


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate(model, processor, system: str, user: str, temp: float, max_tokens: int) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens,
            temperature=max(temp, 0.01), top_p=DEFAULT_TOP_P, top_k=DEFAULT_TOP_K,
            do_sample=temp > 0.05,
        )
    raw = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    try:
        parsed = processor.parse_response(raw)
        return parsed.get("content", str(parsed)) if isinstance(parsed, dict) else str(parsed)
    except Exception:
        return processor.decode(outputs[0][input_len:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# TUI loop
# ---------------------------------------------------------------------------

def _ask(label: str, default: str) -> str:
    raw = input(f"  {label} [{default}]: ").strip()
    return raw or default


def _pick_prompt(default_idx: int = 1) -> tuple[str, int]:
    print(f"\n{YELLOW}Prompt:{RESET}")
    for i, p in enumerate(DEFAULT_PROMPTS, 1):
        marker = " <-" if i == default_idx else ""
        print(f"  [{i}] {p[:78]}{marker}")
    print(f"  [c] Custom    [q] Quit")
    raw = input(f"\n  Choice [{default_idx}]: ").strip().lower() or str(default_idx)
    if raw == "q":
        return "", -1
    if raw == "c":
        return input("  Your prompt: ").strip(), default_idx
    if raw.isdigit() and 1 <= int(raw) <= len(DEFAULT_PROMPTS):
        return DEFAULT_PROMPTS[int(raw) - 1], int(raw)
    return DEFAULT_PROMPTS[default_idx - 1], default_idx


def _pick_scale(default_factor: float) -> float:
    print(f"\n{YELLOW}LoRA scale amplification?{RESET}  {GREY}default scale = lora_alpha/lora_r = 2.0x{RESET}")
    print(f"  [5] 0.5x  (effective 1.0x - half the adapter)")
    print(f"  [1] 1x    (effective 2.0x - default training scale)")
    print(f"  [2] 2x    (effective 4.0x - doubled)")
    print(f"  [3] 4x    (effective 8.0x - strong push)")
    print(f"  [4] 8x    (effective 16.0x - aggressive, may break coherence)")
    default_key = next((k for k, v in SCALE_OPTIONS.items() if v == default_factor), "2")
    raw = input(f"\n  Choice [{default_key}]: ").strip() or default_key
    return SCALE_OPTIONS.get(raw, default_factor)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{BOLD}{MAGENTA}{'='*60}{RESET}")
    print(f"  {BOLD}Honest Compare: Vanilla vs Tuned (loop mode){RESET}")
    print(f"  Loads both once. Iterates on prompt + scale + temp.")
    print(f"{BOLD}{MAGENTA}{'='*60}{RESET}")

    adapter_path = _pick_adapter()
    if not adapter_path:
        return
    alias = _detect_alias(adapter_path.name)
    model_id = resolve_model_id(alias)
    dev = _detect_device()
    dtype = _detect_dtype(dev)
    hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN") or os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)

    from transformers import AutoProcessor, AutoModelForMultimodalLM
    from peft import PeftModel

    print(f"\n{CYAN}Loading processor...{RESET}")
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)

    print(f"{CYAN}Loading vanilla base ({alias})...{RESET}")
    vanilla = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=dtype, token=hf_token)
    if dev in ("cuda", "mps"):
        vanilla = vanilla.to(dev)
    vanilla.eval()
    print(f"{CYAN}Vanilla ready.{RESET}")

    print(f"\n{MAGENTA}Loading tuned base ({alias}) + adapter...{RESET}")
    tuned_base = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=dtype, token=hf_token)
    if dev in ("cuda", "mps"):
        tuned_base = tuned_base.to(dev)
    tuned = PeftModel.from_pretrained(tuned_base, str(adapter_path), low_cpu_mem_usage=False)
    tuned.eval()
    original_scaling = _capture_original_scaling(tuned)
    n_lora = len(original_scaling)
    print(f"{MAGENTA}Tuned ready. {n_lora} LoRA modules captured.{RESET}")

    last_scale = 1.0
    last_prompt_idx = 1
    last_temp = DEFAULT_TEMP

    while True:
        print(f"\n{BOLD}{YELLOW}{'─'*60}{RESET}")
        prompt, last_prompt_idx = _pick_prompt(last_prompt_idx)
        if last_prompt_idx == -1:
            break

        scale_factor = _pick_scale(last_scale)
        last_scale = scale_factor

        temp_raw = _ask(f"Temperature", f"{last_temp}")
        try:
            temp = float(temp_raw)
        except ValueError:
            temp = last_temp
        last_temp = temp

        print(f"\n{GREY}prompt: {prompt[:80]}...{RESET}")
        print(f"{GREY}scale:  {scale_factor}x  |  temp: {temp}{RESET}")

        # Reset + apply scale on tuned
        n = _apply_scale(tuned, original_scaling, scale_factor)
        print(f"{GREY}LoRA scale set on {n} modules (effective {2.0 * scale_factor:.2f}x){RESET}")

        print(f"\n{CYAN}Generating vanilla...{RESET}")
        v_out = _generate(vanilla, processor, SYSTEM_PROMPT, prompt, temp, DEFAULT_MAX_TOKENS)

        print(f"{MAGENTA}Generating tuned...{RESET}")
        t_out = _generate(tuned, processor, SYSTEM_PROMPT, prompt, temp, DEFAULT_MAX_TOKENS)

        v_markers = _scan_markers(v_out)
        t_markers = _scan_markers(t_out)

        _print_block("VANILLA", CYAN, v_out)
        _print_block(f"TUNED  scale={scale_factor}x  temp={temp}", MAGENTA, t_out)

        print(f"\n{BOLD}{YELLOW}=== Verdict ==={RESET}")
        print(f"  vanilla trace markers: {v_markers or 'none'}")
        print(f"  tuned trace markers:   {t_markers or 'none'}")
        if t_markers and not v_markers:
            verdict = "Tuned spontaneously uses trained format. Fine-tuning worked."
            print(f"  {GREEN}{verdict}{RESET}")
        elif not t_markers and not v_markers:
            verdict = "Neither emits trace format. Adapter changes wording but not structure."
            print(f"  {YELLOW}{verdict}{RESET}")
        else:
            verdict = "Mixed result, see outputs above."
            print(f"  {YELLOW}{verdict}{RESET}")

        # Save with timestamp so nothing gets overwritten
        ts = datetime.now().strftime("%H%M%S")
        tag = f"scale{scale_factor:g}x_temp{temp:g}_{ts}"
        out = {
            "prompt": prompt, "system": SYSTEM_PROMPT, "alias": alias,
            "adapter": adapter_path.name,
            "lora_scale_factor": scale_factor, "temperature": temp,
            "vanilla": v_out, "tuned": t_out,
            "vanilla_markers": v_markers, "tuned_markers": t_markers,
            "verdict": verdict,
        }
        out_dir = REPO_ROOT / "data" / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"honest_compare_{alias}_{tag}.json"
        json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

        md = [
            f"# Honest Compare: Vanilla vs Tuned",
            "",
            f"**Model:** {alias} | **Adapter:** `{adapter_path.name}`",
            f"**LoRA scale:** {scale_factor}x (effective {2.0 * scale_factor:.1f}x) | **Temp:** {temp}",
            "",
            f"**System:** `{SYSTEM_PROMPT}`",
            "",
            f"**Prompt:**", f"> {prompt}",
            "", "---", "", "## Vanilla", "", v_out.strip(),
            "", "---", "", f"## Tuned (scale {scale_factor}x, temp {temp})", "", t_out.strip(),
            "", "---", "", "## Verdict", "",
            f"- Vanilla trace markers: `{v_markers or 'none'}`",
            f"- Tuned trace markers: `{t_markers or 'none'}`",
            "", f"**{verdict}**",
        ]
        md_path = out_dir / f"honest_compare_{alias}_{tag}.md"
        md_path.write_text("\n".join(md), encoding="utf-8")
        print(f"\n{GREEN}Saved: {json_path.name}{RESET}")
        print(f"{GREEN}Saved: {md_path.name}{RESET}")

    print(f"\n{GREY}Bye.{RESET}")


if __name__ == "__main__":
    main()
