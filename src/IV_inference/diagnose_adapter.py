"""
diagnose_adapter.py - Sanity check that the LoRA adapter is actually doing something.

For one prompt, generate twice with deterministic settings:
  1. Vanilla base model (no adapter)
  2. Same base + your fine-tuned adapter

Then diff the outputs. If they are byte-identical, the adapter is not being applied.
If they differ but neither contains the trace markers (### Curiosity / ## Iteration),
the adapter is loaded but its signal is too weak to override the base style.

Run:
    python src/IV_inference/diagnose_adapter.py
"""

from __future__ import annotations

import sys
import json
import difflib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.IV_inference.gemma4_integration import (
    Gemma4,
    load_finetuned_gemma4,
    MODELS,
)

CYAN = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; MAGENTA = "\033[95m"
RED = "\033[91m"; GREY = "\033[90m"; BOLD = "\033[1m"; RESET = "\033[0m"

TRAINING_SYSTEM_PROMPT = "You are a creative reasoning assistant."
DEFAULT_PROMPT = (
    "Invent an educational product that becomes more useful when bandwidth gets worse, not better."
)
TRACE_MARKERS = ["## Iteration", "### Curiosity", "### Creativity", "## Final Output", "Branch seeds:"]
MAX_TOKENS = 400


def _pick_adapter() -> Path | None:
    models_dir = REPO_ROOT / "data" / "output" / "models"
    if not models_dir.exists():
        print(f"  {YELLOW}No data/output/models directory.{RESET}")
        return None
    adapters = sorted([p for p in models_dir.glob("*") if p.is_dir()])
    if not adapters:
        print(f"  {YELLOW}No adapters found.{RESET}")
        return None

    print(f"\n{YELLOW}Pick adapter to diagnose:{RESET}")
    for i, p in enumerate(adapters, 1):
        print(f"  [{i}] {p.name}")
    raw = input(f"\n  Choice [1]: ").strip() or "1"
    idx = int(raw) - 1 if raw.isdigit() and 1 <= int(raw) <= len(adapters) else 0
    return adapters[idx]


def _detect_alias(adapter_path: Path) -> str:
    name = adapter_path.name.lower()
    for alias in ("e4b", "e2b", "26b", "31b"):
        if alias in name:
            return alias
    return "e2b"


def _generate_deterministic(generate_fn, system: str, user: str) -> str:
    """
    Wrap generate_fn so we ignore its sampling defaults and try greedy.
    The shipped generate_fn does sampling internally; here we just call it
    and accept the small variance, but we set a fixed prompt for fairness.
    """
    return generate_fn(system, user)


def _scan_markers(text: str) -> list[str]:
    return [m for m in TRACE_MARKERS if m in text]


def _diff_summary(a: str, b: str) -> dict:
    if a == b:
        return {"identical": True, "char_diff": 0, "ratio": 1.0}
    matcher = difflib.SequenceMatcher(None, a, b)
    return {
        "identical": False,
        "char_diff": abs(len(a) - len(b)),
        "ratio": round(matcher.ratio(), 3),
    }


def _print_block(title: str, color: str, text: str, max_lines: int = 18):
    print(f"\n{BOLD}{color}{title}{RESET}")
    print(f"{color}{'-' * 60}{RESET}")
    lines = text.strip().split("\n")
    for line in lines[:max_lines]:
        print(f"  {line}")
    if len(lines) > max_lines:
        print(f"  {GREY}... ({len(lines) - max_lines} more lines){RESET}")


def main():
    print(f"\n{BOLD}{MAGENTA}{'=' * 60}{RESET}")
    print(f"  {BOLD}Adapter Diagnostic{RESET}")
    print(f"  Vanilla vs Fine-tuned, same prompt, same template.")
    print(f"{BOLD}{MAGENTA}{'=' * 60}{RESET}")

    adapter_path = _pick_adapter()
    if not adapter_path:
        return
    alias = _detect_alias(adapter_path)
    print(f"\n{GREEN}Adapter:{RESET} {adapter_path.name}")
    print(f"{GREEN}Base alias:{RESET} {alias}")

    raw = input(f"\n{YELLOW}Custom prompt? [enter for default]:{RESET}\n  > ").strip()
    prompt = raw or DEFAULT_PROMPT
    print(f"{GREY}Prompt: {prompt[:80]}...{RESET}\n")

    print(f"{CYAN}Loading vanilla base model {alias}...{RESET}")
    vanilla = Gemma4(alias, system=TRAINING_SYSTEM_PROMPT, max_new_tokens=MAX_TOKENS, temperature=0.1, top_p=1.0, top_k=1)
    vanilla_fn = vanilla.generate_fn()

    print(f"\n{CYAN}Generating vanilla output...{RESET}")
    vanilla_out = _generate_deterministic(vanilla_fn, TRAINING_SYSTEM_PROMPT, prompt)

    # Free up memory before loading tuned
    del vanilla, vanilla_fn
    import gc, torch
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{MAGENTA}Loading base + adapter...{RESET}")
    tuned_fn = load_finetuned_gemma4(
        str(adapter_path),
        base_model=alias,
        max_new_tokens=MAX_TOKENS,
        temperature=0.1,
        top_p=1.0,
        top_k=1,
    )

    print(f"\n{MAGENTA}Generating tuned output...{RESET}")
    tuned_out = _generate_deterministic(tuned_fn, TRAINING_SYSTEM_PROMPT, prompt)

    # Bonus test: forced prefix priming.
    # If we manually start the assistant turn with the trace header, does the
    # tuned model continue in the trained format? This tells us whether the
    # format is in the weights but suppressed by base style, vs not learned.
    print(f"\n{MAGENTA}Generating tuned output with forced prefix priming...{RESET}")
    primed_prompt = (
        prompt
        + "\n\nFormat your answer EXACTLY like this template, filling in real content:\n"
        + "## Iteration 1\n### Curiosity\nHidden assumptions:\n- ...\nKey questions:\n- ...\n"
        + "Branch seeds: ...\n### Creativity\nResearch:\n- ...\nBranches:\n- B1: ...\n"
        + "Combinations:\n- ...\nCandidates:\n- ...\n\n## Final Output\n- ..."
    )
    primed_out = _generate_deterministic(tuned_fn, TRAINING_SYSTEM_PROMPT, primed_prompt)

    diff = _diff_summary(vanilla_out, tuned_out)
    vanilla_markers = _scan_markers(vanilla_out)
    tuned_markers = _scan_markers(tuned_out)
    primed_markers = _scan_markers(primed_out)

    _print_block("VANILLA OUTPUT", CYAN, vanilla_out)
    _print_block("TUNED OUTPUT (free)", MAGENTA, tuned_out)
    _print_block("TUNED OUTPUT (forced prefix)", YELLOW, primed_out)

    print(f"\n{BOLD}{YELLOW}=== Diagnostic Summary ==={RESET}")
    print(f"  identical:      {diff['identical']}")
    print(f"  char len diff:  {diff['char_diff']}")
    print(f"  similarity:     {diff['ratio']}")
    print(f"  vanilla trace markers:        {vanilla_markers or 'none'}")
    print(f"  tuned trace markers (free):   {tuned_markers or 'none'}")
    print(f"  tuned trace markers (primed): {primed_markers or 'none'}")

    print(f"\n{BOLD}Verdict:{RESET}")
    if diff["identical"]:
        print(f"  {RED}Adapter has zero effect. PEFT not loading or zeroed weights.{RESET}")
    elif diff["ratio"] > 0.95:
        print(f"  {YELLOW}Adapter loaded but signal is very weak (>95% similar).{RESET}")
    elif tuned_markers and not vanilla_markers:
        print(f"  {GREEN}Adapter spontaneously emits trace format. Architecture transferred.{RESET}")
    elif primed_markers and not tuned_markers:
        print(f"  {GREEN}Format is in the weights but suppressed by base style at sampling time.{RESET}")
        print(f"  {GREEN}Solution: prime the prompt, lower temperature, or train more.{RESET}")
    elif tuned_markers and vanilla_markers:
        print(f"  {YELLOW}Both produce trace markers. Effect unclear from one sample.{RESET}")
    else:
        print(f"  {YELLOW}Adapter changes output (sim={diff['ratio']}) but no trace format anywhere.{RESET}")
        print(f"  {YELLOW}Format was not learned strongly enough. Need more data / higher LoRA rank.{RESET}")

    out_dir = REPO_ROOT / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "adapter": adapter_path.name,
        "base": alias,
        "prompt": prompt,
        "vanilla_output": vanilla_out,
        "tuned_output_free": tuned_out,
        "tuned_output_primed": primed_out,
        "diff_vanilla_vs_tuned": diff,
        "vanilla_trace_markers": vanilla_markers,
        "tuned_trace_markers_free": tuned_markers,
        "tuned_trace_markers_primed": primed_markers,
    }
    out_path = out_dir / f"adapter_diagnostic_{adapter_path.name}.json"
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{GREEN}Saved: {out_path}{RESET}")


if __name__ == "__main__":
    main()
