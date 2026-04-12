"""
evaluate.py - 3-tier evaluation: vanilla vs scaffolded vs fine-tuned.

Run interactively:
    python src/IV_inference/evaluate.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Colors
CYAN = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; MAGENTA = "\033[95m"
RED = "\033[91m"; GREY = "\033[90m"; BOLD = "\033[1m"; RESET = "\033[0m"

SCAFFOLDED_SYSTEM_PROMPT = (
    "You are a creative reasoning engine. When given a task, think through it "
    "using Curiosity -> Creativity -> Critic. Surface hidden assumptions, "
    "explore diverse branches, cross-pollinate, critically evaluate, then present "
    "your strongest candidates.\n\n"
    "Structure: 1) Curiosity 2) Creativity 3) Critic 4) Final Output"
)


def load_eval_prompts(path: Path | None = None) -> list[str]:
    if path is None:
        path = REPO_ROOT / "data" / "input" / "seed_prompts.json"
    with open(path) as f:
        return json.load(f).get("eval_held_out", [])


def evaluate(generate_fn_vanilla, generate_fn_tuned=None, prompts=None, verbose=True) -> list[dict]:
    if prompts is None:
        prompts = load_eval_prompts()

    results = []
    for i, prompt in enumerate(prompts, 1):
        if verbose:
            print(f"\n{BOLD}{'='*60}{RESET}")
            print(f"  {BOLD}Eval {i}/{len(prompts)}{RESET}")
            print(f"  {prompt[:70]}...")
            print(f"{'='*60}")

        row = {"prompt": prompt, "timestamp": datetime.now().isoformat()}

        # Tier 1: vanilla
        if verbose:
            print(f"\n  {CYAN}Tier 1: Vanilla...{RESET}")
        row["tier1_vanilla"] = generate_fn_vanilla("You are a helpful assistant.", prompt)
        if verbose:
            _print_response("Tier 1", row["tier1_vanilla"], CYAN)

        # Tier 2: scaffolded
        if verbose:
            print(f"\n  {GREEN}Tier 2: Scaffolded...{RESET}")
        row["tier2_scaffolded"] = generate_fn_vanilla(SCAFFOLDED_SYSTEM_PROMPT, prompt)
        if verbose:
            _print_response("Tier 2", row["tier2_scaffolded"], GREEN)

        # Tier 3: fine-tuned
        if generate_fn_tuned:
            if verbose:
                print(f"\n  {MAGENTA}Tier 3: Fine-tuned...{RESET}")
            row["tier3_tuned"] = generate_fn_tuned("You are a helpful assistant.", prompt)
            if verbose:
                _print_response("Tier 3", row["tier3_tuned"], MAGENTA)
        else:
            row["tier3_tuned"] = "(not available)"

        results.append(row)
    return results


def _print_response(label: str, text: str, color: str):
    # Truncate long responses for readability
    lines = text.strip().split("\n")
    preview = "\n".join(lines[:15])
    if len(lines) > 15:
        preview += f"\n{GREY}  ... ({len(lines) - 15} more lines){RESET}"
    for line in preview.split("\n"):
        print(f"  {color}  {line}{RESET}")


def export_eval(results: list[dict], output_path: Path | None = None):
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = REPO_ROOT / "data" / "output" / f"eval_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  {GREEN}Saved: {output_path.name}{RESET}")


# ---------------------------------------------------------------------------
# Interactive TUI
# ---------------------------------------------------------------------------

def _tui():
    from src.IV_inference.gemma4_integration import load_gemma4, MODELS

    print(f"\n{BOLD}{MAGENTA}{'='*60}{RESET}")
    print(f"  {BOLD}3-Tier Evaluation{RESET}")
    print(f"  Vanilla vs Scaffolded vs Fine-tuned")
    print(f"{BOLD}{MAGENTA}{'='*60}{RESET}")

    # Model
    print(f"\n{YELLOW}Model:{RESET}")
    aliases = list(MODELS.keys())
    for i, alias in enumerate(aliases, 1):
        info = MODELS[alias]
        default = f" {GREY}<- default{RESET}" if i == 1 else ""
        print(f"  [{i}] {alias:<5} {info['ram_gb']:>3}GB  {info['description']}{default}")
    raw = input(f"\n  Choice [1]: ").strip() or "1"
    model_alias = aliases[int(raw) - 1] if raw.isdigit() and 1 <= int(raw) <= len(aliases) else aliases[0]

    generate_fn = load_gemma4(model=model_alias)

    # Prompts
    prompts = load_eval_prompts()
    print(f"\n{YELLOW}Eval prompts:{RESET}")
    for i, p in enumerate(prompts, 1):
        print(f"  [{i}] {p[:65]}...")
    print(f"  [a] all ({len(prompts)})")
    raw = input(f"\n  Choice [a]: ").strip().lower() or "a"
    if raw == "a":
        selected = prompts
    elif raw.isdigit() and 1 <= int(raw) <= len(prompts):
        selected = [prompts[int(raw) - 1]]
    else:
        selected = prompts

    # Run
    results = evaluate(
        generate_fn_vanilla=generate_fn,
        generate_fn_tuned=None,
        prompts=selected,
        verbose=True,
    )
    export_eval(results)
    print(f"\n  {GREEN}Evaluation complete: {len(results)} prompts.{RESET}")


if __name__ == "__main__":
    _tui()
