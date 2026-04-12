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

ADVANCED_SCAFFOLDED_SYSTEM_PROMPT = (
    "You are a creative reasoning engine. When given a task, use a full staged process: "
    "1) Curiosity map 2) Curiosity expand 3) Curiosity distill 4) Socratic output "
    "5) Creativity research plan 6) Creativity branch 7) Develop each branch "
    "8) Selection and pruning 9) Combinatory mixing 10) Final synthesis 11) Critic.\n\n"
    "Keep the stages explicit and feed curiosity steering into creativity throughout."
)


def load_eval_prompts(path: Path | None = None) -> list[str]:
    if path is None:
        path = REPO_ROOT / "data" / "input" / "seed_prompts.json"
    with open(path) as f:
        return json.load(f).get("eval_held_out", [])


def _summarize_pipeline_result(result: dict) -> str:
    final_output = result.get("final_output", [])
    verdict = "?"
    iterations = result.get("loop", [])
    if iterations:
        verdict = iterations[-1].get("critic", {}).get("verdict", "?")
    runner = result.get("_meta", {}).get("runner", "simple")
    lines = [f"Pipeline runner: {runner}", f"Critic verdict: {verdict}"]
    if final_output:
        lines.append("Final output:")
        for item in final_output:
            lines.append(f"- {item}")
    return "\n".join(lines)


def _run_eval_mode(generate_fn, prompt: str, mode: str) -> tuple[str, dict | None]:
    mode = (mode or "prompt_simple").strip().lower()
    if mode == "prompt_advanced":
        return generate_fn(ADVANCED_SCAFFOLDED_SYSTEM_PROMPT, prompt), None
    if mode == "simple_pipeline":
        from src.I_pipeline.runner import run_loop

        result = run_loop(task=prompt, generate_fn=generate_fn, max_iterations=1, verbose=False)
        return _summarize_pipeline_result(result), result
    if mode == "advanced_pipeline":
        from src.I_pipeline.runner_advanced import run_advanced_loop

        result = run_advanced_loop(task=prompt, generate_fn=generate_fn, max_iterations=1, verbose=False)
        return _summarize_pipeline_result(result), result
    return generate_fn(SCAFFOLDED_SYSTEM_PROMPT, prompt), None


def evaluate(generate_fn_vanilla, generate_fn_tuned=None, prompts=None, verbose=True, scaffold_mode: str = "prompt_simple") -> list[dict]:
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
        tier2_text, tier2_trace = _run_eval_mode(generate_fn_vanilla, prompt, scaffold_mode)
        row["tier2_scaffolded"] = tier2_text
        row["tier2_mode"] = scaffold_mode
        if tier2_trace is not None:
            row["tier2_trace"] = tier2_trace
        if verbose:
            _print_response("Tier 2", row["tier2_scaffolded"], GREEN)

        # Tier 3: fine-tuned
        if generate_fn_tuned:
            if verbose:
                print(f"\n  {MAGENTA}Tier 3: Fine-tuned...{RESET}")
            if scaffold_mode in ("simple_pipeline", "advanced_pipeline"):
                tier3_text, tier3_trace = _run_eval_mode(generate_fn_tuned, prompt, scaffold_mode)
                row["tier3_tuned"] = tier3_text
                row["tier3_mode"] = scaffold_mode
                if tier3_trace is not None:
                    row["tier3_trace"] = tier3_trace
            elif scaffold_mode == "prompt_advanced":
                row["tier3_tuned"] = generate_fn_tuned(ADVANCED_SCAFFOLDED_SYSTEM_PROMPT, prompt)
                row["tier3_mode"] = scaffold_mode
            else:
                row["tier3_tuned"] = generate_fn_tuned("You are a helpful assistant.", prompt)
                row["tier3_mode"] = "direct"
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
    from src.IV_inference.gemma4_integration import load_gemma4, load_finetuned_gemma4, MODELS

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

    tuned_fn = None
    print(f"\n{YELLOW}Include fine-tuned adapter in Tier 3? [y/N]{RESET}")
    if input("  > ").strip().lower() in ("y", "yes"):
        models_dir = REPO_ROOT / "data" / "output" / "models"
        adapter_dirs = [p for p in sorted(models_dir.glob("*")) if p.is_dir()]
        if not adapter_dirs:
            print(f"  {YELLOW}No adapter dirs found in data/output/models/{RESET}")
        else:
            print(f"\n{YELLOW}Available adapters:{RESET}")
            for i, path in enumerate(adapter_dirs, 1):
                print(f"  [{i}] {path.name}")
            raw = input(f"\n  Choice [1]: ").strip() or "1"
            idx = int(raw) - 1 if raw.isdigit() and 1 <= int(raw) <= len(adapter_dirs) else 0
            adapter_path = adapter_dirs[idx]
            tuned_fn = load_finetuned_gemma4(str(adapter_path), base_model=model_alias)

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

    print(f"\n{YELLOW}Tier 2/Tier 3 scaffold mode:{RESET}")
    print("  [1] prompt scaffold (simple)")
    print("  [2] prompt scaffold (advanced)")
    print("  [3] simple pipeline runner")
    print("  [4] advanced pipeline runner")
    scaffold_raw = input("\n  Choice [1]: ").strip() or "1"
    scaffold_mode = {
        "1": "prompt_simple",
        "2": "prompt_advanced",
        "3": "simple_pipeline",
        "4": "advanced_pipeline",
    }.get(scaffold_raw, "prompt_simple")

    # Run
    results = evaluate(
        generate_fn_vanilla=generate_fn,
        generate_fn_tuned=tuned_fn,
        prompts=selected,
        verbose=True,
        scaffold_mode=scaffold_mode,
    )
    export_eval(results)
    print(f"\n  {GREEN}Evaluation complete: {len(results)} prompts.{RESET}")


if __name__ == "__main__":
    _tui()
