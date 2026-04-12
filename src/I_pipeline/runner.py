"""
runner.py - The Curiosity -> Creativity -> Critic loop runner.

Run interactively:
    python src/I_pipeline/runner.py
    # or from repo root: python -m src.I_pipeline.runner
"""

import json
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

# Fix "attempted relative import with no known parent package" when running this file directly.
if __name__ == "__main__" and not __package__:
    _repo = Path(__file__).resolve().parents[2]  # .../src/I_pipeline -> repo root
    _repo_str = str(_repo)
    if _repo_str not in sys.path:
        sys.path.insert(0, _repo_str)
    __package__ = "src.I_pipeline"  # noqa: PLC2201 — enables sibling relative imports

from .prompts import (
    CURIOSITY_SYSTEM, CURIOSITY_SYSTEM_WITH_FEEDBACK,
    CREATIVITY_SYSTEM, CRITIC_SYSTEM,
    build_curiosity_prompt, build_creativity_prompt, build_critic_prompt,
)
from .schema import (
    validate_curiosity, validate_creativity, validate_critic,
    normalize_curiosity, normalize_creativity, normalize_critic,
)

# ---------------------------------------------------------------------------
# Terminal colors
# ---------------------------------------------------------------------------

CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
MAGENTA = "\033[95m"
RED     = "\033[91m"
GREY    = "\033[90m"
BOLD    = "\033[1m"
RESET   = "\033[0m"

STAGE_COLORS = {
    "curiosity": CYAN,
    "creativity": GREEN,
    "critic":    MAGENTA,
}

MAX_LOOP_ITERATIONS = 3
MAX_RETRIES_PER_CALL = 2


# ---------------------------------------------------------------------------
# Pretty printing per stage
# ---------------------------------------------------------------------------

def _print_curiosity(data: dict):
    c = STAGE_COLORS["curiosity"]
    print(f"\n{c}{BOLD}    CURIOSITY{RESET}")
    for a in data.get("hidden_assumptions", []):
        print(f"{c}    ! {a}{RESET}")
    for q in data.get("questions", []):
        qtext = q.get("question", str(q)) if isinstance(q, dict) else str(q)
        print(f"{c}    ? {qtext}{RESET}")
    seeds = data.get("branch_seeds", [])
    if seeds:
        print(f"{c}    Seeds: {' | '.join(seeds)}{RESET}")


def _print_creativity(data: dict):
    c = STAGE_COLORS["creativity"]
    print(f"\n{c}{BOLD}    CREATIVITY{RESET}")
    for r in data.get("research", [])[:3]:
        print(f"{GREY}    ~ {r}{RESET}")
    for b in data.get("branches", []):
        cands = ", ".join(b.get("candidates", [])[:3])
        print(f"{c}    {b.get('id','?')}: {b.get('frame','')} -> {cands}{RESET}")
    for p in data.get("pruned", []):
        print(f"{RED}    x {p.get('id','?')}: {p.get('reason','')}{RESET}")
    for combo in data.get("combinations", []):
        print(f"{c}    + {combo.get('from',[])} -> {combo.get('result','')}{RESET}")
        if combo.get("novelty_note"):
            print(f"{GREY}      ({combo['novelty_note']}){RESET}")
    for de in data.get("dead_ends", []):
        print(f"{RED}    - {de}{RESET}")
    print(f"{c}{BOLD}    Output:{RESET}")
    for o in data.get("output", []):
        print(f"{c}    * {o}{RESET}")


def _print_critic(data: dict):
    c = STAGE_COLORS["critic"]
    verdict = data.get("verdict", "?").upper()
    vc = GREEN if verdict == "PASS" else RED
    print(f"\n{c}{BOLD}    CRITIC{RESET}  {vc}{BOLD}[{verdict}]{RESET}")
    for s in data.get("scores", []):
        n, r = s.get("novelty", "?"), s.get("relevance", "?")
        print(f"{c}    {s.get('candidate','?')[:50]}  n={n}/10 r={r}/10{RESET}")
        if s.get("notes"):
            print(f"{GREY}      {s['notes']}{RESET}")
    if verdict == "FAIL":
        for fb in data.get("feedback_for_curiosity", []):
            print(f"{YELLOW}    -> {fb}{RESET}")


STAGE_PRINTERS = {
    "curiosity":  _print_curiosity,
    "creativity": _print_creativity,
    "critic":     _print_critic,
}


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def run_loop(
    task: str,
    generate_fn: Callable[[str, str], str],
    domain: str = "general",
    max_iterations: int = MAX_LOOP_ITERATIONS,
    verbose: bool = True,
) -> dict:
    loop_trace = []
    critic_feedback = None

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"\n{BOLD}{'='*60}{RESET}")
            print(f"  {BOLD}Iteration {iteration}/{max_iterations}{RESET}")
            print(f"{'='*60}")

        # CURIOSITY
        if verbose:
            print(f"\n  {CYAN}[1/3] Running Curiosity...{RESET}")
        c_sys = CURIOSITY_SYSTEM_WITH_FEEDBACK if critic_feedback else CURIOSITY_SYSTEM
        c_prompt = build_curiosity_prompt(task, critic_feedback)
        curiosity_out = _call_stage(generate_fn, c_sys, c_prompt, "curiosity", verbose)
        if verbose:
            _print_curiosity(curiosity_out)

        # CREATIVITY
        if verbose:
            print(f"\n  {GREEN}[2/3] Running Creativity...{RESET}")
        cr_prompt = build_creativity_prompt(task, curiosity_out)
        creativity_out = _call_stage(generate_fn, CREATIVITY_SYSTEM, cr_prompt, "creativity", verbose)
        if verbose:
            _print_creativity(creativity_out)

        # CRITIC
        if verbose:
            print(f"\n  {MAGENTA}[3/3] Running Critic...{RESET}")
        crit_prompt = build_critic_prompt(task, creativity_out)
        critic_out = _call_stage(generate_fn, CRITIC_SYSTEM, crit_prompt, "critic", verbose)
        if verbose:
            _print_critic(critic_out)

        loop_trace.append({
            "iteration": iteration,
            "curiosity": curiosity_out,
            "creativity": creativity_out,
            "critic": critic_out,
        })

        verdict = critic_out.get("verdict", "FAIL").upper()
        if verdict == "PASS":
            if verbose:
                print(f"\n  {GREEN}{BOLD}PASS on iteration {iteration}{RESET}")
            break

        critic_feedback = {
            "unexplored_directions": critic_out.get("unexplored_directions", []),
            "feedback_for_curiosity": critic_out.get("feedback_for_curiosity", []),
        }

    last_creativity = loop_trace[-1]["creativity"]
    return {
        "domain": domain,
        "input": task,
        "loop": loop_trace,
        "final_output": last_creativity.get("output", []),
    }


# ---------------------------------------------------------------------------
# Stage caller with JSON parsing + retry
# ---------------------------------------------------------------------------

def _call_stage(generate_fn, system_prompt, user_prompt, stage_name, verbose) -> dict:
    for attempt in range(1, MAX_RETRIES_PER_CALL + 1):
        try:
            raw = generate_fn(system_prompt, user_prompt)
            parsed = _extract_json(raw)
            if parsed is not None:
                parsed = _normalize_stage(stage_name, parsed)
                errors = _validate_stage(stage_name, parsed)
                if errors and verbose:
                    print(f"    {YELLOW}[warn] {stage_name}: {errors}{RESET}")
                return parsed
            # One repair pass before full retry/fallback.
            repaired = _repair_json(generate_fn, raw, stage_name, verbose)
            if repaired is not None:
                repaired = _normalize_stage(stage_name, repaired)
                errors = _validate_stage(stage_name, repaired)
                if errors and verbose:
                    print(f"    {YELLOW}[warn] repaired {stage_name}: {errors}{RESET}")
                return repaired
            if verbose:
                print(f"    {YELLOW}[retry {attempt}] no valid JSON{RESET}")
        except Exception as e:
            if verbose:
                print(f"    {RED}[error] attempt {attempt}: {e}{RESET}")
    if verbose:
        print(f"    {RED}[fallback] returning empty structure{RESET}")
    return _fallback(stage_name)


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for marker in ("```json", "```"):
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.find("```", start)
            if end > start:
                try:
                    return json.loads(text[start:end].strip())
                except json.JSONDecodeError:
                    pass
    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def _repair_json(generate_fn, raw_text: str, stage_name: str, verbose: bool) -> dict | None:
    """Ask the model to repair malformed output into valid JSON only."""
    try:
        repair_system = (
            "You repair malformed LLM output into valid JSON. "
            "Return ONLY a valid JSON object. "
            "Preserve meaning, compress wording, do not add markdown."
        )
        repair_user = (
            f"Stage: {stage_name}\n\n"
            "Convert the following malformed output into one valid JSON object.\n\n"
            f"{raw_text}"
        )
        repaired_raw = generate_fn(repair_system, repair_user)
        repaired = _extract_json(repaired_raw)
        if repaired is not None and verbose:
            print(f"    {GREY}[repair] recovered valid JSON for {stage_name}{RESET}")
        return repaired
    except Exception as e:
        if verbose:
            print(f"    {GREY}[repair-failed] {stage_name}: {e}{RESET}")
        return None


def _validate_stage(stage_name, data):
    if stage_name == "curiosity":  return validate_curiosity(data)
    if stage_name == "creativity": return validate_creativity(data)
    if stage_name == "critic":     return validate_critic(data)
    return []


def _normalize_stage(stage_name, data):
    if stage_name == "curiosity":  return normalize_curiosity(data)
    if stage_name == "creativity": return normalize_creativity(data)
    if stage_name == "critic":     return normalize_critic(data)
    return data


def _fallback(stage_name: str) -> dict:
    if stage_name == "curiosity":
        return {"hidden_assumptions": [], "unexplored_domains": [], "questions": [], "branch_seeds": ["explore differently"]}
    if stage_name == "creativity":
        return {"research": [], "branches": [], "pruned": [], "combinations": [], "dead_ends": [], "output": ["(generation failed)"]}
    return {"scores": [], "verdict": "FAIL", "unexplored_directions": ["retry"], "feedback_for_curiosity": ["previous attempt failed"]}


# ---------------------------------------------------------------------------
# Interactive TUI entry point
# ---------------------------------------------------------------------------

def _tui():
    """Interactive pipeline runner. No args needed."""
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(REPO_ROOT))

    from src.IV_inference.gemma4_integration import load_gemma4, MODELS as HF_MODELS
    from src.IV_inference.ollama_integration import load_ollama_gemma4, MODELS as OLLAMA_MODELS

    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"  {BOLD}Pipeline Runner{RESET}")
    print(f"  Curiosity -> Creativity -> Critic")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

    print(f"\n{YELLOW}Backend:{RESET}")
    print("  [1] Hugging Face transformers")
    print("  [2] Ollama local")
    backend = input("\n  Choice [1]: ").strip() or "1"

    if backend == "2":
        print(f"\n{YELLOW}Ollama model:{RESET}")
        aliases = list(OLLAMA_MODELS.keys())
        for i, alias in enumerate(aliases, 1):
            info = OLLAMA_MODELS[alias]
            default = f" {GREY}<- default{RESET}" if i == 1 else ""
            print(f"  [{i}] {alias:<5} {info['id']:<16} {info['description']}{default}")
        print("  [c] custom local tag")
        raw = input(f"\n  Choice [1]: ").strip().lower() or "1"
        if raw == "c":
            model_alias = input("  Custom Ollama tag: ").strip() or "gemma4:e2b"
        else:
            model_alias = aliases[int(raw) - 1] if raw.isdigit() and 1 <= int(raw) <= len(aliases) else aliases[0]
        generate_fn = load_ollama_gemma4(model=model_alias, thinking=False, use_memory=False)
    else:
        print(f"\n{YELLOW}Model:{RESET}")
        aliases = list(HF_MODELS.keys())
        for i, alias in enumerate(aliases, 1):
            info = HF_MODELS[alias]
            default = f" {GREY}<- default{RESET}" if i == 1 else ""
            print(f"  [{i}] {alias:<5} {info['ram_gb']:>3}GB  {info['description']}{default}")
        raw = input(f"\n  Choice [1]: ").strip() or "1"
        model_alias = aliases[int(raw) - 1] if raw.isdigit() and 1 <= int(raw) <= len(aliases) else aliases[0]
        generate_fn = load_gemma4(model=model_alias, thinking=False, use_memory=False)

    while True:
        print(f"\n{YELLOW}Enter task (or 'q' to quit):{RESET}")
        task = input("  > ").strip()
        if task.lower() in ("q", "quit", "exit", ""):
            break

        print(f"\n{YELLOW}Max iterations [2]:{RESET}")
        mi = input("  > ").strip() or "2"
        max_iters = int(mi) if mi.isdigit() else 2

        print(f"\n{YELLOW}Domain label [general]:{RESET}")
        domain = input("  > ").strip() or "general"

        start = time.time()
        result = run_loop(task=task, generate_fn=generate_fn, domain=domain, max_iterations=max_iters)
        elapsed = round(time.time() - start, 1)

        print(f"\n{BOLD}{GREEN}{'='*60}{RESET}")
        print(f"  {BOLD}FINAL OUTPUT{RESET}  ({elapsed}s, {len(result['loop'])} iteration(s))")
        print(f"{BOLD}{GREEN}{'='*60}{RESET}")
        for o in result.get("final_output", []):
            print(f"  {GREEN}* {o}{RESET}")

        # Always save; print absolute paths (many terminals cmd+click to open file)
        from src.V_utility.export import save_markdown_alongside
        out_path = REPO_ROOT / "data" / "output" / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n{GREEN}Saved JSON:{RESET} {out_path.resolve()}")
        md_path = save_markdown_alongside(out_path)
        print(f"{GREEN}Saved MD:{RESET}   {md_path.resolve()}")

        print(f"\n{GREY}Run another task or 'q' to quit.{RESET}")


if __name__ == "__main__":
    _tui()
