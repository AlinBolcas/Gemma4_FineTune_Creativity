"""
runner_advanced.py - Full step-by-step curiosity -> creativity -> critic runner.

Run interactively:
    python src/I_pipeline/runner_advanced.py
    # or from repo root:
    python -m src.I_pipeline.runner_advanced
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

# Fix direct-file execution.
if __name__ == "__main__" and not __package__:
    _repo = Path(__file__).resolve().parents[2]
    _repo_str = str(_repo)
    if _repo_str not in sys.path:
        sys.path.insert(0, _repo_str)
    __package__ = "src.I_pipeline"

from .prompts_advanced import (
    CREATIVITY_BRANCH_SYSTEM,
    CREATIVITY_DEVELOP_BRANCH_SYSTEM,
    CREATIVITY_FINAL_SYNTHESIS_SYSTEM,
    CREATIVITY_MIXING_SYSTEM,
    CREATIVITY_RESEARCH_PLAN_SYSTEM,
    CREATIVITY_SELECTION_SYSTEM,
    CRITIC_ADVANCED_SYSTEM,
    CURIOSITY_DISTILL_SYSTEM,
    CURIOSITY_EXPAND_SYSTEM,
    CURIOSITY_MAP_SYSTEM,
    CURIOSITY_MAP_SYSTEM_WITH_FEEDBACK,
    SOCRATIC_OUTPUT_SYSTEM,
    build_creativity_branch_prompt,
    build_creativity_develop_branch_prompt,
    build_creativity_final_synthesis_prompt,
    build_creativity_mixing_prompt,
    build_creativity_research_plan_prompt,
    build_creativity_selection_prompt,
    build_critic_advanced_prompt,
    build_curiosity_distill_prompt,
    build_curiosity_expand_prompt,
    build_curiosity_map_prompt,
    build_curiosity_packet,
    build_socratic_output_prompt,
)
from .schema_advanced import (
    fallback_stage,
    normalize_stage,
    summarize_creativity_trace,
    summarize_curiosity_trace,
    validate_full_loop,
    validate_stage,
)


# ---------------------------------------------------------------------------
# Terminal colors
# ---------------------------------------------------------------------------

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
GREY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"

MAX_LOOP_ITERATIONS = 2
MAX_RETRIES_PER_CALL = 2


# ---------------------------------------------------------------------------
# Lightweight debug printers
# ---------------------------------------------------------------------------

def _print_stage_header(label: str, color: str):
    print(f"\n  {color}{BOLD}{label}{RESET}")


def _print_curiosity_snapshot(curiosity_map: dict, socratic_output: dict):
    print(f"{CYAN}    Domains: {', '.join(d.get('domain', '') for d in curiosity_map.get('curiosity_domains', [])[:4])}{RESET}")
    for question in socratic_output.get("question_set", [])[:4]:
        print(f"{CYAN}    ? {question}{RESET}")


def _print_branch_snapshot(branch_plan: dict, developed_branches: list[dict]):
    developed_by_id = {d.get("branch_id"): d for d in developed_branches}
    for branch in branch_plan.get("branches", []):
        outputs = developed_by_id.get(branch.get("id"), {}).get("branch_outputs", [])
        preview = ", ".join(outputs[:2]) if outputs else "(no outputs)"
        print(f"{GREEN}    {branch.get('id', '?')}: {branch.get('frame', '')} -> {preview}{RESET}")


def _print_selection_snapshot(selection: dict, mixing: dict, final_synthesis: dict):
    kept = ", ".join(selection.get("kept_branch_ids", []))
    print(f"{GREEN}    Kept: {kept or '(none)'}{RESET}")
    for hybrid in mixing.get("hybrids", [])[:3]:
        print(f"{GREEN}    + {hybrid.get('from_branch_ids', [])} -> {hybrid.get('concept', '')}{RESET}")
    print(f"{GREEN}{BOLD}    Output:{RESET}")
    for item in final_synthesis.get("output", [])[:5]:
        print(f"{GREEN}    * {item}{RESET}")


def _print_critic_snapshot(critic: dict):
    verdict = critic.get("verdict", "?").upper()
    color = GREEN if verdict == "PASS" else RED
    print(f"{MAGENTA}{BOLD}    Critic: {color}{verdict}{RESET}")
    for score in critic.get("scores", [])[:4]:
        print(
            f"{MAGENTA}    {score.get('candidate_id', '?')}: "
            f"n={score.get('novelty', '?')}/10 "
            f"r={score.get('relevance', '?')}/10{RESET}"
        )


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def run_advanced_loop(
    task: str,
    generate_fn: Callable[[str, str], str],
    domain: str = "general",
    max_iterations: int = MAX_LOOP_ITERATIONS,
    verbose: bool = True,
) -> dict:
    loop_trace = []
    critic_feedback = None
    total_calls = 0
    started_at = time.time()

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"\n{BOLD}{'=' * 72}{RESET}")
            print(f"  {BOLD}Advanced Iteration {iteration}/{max_iterations}{RESET}")
            print(f"{'=' * 72}")

        if verbose:
            _print_stage_header("[1/11] Curiosity map", CYAN)
        curiosity_map, call_count = _call_stage(
            generate_fn=generate_fn,
            system_prompt=CURIOSITY_MAP_SYSTEM_WITH_FEEDBACK if critic_feedback else CURIOSITY_MAP_SYSTEM,
            user_prompt=build_curiosity_map_prompt(task, critic_feedback),
            stage_name="curiosity_map",
            verbose=verbose,
        )
        total_calls += call_count

        if verbose:
            _print_stage_header("[2/11] Curiosity expand", CYAN)
        curiosity_expand, call_count = _call_stage(
            generate_fn=generate_fn,
            system_prompt=CURIOSITY_EXPAND_SYSTEM,
            user_prompt=build_curiosity_expand_prompt(task, curiosity_map),
            stage_name="curiosity_expand",
            verbose=verbose,
        )
        total_calls += call_count

        if verbose:
            _print_stage_header("[3/11] Curiosity distill", CYAN)
        curiosity_distill, call_count = _call_stage(
            generate_fn=generate_fn,
            system_prompt=CURIOSITY_DISTILL_SYSTEM,
            user_prompt=build_curiosity_distill_prompt(task, curiosity_map, curiosity_expand),
            stage_name="curiosity_distill",
            verbose=verbose,
        )
        total_calls += call_count

        if verbose:
            _print_stage_header("[4/11] Socratic output", CYAN)
        socratic_output, call_count = _call_stage(
            generate_fn=generate_fn,
            system_prompt=SOCRATIC_OUTPUT_SYSTEM,
            user_prompt=build_socratic_output_prompt(task, curiosity_map, curiosity_expand, curiosity_distill),
            stage_name="socratic_output",
            verbose=verbose,
        )
        total_calls += call_count

        curiosity_packet = build_curiosity_packet(
            curiosity_map=curiosity_map,
            curiosity_expand=curiosity_expand,
            curiosity_distill=curiosity_distill,
            socratic_output=socratic_output,
        )

        if verbose:
            _print_curiosity_snapshot(curiosity_map, socratic_output)

        if verbose:
            _print_stage_header("[5/11] Creativity research plan", GREEN)
        creativity_research_plan, call_count = _call_stage(
            generate_fn=generate_fn,
            system_prompt=CREATIVITY_RESEARCH_PLAN_SYSTEM,
            user_prompt=build_creativity_research_plan_prompt(task, curiosity_packet),
            stage_name="creativity_research_plan",
            verbose=verbose,
        )
        total_calls += call_count

        if verbose:
            _print_stage_header("[6/11] Creativity branch", GREEN)
        creativity_branch, call_count = _call_stage(
            generate_fn=generate_fn,
            system_prompt=CREATIVITY_BRANCH_SYSTEM,
            user_prompt=build_creativity_branch_prompt(task, curiosity_packet, creativity_research_plan),
            stage_name="creativity_branch",
            verbose=verbose,
        )
        total_calls += call_count

        developed_branches = []
        branch_budget = creativity_research_plan.get("branch_budget", 3)
        planned_branches = creativity_branch.get("branches", [])[:branch_budget]

        if verbose:
            _print_stage_header("[7/11] Develop each branch", GREEN)
        for idx, branch in enumerate(planned_branches, 1):
            if verbose:
                print(f"{GREY}    Developing {branch.get('id', '?')} ({idx}/{len(planned_branches)}){RESET}")
            developed, call_count = _call_stage(
                generate_fn=generate_fn,
                system_prompt=CREATIVITY_DEVELOP_BRANCH_SYSTEM,
                user_prompt=build_creativity_develop_branch_prompt(
                    task,
                    curiosity_packet,
                    creativity_research_plan,
                    branch,
                ),
                stage_name="creativity_develop_branch",
                verbose=verbose,
                expected_branch_id=branch.get("id"),
            )
            total_calls += call_count
            developed_branches.append(developed)

        if verbose:
            _print_stage_header("[8/11] Selection and pruning", GREEN)
        creativity_selection, call_count = _call_stage(
            generate_fn=generate_fn,
            system_prompt=CREATIVITY_SELECTION_SYSTEM,
            user_prompt=build_creativity_selection_prompt(
                task,
                curiosity_packet,
                creativity_research_plan,
                creativity_branch,
                developed_branches,
            ),
            stage_name="creativity_selection",
            verbose=verbose,
        )
        total_calls += call_count

        if verbose:
            _print_stage_header("[9/11] Combinatory mixing", GREEN)
        creativity_mixing, call_count = _call_stage(
            generate_fn=generate_fn,
            system_prompt=CREATIVITY_MIXING_SYSTEM,
            user_prompt=build_creativity_mixing_prompt(
                task,
                curiosity_packet,
                creativity_research_plan,
                creativity_selection,
                developed_branches,
            ),
            stage_name="creativity_mixing",
            verbose=verbose,
        )
        total_calls += call_count

        if verbose:
            _print_stage_header("[10/11] Final synthesis", GREEN)
        creativity_final_synthesis, call_count = _call_stage(
            generate_fn=generate_fn,
            system_prompt=CREATIVITY_FINAL_SYNTHESIS_SYSTEM,
            user_prompt=build_creativity_final_synthesis_prompt(
                task,
                curiosity_packet,
                creativity_research_plan,
                creativity_selection,
                creativity_mixing,
                developed_branches,
            ),
            stage_name="creativity_final_synthesis",
            verbose=verbose,
        )
        total_calls += call_count

        if verbose:
            _print_branch_snapshot(creativity_branch, developed_branches)
            _print_selection_snapshot(creativity_selection, creativity_mixing, creativity_final_synthesis)

        if verbose:
            _print_stage_header("[11/11] Critic", MAGENTA)
        critic_advanced, call_count = _call_stage(
            generate_fn=generate_fn,
            system_prompt=CRITIC_ADVANCED_SYSTEM,
            user_prompt=build_critic_advanced_prompt(
                task,
                curiosity_packet,
                creativity_research_plan,
                creativity_selection,
                creativity_final_synthesis,
            ),
            stage_name="critic_advanced",
            verbose=verbose,
        )
        total_calls += call_count

        if verbose:
            _print_critic_snapshot(critic_advanced)

        curiosity_summary = summarize_curiosity_trace(
            curiosity_map=curiosity_map,
            curiosity_expand=curiosity_expand,
            curiosity_distill=curiosity_distill,
            socratic_output=socratic_output,
        )
        creativity_summary = summarize_creativity_trace(
            creativity_research_plan=creativity_research_plan,
            creativity_branch=creativity_branch,
            creativity_develop=developed_branches,
            creativity_selection=creativity_selection,
            creativity_mixing=creativity_mixing,
            creativity_final_synthesis=creativity_final_synthesis,
        )

        loop_trace.append(
            {
                "iteration": iteration,
                "curiosity": curiosity_summary,
                "creativity": creativity_summary,
                "critic": critic_advanced,
                "advanced": {
                    "curiosity_map": curiosity_map,
                    "curiosity_expand": curiosity_expand,
                    "curiosity_distill": curiosity_distill,
                    "socratic_output": socratic_output,
                    "creativity_research_plan": creativity_research_plan,
                    "creativity_branch": creativity_branch,
                    "creativity_develop": developed_branches,
                    "creativity_selection": creativity_selection,
                    "creativity_mixing": creativity_mixing,
                    "creativity_final_synthesis": creativity_final_synthesis,
                },
            }
        )

        if critic_advanced.get("verdict", "FAIL").upper() == "PASS":
            if verbose:
                print(f"\n  {GREEN}{BOLD}PASS on iteration {iteration}{RESET}")
            break

        critic_feedback = {
            "unexplored_directions": critic_advanced.get("unexplored_directions", []),
            "feedback_for_curiosity": critic_advanced.get("feedback_for_curiosity", []),
            "feedback_for_creativity": critic_advanced.get("feedback_for_creativity", []),
        }

    final_synthesis = loop_trace[-1]["advanced"]["creativity_final_synthesis"]
    result = {
        "domain": domain,
        "input": task,
        "loop": loop_trace,
        "final_output": final_synthesis.get("output", []),
        "advanced_final": final_synthesis,
        "_meta": {
            "runner": "advanced",
            "timestamp": datetime.now().isoformat(),
            "elapsed_sec": round(time.time() - started_at, 1),
            "llm_calls": total_calls,
        },
    }
    result["_validation"] = validate_full_loop(result)
    return result


# ---------------------------------------------------------------------------
# Stage caller with JSON parsing + repair
# ---------------------------------------------------------------------------

def _call_stage(
    generate_fn: Callable[[str, str], str],
    system_prompt: str,
    user_prompt: str,
    stage_name: str,
    verbose: bool,
    expected_branch_id: str | None = None,
) -> tuple[dict, int]:
    calls_used = 0
    for attempt in range(1, MAX_RETRIES_PER_CALL + 1):
        try:
            calls_used += 1
            raw = generate_fn(system_prompt, user_prompt)
            parsed = _extract_json(raw)
            if parsed is not None:
                parsed = normalize_stage(stage_name, parsed, expected_branch_id=expected_branch_id)
                errors = validate_stage(stage_name, parsed)
                if errors and verbose:
                    print(f"    {YELLOW}[warn] {stage_name}: {errors}{RESET}")
                return parsed, calls_used

            calls_used += 1
            repaired = _repair_json(generate_fn, raw, stage_name, verbose)
            if repaired is not None:
                repaired = normalize_stage(stage_name, repaired, expected_branch_id=expected_branch_id)
                errors = validate_stage(stage_name, repaired)
                if errors and verbose:
                    print(f"    {YELLOW}[warn] repaired {stage_name}: {errors}{RESET}")
                return repaired, calls_used

            if verbose:
                print(f"    {YELLOW}[retry {attempt}] no valid JSON{RESET}")
        except Exception as exc:
            if verbose:
                print(f"    {RED}[error] attempt {attempt}: {exc}{RESET}")

    if verbose:
        print(f"    {RED}[fallback] returning fallback for {stage_name}{RESET}")
    return fallback_stage(stage_name, expected_branch_id=expected_branch_id), calls_used


def _extract_json(text: str) -> dict | None:
    text = (text or "").strip()
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
        for idx in range(brace_start, len(text)):
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start : idx + 1])
                    except json.JSONDecodeError:
                        break
    return None


def _repair_json(
    generate_fn: Callable[[str, str], str],
    raw_text: str,
    stage_name: str,
    verbose: bool,
) -> dict | None:
    """One repair pass keeps runs resilient without changing stage semantics."""
    try:
        repair_system = (
            "You repair malformed LLM output into valid JSON. "
            "Return ONLY one valid JSON object. "
            "Preserve meaning. Do not add markdown."
        )
        repair_user = (
            f"Stage: {stage_name}\n\n"
            "Convert this malformed output into one valid JSON object:\n\n"
            f"{raw_text}"
        )
        repaired_raw = generate_fn(repair_system, repair_user)
        repaired = _extract_json(repaired_raw)
        if repaired is not None and verbose:
            print(f"    {GREY}[repair] recovered valid JSON for {stage_name}{RESET}")
        return repaired
    except Exception as exc:
        if verbose:
            print(f"    {GREY}[repair-failed] {stage_name}: {exc}{RESET}")
        return None


# ---------------------------------------------------------------------------
# Interactive TUI
# ---------------------------------------------------------------------------

def _tui():
    """Interactive advanced runner. No CLI args needed."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(repo_root))

    from src.IV_inference.gemma4_integration import MODELS as HF_MODELS
    from src.IV_inference.gemma4_integration import load_gemma4
    from src.IV_inference.ollama_integration import MODELS as OLLAMA_MODELS
    from src.IV_inference.ollama_integration import load_ollama_gemma4
    from src.V_utility.export import save_markdown_alongside

    print(f"\n{BOLD}{CYAN}{'=' * 72}{RESET}")
    print(f"  {BOLD}Advanced Pipeline Runner{RESET}")
    print("  Full step-by-step curiosity and creativity architecture")
    print(f"{BOLD}{CYAN}{'=' * 72}{RESET}")

    print(f"\n{YELLOW}Backend:{RESET}")
    print("  [1] Hugging Face transformers")
    print("  [2] Ollama local")
    backend = input("\n  Choice [1]: ").strip() or "1"

    if backend == "2":
        print(f"\n{YELLOW}Ollama model:{RESET}")
        aliases = list(OLLAMA_MODELS.keys())
        for idx, alias in enumerate(aliases, 1):
            info = OLLAMA_MODELS[alias]
            default = f" {GREY}<- default{RESET}" if idx == 1 else ""
            print(f"  [{idx}] {alias:<5} {info['id']:<16} {info['description']}{default}")
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
        for idx, alias in enumerate(aliases, 1):
            info = HF_MODELS[alias]
            default = f" {GREY}<- default{RESET}" if idx == 1 else ""
            print(f"  [{idx}] {alias:<5} {info['ram_gb']:>3}GB  {info['description']}{default}")
        raw = input(f"\n  Choice [1]: ").strip() or "1"
        model_alias = aliases[int(raw) - 1] if raw.isdigit() and 1 <= int(raw) <= len(aliases) else aliases[0]
        generate_fn = load_gemma4(model=model_alias, thinking=False, use_memory=False)

    while True:
        print(f"\n{YELLOW}Enter task (or 'q' to quit):{RESET}")
        task = input("  > ").strip()
        if task.lower() in ("q", "quit", "exit", ""):
            break

        print(f"\n{YELLOW}Max iterations [1]:{RESET}")
        raw_iters = input("  > ").strip() or "1"
        max_iters = int(raw_iters) if raw_iters.isdigit() else 1

        print(f"\n{YELLOW}Domain label [general]:{RESET}")
        domain = input("  > ").strip() or "general"

        started_at = time.time()
        result = run_advanced_loop(
            task=task,
            generate_fn=generate_fn,
            domain=domain,
            max_iterations=max_iters,
            verbose=True,
        )
        elapsed = round(time.time() - started_at, 1)

        print(f"\n{BOLD}{GREEN}{'=' * 72}{RESET}")
        print(
            f"  {BOLD}FINAL OUTPUT{RESET}  "
            f"({elapsed}s, {len(result['loop'])} iteration(s), {result.get('_meta', {}).get('llm_calls', 0)} calls)"
        )
        print(f"{BOLD}{GREEN}{'=' * 72}{RESET}")
        for item in result.get("final_output", []):
            print(f"  {GREEN}* {item}{RESET}")

        out_path = repo_root / "data" / "output" / f"pipeline_advanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)

        print(f"\n{GREEN}Saved JSON:{RESET} {out_path.resolve()}")
        md_path = save_markdown_alongside(out_path)
        print(f"{GREEN}Saved MD:{RESET}   {md_path.resolve()}")

        if result.get("_validation"):
            print(f"{YELLOW}Validation warnings:{RESET} {result['_validation']}")

        print(f"\n{GREY}Run another task or 'q' to quit.{RESET}")


if __name__ == "__main__":
    _tui()
