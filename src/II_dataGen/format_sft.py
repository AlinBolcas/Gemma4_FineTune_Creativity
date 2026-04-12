"""
format_sft.py - Convert pipeline JSONL into SFT chat-format training data.

Run interactively:
    python src/II_dataGen/format_sft.py
"""

import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "output"
INPUT_DIR = REPO_ROOT / "data" / "input"
TRAIN_DIR = INPUT_DIR / "train"
EVAL_DIR = INPUT_DIR / "eval"
TEST_DIR = INPUT_DIR / "test"

CYAN = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; GREY = "\033[90m"
BOLD = "\033[1m"; RESET = "\033[0m"

SYSTEM_PROMPT_MODES = {
    "none": "",
    "minimal": "You are a creative reasoning assistant.",
    "explicit": (
        "You are a creative reasoning assistant. "
        "Approach open-ended tasks with curiosity, branching, synthesis, and self-critique."
    ),
    "advanced": (
        "You are a creative reasoning assistant. "
        "Use a full staged process: curiosity map, question expansion, distillation, "
        "socratic steering, research planning, branch generation, branch development, "
        "selection, combinatory mixing, final synthesis, and critique."
    ),
}


def _is_advanced_example(example: dict) -> bool:
    if example.get("advanced_final"):
        return True
    if example.get("_meta", {}).get("runner") == "advanced":
        return True
    return any("advanced" in iteration for iteration in example.get("loop", []))


def _format_simple_iteration(iteration: dict) -> list[str]:
    parts = []

    cur = iteration.get("curiosity", {})
    parts.append("### Curiosity")
    if cur.get("hidden_assumptions"):
        parts.append("Hidden assumptions:")
        for item in cur["hidden_assumptions"]:
            parts.append(f"- {item}")
    if cur.get("questions"):
        parts.append("Key questions:")
        for item in cur["questions"]:
            qtext = item.get("question", "") if isinstance(item, dict) else str(item)
            parts.append(f"- {qtext}")
    if cur.get("branch_seeds"):
        parts.append(f"Branch seeds: {', '.join(cur['branch_seeds'])}")

    cre = iteration.get("creativity", {})
    parts.append("### Creativity")
    if cre.get("research"):
        parts.append("Research:")
        for item in cre["research"][:4]:
            parts.append(f"- {item}")
    if cre.get("branches"):
        parts.append("Branches:")
        for item in cre["branches"]:
            bid = item.get("id", "?")
            frame = item.get("frame", "")
            candidates = ", ".join(item.get("candidates", [])[:3])
            parts.append(f"- {bid}: {frame} [{candidates}]")
    if cre.get("pruned"):
        parts.append("Pruned:")
        for item in cre["pruned"]:
            parts.append(f"- {item.get('id', '?')}: {item.get('reason', '')}")
    if cre.get("combinations"):
        parts.append("Combinations:")
        for item in cre["combinations"]:
            parts.append(f"- {item.get('from', [])} -> {item.get('result', '')} ({item.get('novelty_note', '')})")
    if cre.get("output"):
        parts.append("Candidates:")
        for item in cre["output"]:
            parts.append(f"- {item}")

    cri = iteration.get("critic", {})
    parts.append("### Critic")
    verdict = cri.get("verdict", "?")
    parts.append(f"Verdict: {verdict}")
    if cri.get("scores"):
        for item in cri["scores"]:
            parts.append(
                f"- {item.get('candidate', '?')}: novelty={item.get('novelty', '?')}, "
                f"relevance={item.get('relevance', '?')} | {item.get('notes', '')}"
            )
    if verdict == "FAIL" and cri.get("feedback_for_curiosity"):
        parts.append("Feedback for next pass:")
        for item in cri["feedback_for_curiosity"]:
            parts.append(f"- {item}")

    return parts


def _format_advanced_iteration(iteration: dict) -> list[str]:
    parts = []
    advanced = iteration.get("advanced", {})

    cmap = advanced.get("curiosity_map", {})
    parts.append("### Curiosity Map")
    if cmap.get("global_novelty_estimate") is not None:
        parts.append(f"Novelty estimate: {cmap.get('global_novelty_estimate', '?')}/10")
    if cmap.get("branch_budget"):
        parts.append(f"Branch budget: {cmap.get('branch_budget')}")
    if cmap.get("known_context"):
        parts.append("Known context:")
        for item in cmap["known_context"]:
            parts.append(f"- {item}")
    if cmap.get("hidden_assumptions"):
        parts.append("Hidden assumptions:")
        for item in cmap["hidden_assumptions"]:
            parts.append(f"- {item}")
    if cmap.get("curiosity_domains"):
        parts.append("Curiosity domains:")
        for item in cmap["curiosity_domains"]:
            parts.append(
                f"- {item.get('id', '?')}: [{item.get('lens', '')}] {item.get('domain', '')} | "
                f"{item.get('novelty_opportunity', '')}"
            )
    if cmap.get("seed_questions"):
        parts.append("Seed questions:")
        for item in cmap["seed_questions"]:
            parts.append(f"- {item.get('id', '?')}: {item.get('question', '')}")

    cexpand = advanced.get("curiosity_expand", {})
    parts.append("### Curiosity Expand")
    for item in cexpand.get("expanded_branches", []):
        parts.append(
            f"- {item.get('id', '?')}: {item.get('direction', '')} | "
            f"strength={item.get('curiosity_strength', '?')}/10 | keep={item.get('keep', True)}"
        )
        for question in item.get("questions", []):
            parts.append(f"  - {question}")
    if cexpand.get("pruned_branches"):
        parts.append("Pruned curiosity branches:")
        for item in cexpand["pruned_branches"]:
            parts.append(f"- {item.get('id', '?')}: {item.get('reason', '')}")

    cdistill = advanced.get("curiosity_distill", {})
    parts.append("### Curiosity Distill")
    for item in cdistill.get("best_questions", []):
        parts.append(
            f"- {item.get('id', '?')}: {item.get('question', '')} | "
            f"leverage={item.get('leverage_score', '?')}/10 | {item.get('why_high_leverage', '')}"
        )
    if cdistill.get("socratic_scaffold"):
        parts.append("Socratic scaffold:")
        for item in cdistill["socratic_scaffold"]:
            parts.append(f"- {item}")
    if cdistill.get("exploration_direction"):
        parts.append(f"Exploration direction: {cdistill.get('exploration_direction')}")
    if cdistill.get("steering_signals"):
        parts.append("Steering signals:")
        for item in cdistill["steering_signals"]:
            parts.append(f"- {item}")

    socratic = advanced.get("socratic_output", {})
    parts.append("### Socratic Output")
    if socratic.get("question_set"):
        parts.append("Question set:")
        for item in socratic["question_set"]:
            parts.append(f"- {item}")
    if socratic.get("scaffold"):
        parts.append("Scaffold:")
        for item in socratic["scaffold"]:
            parts.append(f"- {item}")
    if socratic.get("direction"):
        parts.append(f"Direction: {socratic.get('direction')}")
    if socratic.get("constraints"):
        parts.append("Constraints:")
        for item in socratic["constraints"]:
            parts.append(f"- {item}")
    if socratic.get("novelty_focus"):
        parts.append("Novelty focus:")
        for item in socratic["novelty_focus"]:
            parts.append(f"- {item}")

    research = advanced.get("creativity_research_plan", {})
    parts.append("### Creativity Research Plan")
    if research.get("complexity") is not None:
        parts.append(f"Complexity: {research.get('complexity', '?')}/10")
    if research.get("branch_budget"):
        parts.append(f"Branch budget: {research.get('branch_budget')}")
    if research.get("known_patterns"):
        parts.append("Known patterns:")
        for item in research["known_patterns"]:
            parts.append(f"- {item}")
    if research.get("adjacent_domains"):
        parts.append("Adjacent domains:")
        for item in research["adjacent_domains"]:
            parts.append(f"- {item}")
    if research.get("creative_tensions"):
        parts.append("Creative tensions:")
        for item in research["creative_tensions"]:
            parts.append(f"- {item}")
    if research.get("research_queries"):
        parts.append("Research queries:")
        for item in research["research_queries"]:
            parts.append(f"- {item}")

    cbranch = advanced.get("creativity_branch", {})
    parts.append("### Creativity Branch")
    for item in cbranch.get("branches", []):
        examples = ", ".join(item.get("examples", [])[:3])
        parts.append(
            f"- {item.get('id', '?')}: {item.get('frame', '')} | domain={item.get('domain', '')} | "
            f"constraint={item.get('constraint', '')} | examples=[{examples}]"
        )
        if item.get("why_distinct"):
            parts.append(f"  - why distinct: {item.get('why_distinct')}")

    parts.append("### Develop Branches")
    for item in advanced.get("creativity_develop", []):
        parts.append(f"- {item.get('branch_id', '?')}")
        if item.get("chain_steps"):
            parts.append("  - chain steps:")
            for step in item["chain_steps"]:
                parts.append(f"    - {step}")
        if item.get("branch_outputs"):
            parts.append("  - branch outputs:")
            for output in item["branch_outputs"]:
                parts.append(f"    - {output}")
        if item.get("novelty_delta"):
            parts.append(f"  - novelty delta: {item.get('novelty_delta')}")
        if item.get("exhausted_when"):
            parts.append(f"  - exhausted when: {item.get('exhausted_when')}")

    selection = advanced.get("creativity_selection", {})
    parts.append("### Selection")
    for item in selection.get("scored_branches", []):
        parts.append(
            f"- {item.get('branch_id', '?')}: n={item.get('novelty', '?')}/10 "
            f"r={item.get('relevance', '?')}/10 c={item.get('combinability', '?')}/10 "
            f"| {item.get('decision', '?')} | {item.get('reason', '')}"
        )
    if selection.get("kept_branch_ids"):
        parts.append(f"Kept branches: {', '.join(selection['kept_branch_ids'])}")

    mixing = advanced.get("creativity_mixing", {})
    parts.append("### Combinatory Mixing")
    for item in mixing.get("hybrids", []):
        parts.append(
            f"- {item.get('id', '?')}: {item.get('from_branch_ids', [])} -> {item.get('concept', '')} "
            f"| strength={item.get('strength_score', '?')}/10 | {item.get('why_novel', '')}"
        )
    if mixing.get("dead_ends"):
        parts.append("Dead ends:")
        for item in mixing["dead_ends"]:
            parts.append(f"- {item.get('from_branch_ids', [])}: {item.get('reason', '')}")

    synthesis = advanced.get("creativity_final_synthesis", {})
    parts.append("### Final Synthesis")
    for item in synthesis.get("primary_candidates", []):
        parts.append(
            f"- {item.get('id', '?')}: {item.get('title', '')} | {item.get('concept', '')} | "
            f"from={item.get('built_from', [])}"
        )
        if item.get("novelty_notes"):
            parts.append(f"  - novelty: {item.get('novelty_notes')}")
    if synthesis.get("best_combination"):
        parts.append(f"Best combination: {synthesis.get('best_combination')}")
    if synthesis.get("output"):
        parts.append("Final candidates:")
        for item in synthesis["output"]:
            parts.append(f"- {item}")

    critic = iteration.get("critic", {})
    parts.append("### Critic")
    verdict = critic.get("verdict", "?")
    parts.append(f"Verdict: {verdict}")
    for item in critic.get("scores", []):
        candidate_label = item.get("candidate_id", item.get("candidate", "?"))
        parts.append(
            f"- {candidate_label}: novelty={item.get('novelty', '?')}, "
            f"relevance={item.get('relevance', '?')} | {item.get('notes', '')}"
        )
    if verdict == "FAIL":
        if critic.get("feedback_for_curiosity"):
            parts.append("Feedback for curiosity:")
            for item in critic["feedback_for_curiosity"]:
                parts.append(f"- {item}")
        if critic.get("feedback_for_creativity"):
            parts.append("Feedback for creativity:")
            for item in critic["feedback_for_creativity"]:
                parts.append(f"- {item}")

    return parts


def format_trace_as_text(example: dict) -> str:
    """Convert a full-loop example into the assistant's visible reasoning output."""
    parts = []
    advanced = _is_advanced_example(example)

    for iteration in example.get("loop", []):
        it_num = iteration.get("iteration", "?")
        parts.append(f"## Iteration {it_num}")
        parts.extend(_format_advanced_iteration(iteration) if advanced else _format_simple_iteration(iteration))
        parts.append("")

    # Final output
    final = example.get("final_output", [])
    if final:
        parts.append("## Final Output")
        for f in final:
            parts.append(f"- {f}")

    return "\n".join(parts)


def convert_to_chat_format(example: dict, system_mode: str = "minimal") -> dict:
    """Convert a full-loop example to a chat-format training row."""
    task = example.get("input", "")
    trace_text = format_trace_as_text(example)

    messages = []
    system_prompt = SYSTEM_PROMPT_MODES.get(system_mode, SYSTEM_PROMPT_MODES["minimal"])
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.extend(
        [
            {"role": "user", "content": task},
            {"role": "assistant", "content": trace_text},
        ]
    )

    return {"messages": messages}


def _ensure_dirs():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)


def _read_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def _write_jsonl(examples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def _combined_stem(paths: list[Path]) -> str:
    if len(paths) == 1:
        stem = paths[0].stem
        if stem.endswith("_sft"):
            stem = stem[:-4]
        return stem
    return "combined_dataset"


def _split_examples(examples: list[dict], train_ratio=0.9, eval_ratio=0.05, test_ratio=0.05, seed=42):
    # Deterministic shuffle so re-running on same source gives same split.
    items = list(examples)
    rng = random.Random(seed)
    rng.shuffle(items)

    n = len(items)
    if n == 0:
        return [], [], []
    if n == 1:
        return items, [], []
    if n == 2:
        return [items[0]], [items[1]], []

    train_n = max(1, int(n * train_ratio))
    eval_n = max(1, int(n * eval_ratio)) if n >= 10 else max(1, min(2, n - train_n))
    test_n = n - train_n - eval_n
    if test_n < 0:
        test_n = 0
        eval_n = max(0, n - train_n)

    # Ensure all examples are assigned.
    if train_n + eval_n + test_n < n:
        train_n += n - (train_n + eval_n + test_n)

    train = items[:train_n]
    eval_set = items[train_n:train_n + eval_n]
    test = items[train_n + eval_n:]
    return train, eval_set, test


def process_jsonl(
    input_path: Path,
    output_path: Path | None = None,
    split: bool = True,
    system_mode: str = "minimal",
):
    """Read raw pipeline JSONL, convert to SFT chat format, optionally create splits."""
    raw_examples = _read_jsonl(input_path)
    examples = [convert_to_chat_format(raw, system_mode=system_mode) for raw in raw_examples]

    _ensure_dirs()
    if output_path is None:
        output_path = INPUT_DIR / f"{input_path.stem}_sft.jsonl"

    _write_jsonl(examples, output_path)
    print(
        f"  {GREEN}Converted {len(examples)} examples -> "
        f"{output_path.relative_to(REPO_ROOT)}{RESET}"
    )
    print(f"  {GREEN}System prompt mode:{RESET} {system_mode}")

    if split:
        train, eval_set, test = _split_examples(examples)
        base = input_path.stem
        train_path = TRAIN_DIR / f"{base}_train.jsonl"
        eval_path = EVAL_DIR / f"{base}_eval.jsonl"
        test_path = TEST_DIR / f"{base}_test.jsonl"
        _write_jsonl(train, train_path)
        _write_jsonl(eval_set, eval_path)
        _write_jsonl(test, test_path)
        print(f"  {GREEN}Train:{RESET} {train_path.relative_to(REPO_ROOT)} ({len(train)})")
        print(f"  {GREEN}Eval:{RESET}  {eval_path.relative_to(REPO_ROOT)} ({len(eval_set)})")
        print(f"  {GREEN}Test:{RESET}  {test_path.relative_to(REPO_ROOT)} ({len(test)})")

    return examples


def _list_candidate_files() -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return [
        p for p in sorted(OUTPUT_DIR.glob("*.jsonl"))
        if not p.name.endswith("_sft.jsonl")
    ]


def _pick_files() -> list[Path]:
    files = _list_candidate_files()
    if not files:
        print(f"  {YELLOW}No source JSONL files found in data/output/.{RESET}")
        return []

    print(f"\n{YELLOW}Format mode:{RESET}")
    print("  [1] one file")
    print("  [2] multiple files")
    print("  [3] all files")
    mode = input("\n  Choice [1]: ").strip() or "1"

    print(f"\n{YELLOW}Available JSONL files:{RESET}")
    for i, path in enumerate(files, 1):
        print(f"  [{i}] {path.name}")

    if mode == "3":
        return files

    if mode == "2":
        raw = input("\n  Pick multiple (e.g. 1,2,4) [1]: ").strip() or "1"
        picks = []
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                idx = int(part)
                if 1 <= idx <= len(files):
                    picks.append(files[idx - 1])
        seen = set()
        result = []
        for path in picks:
            if path not in seen:
                seen.add(path)
                result.append(path)
        return result

    raw = input("\n  Choice [1]: ").strip() or "1"
    if raw.isdigit() and 1 <= int(raw) <= len(files):
        return [files[int(raw) - 1]]
    return [files[0]]


def _tui():
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"  {BOLD}SFT Formatter{RESET}")
    print("  Raw pipeline JSONL -> Gemma / Unsloth training format")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

    files = _pick_files()
    if not files:
        return

    print(f"\n{YELLOW}System prompt in training rows:{RESET}")
    print("  [1] minimal  <- default")
    print("  [2] none")
    print("  [3] explicit")
    print("  [4] advanced")
    mode_raw = input("\n  Choice [1]: ").strip() or "1"
    system_mode = {"1": "minimal", "2": "none", "3": "explicit", "4": "advanced"}.get(mode_raw, "minimal")

    print(f"\n{YELLOW}Create train/eval/test splits? [Y/n]{RESET}")
    do_split = input("  > ").strip().lower() not in ("n", "no")

    raw_examples = []
    for path in files:
        raw_examples.extend(_read_jsonl(path))

    if not raw_examples:
        print(f"  {YELLOW}No examples found in selected files.{RESET}")
        return

    combined_path = INPUT_DIR / f"{_combined_stem(files)}_sft.jsonl"
    examples = [convert_to_chat_format(raw, system_mode=system_mode) for raw in raw_examples]
    _ensure_dirs()
    _write_jsonl(examples, combined_path)
    print(f"  {GREEN}Combined:{RESET} {combined_path.relative_to(REPO_ROOT)} ({len(examples)})")
    print(f"  {GREEN}System prompt mode:{RESET} {system_mode}")

    if do_split:
        train, eval_set, test = _split_examples(examples)
        base = _combined_stem(files)
        train_path = TRAIN_DIR / f"{base}_train.jsonl"
        eval_path = EVAL_DIR / f"{base}_eval.jsonl"
        test_path = TEST_DIR / f"{base}_test.jsonl"
        _write_jsonl(train, train_path)
        _write_jsonl(eval_set, eval_path)
        _write_jsonl(test, test_path)
        print(f"  {GREEN}Train:{RESET} {train_path.relative_to(REPO_ROOT)} ({len(train)})")
        print(f"  {GREEN}Eval:{RESET}  {eval_path.relative_to(REPO_ROOT)} ({len(eval_set)})")
        print(f"  {GREEN}Test:{RESET}  {test_path.relative_to(REPO_ROOT)} ({len(test)})")

    print(f"\n  {GREEN}Done.{RESET}")


if __name__ == "__main__":
    _tui()
