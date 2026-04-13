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


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _format_simple_iteration(iteration: dict, include_critic: bool = False) -> list[str]:
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

    if include_critic:
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


def _format_advanced_iteration(iteration: dict, include_critic: bool = False) -> list[str]:
    parts = []
    advanced = iteration.get("advanced", {})

    cmap = advanced.get("curiosity_map", {})
    cexpand = advanced.get("curiosity_expand", {})
    cdistill = advanced.get("curiosity_distill", {})
    socratic = advanced.get("socratic_output", {})

    research = advanced.get("creativity_research_plan", {})
    cbranch = advanced.get("creativity_branch", {})
    selection = advanced.get("creativity_selection", {})
    mixing = advanced.get("creativity_mixing", {})
    synthesis = advanced.get("creativity_final_synthesis", {})
    # Canonical training shape for both simple and advanced traces.
    parts.append("### Curiosity")
    if cmap.get("hidden_assumptions"):
        parts.append("Hidden assumptions:")
        for item in cmap["hidden_assumptions"]:
            parts.append(f"- {item}")

    questions = []
    for item in cdistill.get("best_questions", []):
        if isinstance(item, dict) and item.get("question"):
            questions.append(item.get("question"))
    for item in cmap.get("seed_questions", []):
        if isinstance(item, dict) and item.get("question"):
            questions.append(item.get("question"))
    for item in socratic.get("question_set", []):
        questions.append(item)
    questions = _dedupe_keep_order(questions)[:6]
    if questions:
        parts.append("Key questions:")
        for item in questions:
            parts.append(f"- {item}")

    branch_seeds = []
    for item in cexpand.get("expanded_branches", []):
        direction = item.get("direction", "")
        if direction:
            branch_seeds.append(direction)
        for question in item.get("questions", []):
            branch_seeds.append(question)
    branch_seeds.extend(cdistill.get("steering_signals", []))
    branch_seeds.extend(socratic.get("novelty_focus", []))
    branch_seeds = _dedupe_keep_order(branch_seeds)[:8]
    if branch_seeds:
        parts.append(f"Branch seeds: {', '.join(branch_seeds)}")

    parts.append("### Creativity")
    research_items = []
    research_items.extend(research.get("known_patterns", []))
    research_items.extend(research.get("adjacent_domains", []))
    research_items.extend(research.get("creative_tensions", []))
    research_items.extend(research.get("research_queries", []))
    research_items = _dedupe_keep_order(research_items)[:6]
    if research_items:
        parts.append("Research:")
        for item in research_items:
            parts.append(f"- {item}")

    branch_lines = []
    for item in cbranch.get("branches", []):
        bid = item.get("id", "?")
        frame = item.get("frame", "")
        examples = item.get("examples", [])[:3]
        candidates = ", ".join(examples)
        branch_lines.append(f"- {bid}: {frame} [{candidates}]")
    if not branch_lines:
        for item in advanced.get("creativity_develop", []):
            outputs = ", ".join(item.get("branch_outputs", [])[:3])
            branch_lines.append(f"- {item.get('branch_id', '?')}: Developed branch [{outputs}]")
    if branch_lines:
        parts.append("Branches:")
        parts.extend(branch_lines[:8])

    pruned_lines = []
    for item in selection.get("scored_branches", []):
        decision = str(item.get("decision", "")).lower()
        if decision and decision not in ("keep", "kept", "selected"):
            pruned_lines.append(f"- {item.get('branch_id', '?')}: {item.get('reason', '')}")
    if pruned_lines:
        parts.append("Pruned:")
        parts.extend(_dedupe_keep_order(pruned_lines)[:6])

    combination_lines = []
    for item in mixing.get("hybrids", []):
        combination_lines.append(
            f"- {item.get('from_branch_ids', [])} -> {item.get('concept', '')} ({item.get('why_novel', '')})"
        )
    if synthesis.get("best_combination"):
        combination_lines.append(f"- best -> {synthesis.get('best_combination')} ()")
    if combination_lines:
        parts.append("Combinations:")
        parts.extend(_dedupe_keep_order(combination_lines)[:6])

    candidate_items = []
    for item in synthesis.get("output", []):
        candidate_items.append(item)
    for item in synthesis.get("primary_candidates", []):
        title = item.get("title", "")
        concept = item.get("concept", "")
        candidate_items.append(title or concept)
    candidate_items = _dedupe_keep_order(candidate_items)[:6]
    if candidate_items:
        parts.append("Candidates:")
        for item in candidate_items:
            parts.append(f"- {item}")

    if include_critic:
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


def _passed_iterations(example: dict) -> list[dict]:
    passed = []
    for iteration in example.get("loop", []):
        verdict = str(iteration.get("critic", {}).get("verdict", "")).upper().strip()
        if verdict == "PASS":
            passed.append(iteration)
    return passed


def _select_training_iterations(example: dict, pass_only: bool = True, last_pass_only: bool = True) -> list[dict]:
    loop = example.get("loop", []) or []
    if not loop:
        return []

    passed = _passed_iterations(example)
    if pass_only:
        if not passed:
            return []
        return [passed[-1]] if last_pass_only else passed

    return [loop[-1]] if last_pass_only else loop


def _has_validation_errors(example: dict) -> bool:
    errors = example.get("_meta", {}).get("validation_errors", [])
    return bool(errors)


def format_trace_as_text(
    example: dict,
    include_critic: bool = False,
    pass_only: bool = True,
    last_pass_only: bool = True,
) -> str:
    """Convert a full-loop example into the assistant's visible reasoning output."""
    parts = []
    advanced = _is_advanced_example(example)
    iterations = _select_training_iterations(example, pass_only=pass_only, last_pass_only=last_pass_only)

    for iteration in iterations:
        it_num = iteration.get("iteration", "?")
        parts.append(f"## Iteration {it_num}")
        parts.extend(
            _format_advanced_iteration(iteration, include_critic=include_critic)
            if advanced
            else _format_simple_iteration(iteration, include_critic=include_critic)
        )
        parts.append("")

    # Final output
    final = example.get("final_output", [])
    if final:
        parts.append("## Final Output")
        for f in final:
            parts.append(f"- {f}")

    return "\n".join(parts).strip()


def convert_to_chat_format(
    example: dict,
    system_mode: str = "minimal",
    include_critic: bool = False,
    pass_only: bool = True,
    last_pass_only: bool = True,
) -> dict | None:
    """Convert a full-loop example to a chat-format training row."""
    task = example.get("input", "")
    trace_text = format_trace_as_text(
        example,
        include_critic=include_critic,
        pass_only=pass_only,
        last_pass_only=last_pass_only,
    )
    if not task or not trace_text:
        return None

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
    include_critic: bool = False,
    pass_only: bool = True,
    last_pass_only: bool = True,
    skip_invalid: bool = True,
):
    """Read raw pipeline JSONL, convert to SFT chat format, optionally create splits."""
    raw_examples = _read_jsonl(input_path)
    examples = []
    skipped_no_pass = 0
    skipped_invalid = 0
    for raw in raw_examples:
        if skip_invalid and _has_validation_errors(raw):
            skipped_invalid += 1
            continue
        row = convert_to_chat_format(
            raw,
            system_mode=system_mode,
            include_critic=include_critic,
            pass_only=pass_only,
            last_pass_only=last_pass_only,
        )
        if row is None:
            skipped_no_pass += 1
            continue
        examples.append(row)

    _ensure_dirs()
    if output_path is None:
        output_path = INPUT_DIR / f"{input_path.stem}_sft.jsonl"

    _write_jsonl(examples, output_path)
    print(
        f"  {GREEN}Converted {len(examples)} examples -> "
        f"{output_path.relative_to(REPO_ROOT)}{RESET}"
    )
    if skipped_invalid:
        print(f"  {YELLOW}Skipped invalid:{RESET} {skipped_invalid} example(s) with validation errors")
    if skipped_no_pass:
        print(f"  {YELLOW}Skipped no PASS:{RESET} {skipped_no_pass} example(s) without a PASS iteration")
    print(f"  {GREEN}System prompt mode:{RESET} {system_mode}")
    print(f"  {GREEN}Training trace mode:{RESET} last PASS only, critic removed, invalid rows skipped")

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
    examples = []
    skipped_no_pass = 0
    skipped_invalid = 0
    for raw in raw_examples:
        if _has_validation_errors(raw):
            skipped_invalid += 1
            continue
        row = convert_to_chat_format(
            raw,
            system_mode=system_mode,
            include_critic=False,
            pass_only=True,
            last_pass_only=True,
        )
        if row is None:
            skipped_no_pass += 1
            continue
        examples.append(row)
    _ensure_dirs()
    _write_jsonl(examples, combined_path)
    print(f"  {GREEN}Combined:{RESET} {combined_path.relative_to(REPO_ROOT)} ({len(examples)})")
    if skipped_invalid:
        print(f"  {YELLOW}Skipped invalid:{RESET} {skipped_invalid} example(s) with validation errors")
    if skipped_no_pass:
        print(f"  {YELLOW}Skipped no PASS:{RESET} {skipped_no_pass} example(s) without a PASS iteration")
    print(f"  {GREEN}System prompt mode:{RESET} {system_mode}")
    print(f"  {GREEN}Training trace mode:{RESET} last PASS only, critic removed, invalid rows skipped")

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
