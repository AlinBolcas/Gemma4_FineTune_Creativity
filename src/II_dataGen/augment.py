"""
augment.py - Expand raw pipeline JSONL with Gemma 4.

Run interactively:
    python src/II_dataGen/augment.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "output"
DEBUG_DIR = OUTPUT_DIR / "_augment_debug"

sys.path.insert(0, str(REPO_ROOT))

from src.I_pipeline.runner import run_loop
from src.I_pipeline.runner_advanced import run_advanced_loop
from src.I_pipeline.schema import validate_full_loop as validate_simple_full_loop
from src.I_pipeline.schema_advanced import validate_full_loop as validate_advanced_full_loop
from src.V_utility.export import pipeline_to_markdown

CYAN = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; GREY = "\033[90m"
BOLD = "\033[1m"; RESET = "\033[0m"

TASK_SCHEMA = {
    "exhausted": False,
    "reason": "short note",
    "tasks": [
        {
            "domain": "Everyday Creative Framing",
            "input": "One new task prompt.",
            "why_novel": "short note",
        }
    ],
}


def _read_jsonl(path: Path) -> list[dict]:
    examples = []
    if not path.exists():
        return examples
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _write_jsonl(examples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def _append_jsonl(examples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def _rewrite_markdown_from_jsonl(path: Path):
    md_path = path.with_suffix(".md")
    blocks = []
    for idx, example in enumerate(_read_jsonl(path), 1):
        blocks.append(f"<!-- Example {idx} -->\n")
        blocks.append(pipeline_to_markdown(example))
        blocks.append("\n\n---\n\n")
    md_path.write_text("".join(blocks), encoding="utf-8")


def _list_candidate_files() -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return [p for p in sorted(OUTPUT_DIR.glob("*.jsonl")) if not p.name.endswith("_sft.jsonl")]


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _has_pass_iteration(example: dict) -> bool:
    for iteration in example.get("loop", []):
        verdict = str(iteration.get("critic", {}).get("verdict", "")).upper().strip()
        if verdict == "PASS":
            return True
    return False


def _domain_counts(examples: list[dict]) -> Counter:
    return Counter(str(example.get("domain", "unknown")).strip() or "unknown" for example in examples)


def _domain_summary(examples: list[dict]) -> str:
    counts = _domain_counts(examples)
    return "\n".join(f"- {domain}: {count}" for domain, count in sorted(counts.items()))


def _task_bullets(examples: list[dict]) -> str:
    return "\n".join(
        f"- {str(example.get('input', '')).strip()}"
        for example in examples
        if str(example.get("input", "")).strip()
    )


def _recent_task_bullets(items: list[dict]) -> str:
    if not items:
        return "(none yet)"
    return "\n".join(f"- {item['input']}" for item in items if item.get("input"))


def _pick_reference_examples(examples: list[dict], max_examples: int = 4) -> list[dict]:
    chosen = []
    seen_domains = set()
    for example in examples:
        domain = str(example.get("domain", "")).strip()
        if domain in seen_domains:
            continue
        chosen.append(
            {
                "domain": domain,
                "input": example.get("input", ""),
                "final_output": example.get("final_output", [])[:3],
            }
        )
        seen_domains.add(domain)
        if len(chosen) >= max_examples:
            break
    return chosen


def _build_output_path(input_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"{input_path.stem}_augmented_{timestamp}.jsonl"


def _debug_path(input_path: Path, round_idx: int, suffix: str) -> Path:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEBUG_DIR / f"{input_path.stem}_r{round_idx:02d}_{timestamp}.{suffix}"


def _extract_json(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
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

    brace = text.find("{")
    if brace < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(brace, len(text)):
        char = text[idx]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                chunk = text[brace:idx + 1]
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    return None
    return None


def _extract_json_array(text: str) -> list[dict]:
    text = (text or "").strip()
    if not text:
        return []

    start = text.find("[")
    if start < 0:
        return []

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                chunk = text[start:idx + 1]
                try:
                    data = json.loads(chunk)
                except json.JSONDecodeError:
                    return []
                return data if isinstance(data, list) else []
    return []


def _resolve_pipeline_mode(source_examples: list[dict]) -> tuple[str, callable, callable]:
    counts = Counter(str(example.get("_meta", {}).get("pipeline_mode", "simple")).strip().lower() for example in source_examples)
    mode = "advanced" if counts.get("advanced", 0) > counts.get("simple", 0) else "simple"
    if mode == "advanced":
        return "advanced", run_advanced_loop, validate_advanced_full_loop
    return "simple", run_loop, validate_simple_full_loop


def _build_task_prompt(
    source_examples: list[dict],
    seen_inputs: set[str],
    recent_tasks: list[dict],
    batch_size: int,
    round_idx: int,
) -> str:
    domains = sorted(_domain_counts(source_examples))
    references = _pick_reference_examples(source_examples)
    return f"""
You are expanding a creativity training dataset.

Task:
- Propose up to {batch_size} brand new task prompts.
- Keep them clearly different from the existing tasks.
- Use only these domains: {json.dumps(domains, ensure_ascii=False)}
- If you think the dataset is saturated, set exhausted=true.

Rules:
- Return JSON only.
- Each task must be one sentence.
- No duplicates.
- No close paraphrases of existing tasks.
- Keep prompts concrete and high-constraint.
- Favor unexplored combinations of constraints, contexts, and output formats.

Round: {round_idx}

Domain counts:
{_domain_summary(source_examples)}

Recent new tasks:
{_recent_task_bullets(recent_tasks)}

Existing task exclusions:
{chr(10).join(f"- {item}" for item in sorted(seen_inputs))}

Reference examples:
{json.dumps(references, ensure_ascii=False, indent=2)}

Return shape:
{json.dumps(TASK_SCHEMA, ensure_ascii=False, indent=2)}
""".strip()


def _call_task_proposer(generator, prompt: str, input_path: Path, round_idx: int) -> dict:
    owner = getattr(generator, "owner", None)
    raw_text = ""

    if owner and hasattr(owner, "clear"):
        owner.clear()

    if owner and hasattr(owner, "structured"):
        payload = owner.structured(
            prompt,
            system=(
                "You propose new dataset tasks. "
                "Return only valid JSON matching the schema."
            ),
            schema=TASK_SCHEMA,
        ) or {}
        if payload.get("tasks") or payload.get("exhausted"):
            return payload

    raw_text = generator(
        "You propose new dataset tasks. Return JSON only.",
        prompt,
    )

    payload = _extract_json(raw_text)
    if payload:
        return payload

    tasks = _extract_json_array(raw_text)
    if tasks:
        return {"exhausted": False, "reason": "Recovered from raw array output.", "tasks": tasks}

    debug_path = _debug_path(input_path, round_idx, "txt")
    debug_path.write_text(raw_text or "(empty)", encoding="utf-8")
    return {
        "exhausted": False,
        "reason": f"Could not parse model output. Raw saved to {debug_path.relative_to(REPO_ROOT)}",
        "tasks": [],
    }


def _normalize_domain(raw_domain: str, allowed_domains: list[str]) -> str | None:
    raw = str(raw_domain or "").strip()
    if not raw:
        return None
    lower_map = {item.lower(): item for item in allowed_domains}
    if raw.lower() in lower_map:
        return lower_map[raw.lower()]
    return None


def _generate_example_from_task(
    task_item: dict,
    generator,
    run_pipeline,
    validate_pipeline,
    pipeline_mode: str,
    input_path: Path,
    round_idx: int,
    model_label: str,
    max_attempts: int = 3,
) -> tuple[dict | None, str | None]:
    domain = task_item["domain"]
    task = task_item["input"]
    last_error = "unknown generation failure"

    for _attempt in range(1, max_attempts + 1):
        owner = getattr(generator, "owner", None)
        if owner and hasattr(owner, "clear"):
            owner.clear()

        example = run_pipeline(task=task, generate_fn=generator, domain=domain, verbose=False)
        errors = validate_pipeline(example)
        if errors:
            last_error = f"validation failed: {errors[:2]}"
            continue
        if not _has_pass_iteration(example):
            last_error = "no PASS iteration"
            continue

        example["_meta"] = {
            **(example.get("_meta") or {}),
            "pipeline_mode": pipeline_mode,
            "timestamp": datetime.now().isoformat(),
            "validation_errors": [],
            "augmented": True,
            "augmentation_round": round_idx,
            "augmentation_source_file": input_path.name,
            "augmentation_model": model_label,
            "augmentation_task_note": str(task_item.get("why_novel", "")).strip(),
        }
        return example, None

    return None, last_error


def augment_jsonl(
    input_path: Path,
    generator,
    output_path: Path | None = None,
    max_rounds: int = 6,
    batch_size: int = 3,
    verbose: bool = True,
) -> dict:
    source_examples = _read_jsonl(input_path)
    if not source_examples:
        raise ValueError(f"No examples found in {input_path}")

    pipeline_mode, run_pipeline, validate_pipeline = _resolve_pipeline_mode(source_examples)
    output_path = output_path or _build_output_path(input_path)
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    combined_examples = list(source_examples)
    _write_jsonl(combined_examples, output_path)
    _rewrite_markdown_from_jsonl(output_path)

    allowed_domains = sorted(_domain_counts(source_examples))
    seen_inputs = {_normalize_text(example.get("input", "")) for example in combined_examples}
    recent_tasks: list[dict] = []
    added_total = 0
    empty_rounds = 0
    model_label = getattr(generator, "alias", None) or getattr(generator, "model_id", None) or "unknown"

    if verbose:
        print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
        print(f"  {BOLD}Dataset Augmenter{RESET}")
        print(f"  Source: {input_path.name}")
        print(f"  Output: {output_path.name}")
        print(f"  Existing examples copied: {len(source_examples)}")
        print(f"  Pipeline: {pipeline_mode}")
        print(f"  Model: {model_label}")
        print(f"{BOLD}{CYAN}{'='*60}{RESET}")

    for round_idx in range(1, max_rounds + 1):
        if verbose:
            print(f"\n{CYAN}[augment round {round_idx}/{max_rounds}]{RESET} proposing up to {batch_size} new tasks")

        payload = _call_task_proposer(
            generator=generator,
            prompt=_build_task_prompt(
                source_examples=source_examples,
                seen_inputs=seen_inputs,
                recent_tasks=recent_tasks,
                batch_size=batch_size,
                round_idx=round_idx,
            ),
            input_path=input_path,
            round_idx=round_idx,
        )

        reason = str(payload.get("reason") or "").strip()
        exhausted = bool(payload.get("exhausted"))
        proposed = payload.get("tasks") or []

        clean_tasks = []
        rejected_tasks = 0
        for item in proposed:
            if not isinstance(item, dict):
                rejected_tasks += 1
                continue
            domain = _normalize_domain(item.get("domain", ""), allowed_domains)
            task = str(item.get("input") or "").strip()
            task_key = _normalize_text(task)
            if not domain or not task or not task_key or task_key in seen_inputs:
                rejected_tasks += 1
                continue
            clean_tasks.append(
                {
                    "domain": domain,
                    "input": task,
                    "why_novel": str(item.get("why_novel") or "").strip(),
                }
            )
            seen_inputs.add(task_key)
            if len(clean_tasks) >= batch_size:
                break

        if verbose:
            print(f"  {GREEN}task proposals:{RESET} {len(clean_tasks)}")
            if rejected_tasks:
                print(f"  {GREY}task rejects:{RESET} {rejected_tasks}")
            if reason:
                print(f"  {GREY}model note:{RESET} {reason}")

        accepted_examples = []
        example_rejects = 0
        recent_tasks = clean_tasks[-min(len(clean_tasks), 6):]
        for task_item in clean_tasks:
            if verbose:
                print(f"  {GREY}generate -> [{task_item['domain']}] {task_item['input'][:100]}{RESET}")
            example, error = _generate_example_from_task(
                task_item=task_item,
                generator=generator,
                run_pipeline=run_pipeline,
                validate_pipeline=validate_pipeline,
                pipeline_mode=pipeline_mode,
                input_path=input_path,
                round_idx=round_idx,
                model_label=model_label,
            )
            if example is None:
                example_rejects += 1
                if verbose:
                    print(f"  {YELLOW}skip:{RESET} {error}")
                continue
            accepted_examples.append(example)

        if accepted_examples:
            _append_jsonl(accepted_examples, output_path)
            _rewrite_markdown_from_jsonl(output_path)
            combined_examples.extend(accepted_examples)
            added_total += len(accepted_examples)
            empty_rounds = 0
            if verbose:
                print(f"  {GREEN}saved examples:{RESET} {len(accepted_examples)}")
        else:
            empty_rounds += 1
            if verbose:
                print(f"  {YELLOW}no saved examples this round{RESET}")
                if example_rejects:
                    print(f"  {GREY}example rejects:{RESET} {example_rejects}")

        if exhausted:
            if verbose:
                print(f"  {GREEN}model reports exhaustion{RESET}")
            break
        if empty_rounds >= 2:
            if verbose:
                print(f"  {YELLOW}stopping after repeated empty rounds{RESET}")
            break

    total_examples = len(_read_jsonl(output_path))
    if verbose:
        print(f"\n  {GREEN}Added:{RESET} {added_total}")
        print(f"  {GREEN}Total:{RESET} {total_examples}")
        print(f"  {GREEN}Saved:{RESET} {output_path.relative_to(REPO_ROOT)}")

    return {
        "input_path": input_path,
        "output_path": output_path,
        "added_count": added_total,
        "total_count": total_examples,
    }


def _pick_backend_and_model():
    from src.IV_inference.gemma4_integration import MODELS as HF_MODELS
    from src.IV_inference.ollama_integration import MODELS as OLLAMA_MODELS

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
            return "ollama", input("  Custom Ollama tag: ").strip() or "gemma4:e2b"
        model_alias = aliases[int(raw) - 1] if raw.isdigit() and 1 <= int(raw) <= len(aliases) else aliases[0]
        return "ollama", model_alias

    print(f"\n{YELLOW}Model:{RESET}")
    aliases = list(HF_MODELS.keys())
    for i, alias in enumerate(aliases, 1):
        info = HF_MODELS[alias]
        default = f" {GREY}<- default{RESET}" if i == 1 else ""
        print(f"  [{i}] {alias:<5} {info['ram_gb']:>3}GB  {info['description']}{default}")
    raw = input(f"\n  Choice [1]: ").strip() or "1"
    model_alias = aliases[int(raw) - 1] if raw.isdigit() and 1 <= int(raw) <= len(aliases) else aliases[0]
    return "hf", model_alias


def _pick_source_file() -> Path | None:
    files = _list_candidate_files()
    if not files:
        print(f"  {YELLOW}No source JSONL files found in data/output/.{RESET}")
        return None

    print(f"\n{YELLOW}Available source files:{RESET}")
    for i, path in enumerate(files, 1):
        print(f"  [{i}] {path.name}")

    raw = input("\n  Choice [1]: ").strip() or "1"
    if raw.isdigit() and 1 <= int(raw) <= len(files):
        return files[int(raw) - 1]
    return files[0]


def _tui():
    from src.II_dataGen.format_sft import process_jsonl
    from src.IV_inference.gemma4_integration import Gemma4
    from src.IV_inference.ollama_integration import OllamaGemma4

    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"  {BOLD}Dataset Augmenter{RESET}")
    print("  Expand raw pipeline JSONL with Gemma 4")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

    input_path = _pick_source_file()
    if input_path is None:
        return

    backend, model_alias = _pick_backend_and_model()
    if backend == "ollama":
        generator = OllamaGemma4(model=model_alias, thinking=False, use_memory=False).generate_fn()
    else:
        generator = Gemma4(model=model_alias, thinking=False, use_memory=False).generate_fn()

    print(f"\n{YELLOW}Max augmentation rounds [6]:{RESET}")
    rounds_raw = input("  > ").strip() or "6"
    max_rounds = int(rounds_raw) if rounds_raw.isdigit() else 6

    print(f"\n{YELLOW}Target new tasks per round [3]:{RESET}")
    batch_raw = input("  > ").strip() or "3"
    batch_size = int(batch_raw) if batch_raw.isdigit() else 3

    summary = augment_jsonl(
        input_path=input_path,
        generator=generator,
        max_rounds=max_rounds,
        batch_size=batch_size,
        verbose=True,
    )

    print(f"\n{YELLOW}Convert the augmented dataset to SFT format now? [y/N]{RESET}")
    if input("  > ").strip().lower() in ("y", "yes"):
        process_jsonl(summary["output_path"])


if __name__ == "__main__":
    _tui()
