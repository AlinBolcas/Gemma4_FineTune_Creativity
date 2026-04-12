"""
generate.py - Generate synthetic training data via the pipeline.

Run interactively:
    python src/II_dataGen/generate.py
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SEED_PROMPTS_PATH = REPO_ROOT / "data" / "input" / "seed_prompts.json"
OUTPUT_DIR = REPO_ROOT / "data" / "output"
DATA_INPUT_DIR = REPO_ROOT / "data" / "input"
TRAIN_DIR = DATA_INPUT_DIR / "train"
EVAL_DIR = DATA_INPUT_DIR / "eval"
TEST_DIR = DATA_INPUT_DIR / "test"

sys.path.insert(0, str(REPO_ROOT))

from src.I_pipeline.runner import run_loop
from src.I_pipeline.runner_advanced import run_advanced_loop
from src.I_pipeline.schema import validate_full_loop as validate_simple_full_loop
from src.I_pipeline.schema_advanced import validate_full_loop as validate_advanced_full_loop

CYAN = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; GREY = "\033[90m"
RED = "\033[91m"; BOLD = "\033[1m"; RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Paths and data loading
# ---------------------------------------------------------------------------

def ensure_data_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)


def load_seed_prompts(path: Path = SEED_PROMPTS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _count_by_domain(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in _read_jsonl(path):
        key = item.get("_meta", {}).get("domain_key")
        if key:
            counts[key] = counts.get(key, 0) + 1
    return counts


def _append_example(path: Path, example: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")


def _rewrite_markdown_from_jsonl(path: Path):
    try:
        from src.V_utility.export import pipeline_to_markdown
        examples = _read_jsonl(path)
        md_path = path.with_suffix(".md")
        blocks = []
        for i, ex in enumerate(examples, 1):
            blocks.append(f"<!-- Example {i} -->\n")
            blocks.append(pipeline_to_markdown(ex))
            blocks.append("\n\n---\n\n")
        md_path.write_text("".join(blocks), encoding="utf-8")
    except Exception as e:
        print(f"  {GREY}(markdown export skipped: {e}){RESET}")


# ---------------------------------------------------------------------------
# Generation core
# ---------------------------------------------------------------------------

def _resolve_pipeline_mode(pipeline_mode: str):
    mode = (pipeline_mode or "simple").strip().lower()
    if mode == "advanced":
        return "advanced", run_advanced_loop, validate_advanced_full_loop
    return "simple", run_loop, validate_simple_full_loop


def generate_dataset(
    domain_key,
    generate_fn,
    limit=None,
    verbose=True,
    output_path: Path | None = None,
    resume=True,
    pipeline_mode: str = "simple",
) -> list[dict]:
    seeds = load_seed_prompts()
    domain_info = seeds["domains"].get(domain_key)
    if not domain_info:
        raise ValueError(f"Unknown domain '{domain_key}'. Available: {list(seeds['domains'].keys())}")

    pipeline_mode, run_pipeline, validate_pipeline = _resolve_pipeline_mode(pipeline_mode)
    prompts = domain_info["prompts"][:limit] if limit else domain_info["prompts"]
    label = domain_info["label"]
    results = []

    already_done = 0
    if resume and output_path:
        already_done = _count_by_domain(output_path).get(domain_key, 0)
        if already_done and verbose:
            print(f"  {GREY}Resume: skipping first {already_done} saved prompt(s) in {domain_key}.{RESET}")

    prompts = prompts[already_done:]
    total_prompts = len(prompts)

    for i, task in enumerate(prompts, 1):
        if verbose:
            print(f"\n{BOLD}{'#'*60}{RESET}")
            print(f"  {BOLD}[{domain_key}] Prompt {already_done + i}/{already_done + total_prompts}{RESET}")
            print(f"  {task[:100]}")
            print(f"{'#'*60}")

        # Explicitly clear any wrapper memory between examples.
        owner = getattr(generate_fn, "owner", None)
        if owner and hasattr(owner, "clear"):
            owner.clear()

        start = time.time()
        example = run_pipeline(task=task, generate_fn=generate_fn, domain=label, verbose=verbose)
        elapsed = round(time.time() - start, 1)

        errors = validate_pipeline(example)
        if errors and verbose:
            print(f"  {YELLOW}[QA] Validation: {errors}{RESET}")

        existing_meta = example.get("_meta", {})
        example["_meta"] = {
            **existing_meta,
            "domain_key": domain_key,
            "pipeline_mode": pipeline_mode,
            "elapsed_sec": elapsed,
            "timestamp": datetime.now().isoformat(),
            "validation_errors": errors,
        }
        results.append(example)

        # Save every example immediately so interruption does not lose progress.
        if output_path:
            _append_example(output_path, example)
            _rewrite_markdown_from_jsonl(output_path)
            if verbose:
                print(f"  {GREEN}Saved example immediately -> {output_path.name}{RESET}")

    return results


def generate_many(
    domain_keys: list[str],
    generate_fn,
    limit_per_domain=None,
    verbose=True,
    output_path: Path | None = None,
    resume=True,
    pipeline_mode: str = "simple",
) -> list[dict]:
    all_results = []
    for domain_key in domain_keys:
        results = generate_dataset(
            domain_key=domain_key,
            generate_fn=generate_fn,
            limit=limit_per_domain,
            verbose=verbose,
            output_path=output_path,
            resume=resume,
            pipeline_mode=pipeline_mode,
        )
        all_results.extend(results)
    return all_results


# ---------------------------------------------------------------------------
# TUI helpers
# ---------------------------------------------------------------------------

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


def _pick_domain_keys(seeds: dict) -> list[str]:
    domain_keys = list(seeds["domains"].keys())
    rec = seeds.get("_recommended_pilots", {})

    print(f"\n{YELLOW}Domain selection mode:{RESET}")
    print("  [1] one domain")
    print("  [2] multiple domains")
    print("  [3] all domains")
    if rec:
        print("  [4] recommended fast small run")
        print("  [5] recommended strong generalization core")
    mode = input("\n  Choice [1]: ").strip() or "1"

    if mode == "3":
        return domain_keys
    if mode == "4" and rec.get("fast_small_run"):
        return list(rec["fast_small_run"])
    if mode == "5" and rec.get("strong_generalization_core"):
        return list(rec["strong_generalization_core"])

    print(f"\n{YELLOW}Domains:{RESET}")
    for i, key in enumerate(domain_keys, 1):
        info = seeds["domains"][key]
        print(f"  [{i}] {info['label']} ({len(info['prompts'])} prompts)")

    if mode == "2":
        raw = input("\n  Pick multiple (e.g. 1,3,5) [1,2,4]: ").strip() or "1,2,4"
        picks = []
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                idx = int(part)
                if 1 <= idx <= len(domain_keys):
                    picks.append(domain_keys[idx - 1])
        seen = set()
        result = []
        for key in picks:
            if key not in seen:
                seen.add(key)
                result.append(key)
        return result or [domain_keys[0]]

    raw = input("\n  Choice [1]: ").strip() or "1"
    return [domain_keys[int(raw) - 1]] if raw.isdigit() and 1 <= int(raw) <= len(domain_keys) else [domain_keys[0]]


def _pick_pipeline_mode() -> str:
    print(f"\n{YELLOW}Pipeline architecture:{RESET}")
    print("  [1] simple")
    print("  [2] advanced")
    raw = input("\n  Choice [1]: ").strip() or "1"
    return "advanced" if raw == "2" else "simple"


def _build_output_name(domain_keys: list[str], pipeline_mode: str = "simple") -> str:
    if len(domain_keys) == 1:
        stem = domain_keys[0]
    elif len(domain_keys) == len(load_seed_prompts()["domains"]):
        stem = "all_domains"
    else:
        stem = "multi_domain"
    if pipeline_mode == "advanced":
        stem += "_advanced"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{timestamp}.jsonl"


def _pick_output_path(domain_keys: list[str], pipeline_mode: str) -> tuple[Path, bool]:
    existing = sorted(OUTPUT_DIR.glob("*.jsonl"))
    print(f"\n{YELLOW}Output mode:{RESET}")
    print("  [1] new output file")
    if existing:
        print("  [2] resume existing output file")
    raw = input("\n  Choice [1]: ").strip() or "1"

    if raw == "2" and existing:
        print(f"\n{YELLOW}Existing datasets:{RESET}")
        for i, path in enumerate(existing, 1):
            print(f"  [{i}] {path.name}")
        pick = input("\n  Choice [1]: ").strip() or "1"
        idx = int(pick) - 1 if pick.isdigit() and 1 <= int(pick) <= len(existing) else 0
        return existing[idx], True

    return OUTPUT_DIR / _build_output_name(domain_keys, pipeline_mode=pipeline_mode), False


# ---------------------------------------------------------------------------
# Interactive TUI
# ---------------------------------------------------------------------------

def _tui():
    from src.IV_inference.gemma4_integration import load_gemma4
    from src.IV_inference.ollama_integration import load_ollama_gemma4

    ensure_data_dirs()

    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"  {BOLD}Data Generator{RESET}")
    print("  Batch pipeline runs -> JSONL training data")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

    seeds = load_seed_prompts()

    rec = seeds.get("_recommended_pilots", {})
    if rec:
        print(f"\n{GREY}Recommended sets:{RESET}")
        for key, value in rec.items():
            print(f"  - {key}: {value}")

    domain_keys = _pick_domain_keys(seeds)
    print(f"\n  {GREEN}Selected:{RESET} {', '.join(domain_keys)}")

    pipeline_mode = _pick_pipeline_mode()
    print(f"  {GREEN}Architecture:{RESET} {pipeline_mode}")

    smallest = min(len(seeds["domains"][k]["prompts"]) for k in domain_keys)
    default_limit = min(3, smallest)
    print(f"\n{YELLOW}How many prompts per domain? (max {smallest}) [{default_limit}]:{RESET}")
    limit_raw = input("  > ").strip() or str(default_limit)
    limit_per_domain = int(limit_raw) if limit_raw.isdigit() else default_limit

    output_path, resume = _pick_output_path(domain_keys, pipeline_mode=pipeline_mode)
    print(f"\n  {GREEN}Output file:{RESET} {output_path.name}")
    print(f"  {GREEN}Mode:{RESET} {'resume' if resume else 'new'}")

    backend, model_alias = _pick_backend_and_model()
    if backend == "ollama":
        generate_fn = load_ollama_gemma4(model=model_alias, thinking=False, use_memory=False)
    else:
        generate_fn = load_gemma4(model=model_alias, thinking=False, use_memory=False)

    try:
        all_results = generate_many(
            domain_keys=domain_keys,
            generate_fn=generate_fn,
            limit_per_domain=limit_per_domain,
            verbose=True,
            output_path=output_path,
            resume=resume,
            pipeline_mode=pipeline_mode,
        )
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Stopped by user. Saved progress remains in:{RESET} {output_path.resolve()}")
        return

    current_total = len(_read_jsonl(output_path))
    valid = sum(1 for r in _read_jsonl(output_path) if not r.get("_meta", {}).get("validation_errors"))
    print(f"\n  {GREEN}Done:{RESET} {current_total} total saved, {valid} fully valid")

    print(f"\n{YELLOW}Convert this dataset to SFT training format and create train/eval/test splits? [y/N]{RESET}")
    if input("  > ").strip().lower() in ("y", "yes"):
        from src.II_dataGen.format_sft import process_jsonl
        process_jsonl(output_path)


if __name__ == "__main__":
    _tui()
