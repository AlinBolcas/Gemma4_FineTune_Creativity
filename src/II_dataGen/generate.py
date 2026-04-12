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

sys.path.insert(0, str(REPO_ROOT))

from src.I_pipeline.runner import run_loop
from src.I_pipeline.schema import validate_full_loop

# Colors
CYAN = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; GREY = "\033[90m"
RED = "\033[91m"; BOLD = "\033[1m"; RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Data loading + generation core
# ---------------------------------------------------------------------------

def load_seed_prompts(path: Path = SEED_PROMPTS_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def generate_dataset(domain_key, generate_fn, limit=None, verbose=True) -> list[dict]:
    seeds = load_seed_prompts()
    domain_info = seeds["domains"].get(domain_key)
    if not domain_info:
        raise ValueError(f"Unknown domain '{domain_key}'. Available: {list(seeds['domains'].keys())}")

    prompts = domain_info["prompts"][:limit] if limit else domain_info["prompts"]
    label = domain_info["label"]
    results = []

    for i, task in enumerate(prompts, 1):
        if verbose:
            print(f"\n{BOLD}{'#'*60}{RESET}")
            print(f"  {BOLD}[{domain_key}] Prompt {i}/{len(prompts)}{RESET}")
            print(f"  {task[:80]}")
            print(f"{'#'*60}")

        start = time.time()
        example = run_loop(task=task, generate_fn=generate_fn, domain=label, verbose=verbose)
        elapsed = round(time.time() - start, 1)

        errors = validate_full_loop(example)
        if errors and verbose:
            print(f"  {YELLOW}[QA] Validation: {errors}{RESET}")

        example["_meta"] = {
            "domain_key": domain_key,
            "elapsed_sec": elapsed,
            "timestamp": datetime.now().isoformat(),
            "validation_errors": errors,
        }
        results.append(example)

    return results


def export_jsonl(examples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  {GREEN}Exported {len(examples)} examples -> {path.name}{RESET}")

    # Auto-convert each example to its own .md for readability
    try:
        from src.V_utility.export import pipeline_to_markdown
        md_path = path.with_suffix(".md")
        md_lines = []
        for i, ex in enumerate(examples, 1):
            md_lines.append(f"<!-- Example {i} -->\n")
            md_lines.append(pipeline_to_markdown(ex))
            md_lines.append("\n\n---\n\n")
        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"  {GREEN}Markdown: {md_path.name}{RESET}")
    except Exception as e:
        print(f"  {GREY}(markdown export skipped: {e}){RESET}")


# ---------------------------------------------------------------------------
# Interactive TUI
# ---------------------------------------------------------------------------

def _tui():
    from src.IV_inference.gemma4_integration import load_gemma4, MODELS

    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"  {BOLD}Data Generator{RESET}")
    print(f"  Batch pipeline runs -> JSONL training data")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

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

    # Domain
    seeds = load_seed_prompts()
    domain_keys = list(seeds["domains"].keys())
    print(f"\n{YELLOW}Domain:{RESET}")
    for i, k in enumerate(domain_keys, 1):
        label = seeds["domains"][k]["label"]
        count = len(seeds["domains"][k]["prompts"])
        print(f"  [{i}] {label} ({count} prompts)")
    raw = input(f"\n  Choice [1]: ").strip() or "1"
    domain = domain_keys[int(raw) - 1] if raw.isdigit() and 1 <= int(raw) <= len(domain_keys) else domain_keys[0]

    available = len(seeds["domains"][domain]["prompts"])
    print(f"\n{YELLOW}How many prompts? (max {available}) [{min(3, available)}]:{RESET}")
    limit_raw = input("  > ").strip() or str(min(3, available))
    limit = int(limit_raw) if limit_raw.isdigit() else min(3, available)

    # Generate
    results = generate_dataset(domain_key=domain, generate_fn=generate_fn, limit=limit, verbose=True)

    # Export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"{domain}_{timestamp}.jsonl"
    export_jsonl(results, out_path)

    valid = sum(1 for r in results if not r.get("_meta", {}).get("validation_errors"))
    print(f"\n  {GREEN}Done: {len(results)} generated, {valid} fully valid{RESET}")

    # Offer SFT conversion
    print(f"\n{YELLOW}Convert to SFT training format? [y/N]{RESET}")
    if input("  > ").strip().lower() in ("y", "yes"):
        from src.II_dataGen.format_sft import process_jsonl
        sft_path = out_path.with_name(out_path.stem + "_sft.jsonl")
        process_jsonl(out_path, sft_path)


if __name__ == "__main__":
    _tui()
