"""
generate.py - Generate synthetic training data via the pipeline.

Run interactively:
    python src/II_dataGen/generate.py
"""

import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _pick_existing_dataset_file() -> Path | None:
    existing = sorted(OUTPUT_DIR.glob("*.jsonl"))
    if not existing:
        print(f"  {YELLOW}No existing output datasets found in data/output/.{RESET}")
        return None

    print(f"\n{YELLOW}Reference dataset:{RESET}")
    for i, path in enumerate(existing, 1):
        counts = _count_by_domain(path)
        covered = len(counts)
        total = sum(counts.values())
        print(f"  [{i}] {path.name}  {GREY}(rows={total}, domains={covered}){RESET}")

    raw = input("\n  Choice [1]: ").strip() or "1"
    idx = int(raw) - 1 if raw.isdigit() and 1 <= int(raw) <= len(existing) else 0
    return existing[idx]


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _append_example(path: Path, example: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")


def _append_markdown_example(path: Path, example: dict, example_index: int):
    from src.V_utility.export import pipeline_to_markdown

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"<!-- Example {example_index} -->\n")
        f.write(pipeline_to_markdown(example))
        f.write("\n\n---\n\n")


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


def _safe_slug(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return (slug or "task")[:max_len]


def _parts_dir_for_output(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}_parts"


def _example_artifact_paths(parts_dir: Path, job: dict) -> tuple[Path, Path]:
    stem = f"{job['seq']:04d}_{job['domain_key']}_{_safe_slug(job['task'])}"
    return parts_dir / f"{stem}.json", parts_dir / f"{stem}.md"


def _write_example_artifacts(parts_dir: Path, job: dict, example: dict):
    from src.V_utility.export import pipeline_to_markdown

    parts_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = _example_artifact_paths(parts_dir, job)
    json_path.write_text(json.dumps(example, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(pipeline_to_markdown(example), encoding="utf-8")
    return json_path, md_path


def _render_progress_bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = min(width, int((done / total) * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _summarize_jobs(jobs: list[dict]) -> list[str]:
    counts: dict[str, int] = {}
    for job in jobs:
        counts[job["domain_key"]] = counts.get(job["domain_key"], 0) + 1
    return [f"{key}={counts[key]}" for key in sorted(counts)]


def _zero_usage() -> dict:
    return {
        "requests": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
    }


def _openai_usage_snapshot(owner) -> dict:
    if owner and hasattr(owner, "get_usage_snapshot"):
        return owner.get_usage_snapshot()
    return _zero_usage()


def _usage_delta(before: dict, after: dict) -> dict:
    result = {}
    for key, default in _zero_usage().items():
        result[key] = after.get(key, default) - before.get(key, default)
    return result


def _merge_usage(items: list[dict]) -> dict:
    total = _zero_usage()
    for item in items:
        if not item:
            continue
        for key in total:
            total[key] += item.get(key, 0)
    return total


def _format_cost(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"${value:.4f}" if value < 0.01 else f"${value:.2f}"


def _format_usage_line(usage: dict) -> str:
    return (
        f"cost={_format_cost(usage.get('cost_usd', 0.0))}  "
        f"requests={int(usage.get('requests', 0))}  "
        f"in={int(usage.get('input_tokens', 0)):,}  "
        f"cached={int(usage.get('cached_input_tokens', 0)):,}  "
        f"out={int(usage.get('output_tokens', 0)):,}"
    )


def _estimate_openai_run_cost(model_id: str, jobs: list[dict], pipeline_mode: str) -> dict:
    from src.IV_inference.openai_integration import estimate_cost_usd

    # Conservative starter estimates; final reporting uses real API token usage.
    if pipeline_mode == "advanced":
        calls_per_job, input_per_call, output_per_call = 11, 1400, 700
    else:
        calls_per_job, input_per_call, output_per_call = 3, 900, 550
    calls = len(jobs) * calls_per_job
    input_tokens = calls * input_per_call
    output_tokens = calls * output_per_call
    cost = estimate_cost_usd(model_id, input_tokens=input_tokens, output_tokens=output_tokens)
    return {
        "jobs": len(jobs),
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
    }


def _start_progress_monitor(state: dict, stop_event: threading.Event, interval_sec: int = 15):
    def monitor():
        while not stop_event.wait(interval_sec):
            with state["lock"]:
                total = state["total"]
                completed = state["completed"]
                active = state["active"]
                started = state["started"]
                failed = state["failed"]
                elapsed = round(time.time() - state["started_at"], 1)
                queued = max(0, total - started)
                bar = _render_progress_bar(completed, total)
            print(
                f"\n{CYAN}[progress]{RESET} {bar} {completed}/{total} done"
                f"  active={active} queued={queued} failed={failed} elapsed={elapsed}s"
            )

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    return thread


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

        usage_before = _openai_usage_snapshot(owner)
        start = time.time()
        example = run_pipeline(task=task, generate_fn=generate_fn, domain=label, verbose=verbose)
        elapsed = round(time.time() - start, 1)
        usage = _usage_delta(usage_before, _openai_usage_snapshot(owner))

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
            "openai_usage": usage if usage.get("requests") else None,
        }
        results.append(example)

        # Save every example immediately so interruption does not lose progress.
        if output_path:
            _append_example(output_path, example)
            _rewrite_markdown_from_jsonl(output_path)
            if verbose:
                print(f"  {GREEN}Saved example immediately -> {output_path.name}{RESET}")
        if verbose and usage.get("requests"):
            print(f"  {CYAN}[OpenAI cost]{RESET} {_format_usage_line(usage)}")

    return results


def _build_generation_jobs(
    domain_keys: list[str],
    limit_per_domain=None,
    output_path: Path | None = None,
    resume: bool = True,
    verbose: bool = True,
    pipeline_mode: str = "simple",
) -> list[dict]:
    seeds = load_seed_prompts()
    jobs = []
    next_seq = len(_read_jsonl(output_path)) + 1 if output_path and output_path.exists() else 1

    for domain_key in domain_keys:
        domain_info = seeds["domains"].get(domain_key)
        if not domain_info:
            raise ValueError(f"Unknown domain '{domain_key}'. Available: {list(seeds['domains'].keys())}")

        prompts = domain_info["prompts"][:limit_per_domain] if limit_per_domain else domain_info["prompts"]
        already_done = 0
        if resume and output_path:
            already_done = _count_by_domain(output_path).get(domain_key, 0)
            if already_done and verbose:
                print(f"  {GREY}Resume: skipping first {already_done} saved prompt(s) in {domain_key}.{RESET}")

        prompts = prompts[already_done:]
        total_prompts = len(prompts)

        for i, task in enumerate(prompts, 1):
            jobs.append(
                {
                    "seq": next_seq,
                    "task": task,
                    "domain_key": domain_key,
                    "domain_label": domain_info["label"],
                    "pipeline_mode": pipeline_mode,
                    "domain_position": already_done + i,
                    "domain_total": already_done + total_prompts,
                }
            )
            next_seq += 1

    return jobs


def _run_generation_job(
    job: dict,
    generate_fn=None,
    get_thread_generate_fn=None,
    on_start=None,
    on_finish=None,
) -> dict:
    pipeline_mode, run_pipeline, validate_pipeline = _resolve_pipeline_mode(job["pipeline_mode"])
    local_generate_fn = get_thread_generate_fn() if get_thread_generate_fn else generate_fn
    if local_generate_fn is None:
        raise ValueError("A generate function or thread-local generate function provider is required.")

    owner = getattr(local_generate_fn, "owner", None)
    if owner and hasattr(owner, "clear"):
        owner.clear()

    if on_start:
        on_start(job)

    usage_before = _openai_usage_snapshot(owner)
    start = time.time()
    try:
        example = run_pipeline(
            task=job["task"],
            generate_fn=local_generate_fn,
            domain=job["domain_label"],
            verbose=False,
        )
        elapsed = round(time.time() - start, 1)
        usage = _usage_delta(usage_before, _openai_usage_snapshot(owner))
        errors = validate_pipeline(example)

        existing_meta = example.get("_meta", {})
        example["_meta"] = {
            **existing_meta,
            "domain_key": job["domain_key"],
            "pipeline_mode": pipeline_mode,
            "elapsed_sec": elapsed,
            "timestamp": datetime.now().isoformat(),
            "validation_errors": errors,
            "parallel_seq": job["seq"],
            "openai_usage": usage if usage.get("requests") else None,
        }
        if on_finish:
            on_finish(job, elapsed, False)
        return {"job": job, "example": example, "elapsed": elapsed, "errors": errors, "usage": usage}
    except Exception:
        elapsed = round(time.time() - start, 1)
        if on_finish:
            on_finish(job, elapsed, True)
        raise


def _make_thread_local_generate_fn(worker_factory):
    state = threading.local()

    def get_thread_generate_fn():
        if not hasattr(state, "generate_fn"):
            state.generate_fn = worker_factory()
        return state.generate_fn

    return get_thread_generate_fn


def generate_many(
    domain_keys: list[str],
    generate_fn,
    limit_per_domain=None,
    verbose=True,
    output_path: Path | None = None,
    resume=True,
    pipeline_mode: str = "simple",
    parallel_workers: int = 1,
    worker_factory=None,
) -> list[dict]:
    if parallel_workers <= 1:
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
        usage_total = _merge_usage([
            r.get("_meta", {}).get("openai_usage") for r in all_results
            if r.get("_meta", {}).get("openai_usage")
        ])
        if verbose and usage_total.get("requests"):
            print(f"\n{GREEN}OpenAI run total:{RESET} {_format_usage_line(usage_total)}")
        return all_results

    if worker_factory is None:
        if verbose:
            print(f"  {YELLOW}Parallel mode needs a worker factory. Falling back to sequential.{RESET}")
        return generate_many(
            domain_keys=domain_keys,
            generate_fn=generate_fn,
            limit_per_domain=limit_per_domain,
            verbose=verbose,
            output_path=output_path,
            resume=resume,
            pipeline_mode=pipeline_mode,
            parallel_workers=1,
            worker_factory=None,
        )

    jobs = _build_generation_jobs(
        domain_keys=domain_keys,
        limit_per_domain=limit_per_domain,
        output_path=output_path,
        resume=resume,
        verbose=verbose,
        pipeline_mode=pipeline_mode,
    )
    if not jobs:
        if verbose:
            print(f"  {GREY}Nothing left to generate. All selected prompts are already saved.{RESET}")
        return []

    parts_dir = _parts_dir_for_output(output_path) if output_path else None
    master_md_path = output_path.with_suffix(".md") if output_path else None
    if output_path and resume and output_path.exists():
        _rewrite_markdown_from_jsonl(output_path)

    existing_total = len(_read_jsonl(output_path)) if output_path and output_path.exists() else 0
    get_thread_generate_fn = _make_thread_local_generate_fn(worker_factory)
    all_results = []
    usage_results = []
    saved_count = 0
    state = {
        "lock": threading.Lock(),
        "started_at": time.time(),
        "total": len(jobs),
        "started": 0,
        "completed": 0,
        "active": 0,
        "failed": 0,
    }
    stop_event = threading.Event()

    if verbose:
        print(f"\n{GREEN}Parallel generation enabled:{RESET} {parallel_workers} worker threads")
        print(f"  {GREEN}Writer mode:{RESET} per-example files + serial master append")
        print(f"  {GREEN}Queued jobs:{RESET} {len(jobs)}")
        print(f"  {GREEN}By domain:{RESET} {', '.join(_summarize_jobs(jobs))}")
        for preview in jobs[:min(3, len(jobs))]:
            print(f"  {GREY}next -> [{preview['domain_key']}] {preview['task'][:90]}{RESET}")

    def on_start(job: dict):
        with state["lock"]:
            state["started"] += 1
            state["active"] += 1
            started = state["started"]
            total = state["total"]
            active = state["active"]
        if verbose:
            print(
                f"\n{CYAN}[start {started}/{total}]{RESET} "
                f"[{job['domain_key']}] {job['domain_position']}/{job['domain_total']}  active={active}"
            )
            print(f"  {GREY}{job['task'][:120]}{RESET}")

    def on_finish(job: dict, elapsed: float, failed: bool):
        with state["lock"]:
            state["active"] = max(0, state["active"] - 1)
            if failed:
                state["failed"] += 1
            active = state["active"]
            failed_count = state["failed"]
        if verbose:
            status = "failed" if failed else "finished"
            color = RED if failed else GREY
            print(
                f"  {color}[{status}]{RESET} "
                f"[{job['domain_key']}] seq={job['seq']} elapsed={elapsed}s active={active} failed={failed_count}"
            )

    monitor_thread = _start_progress_monitor(state, stop_event, interval_sec=12) if verbose else None

    try:
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            future_to_job = {
                executor.submit(_run_generation_job, job, None, get_thread_generate_fn, on_start, on_finish): job
                for job in jobs
            }

            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    payload = future.result()
                except Exception as e:
                    print(f"\n{RED}Generation failed{RESET} [{job['domain_key']}] {job['task'][:90]}")
                    print(f"  {RED}{e}{RESET}")
                    raise

                example = payload["example"]
                errors = payload["errors"]
                elapsed = payload["elapsed"]
                usage = payload.get("usage") or _zero_usage()
                all_results.append(example)
                if usage.get("requests"):
                    usage_results.append(usage)

                if output_path:
                    if parts_dir:
                        _write_example_artifacts(parts_dir, job, example)
                    _append_example(output_path, example)
                    if master_md_path:
                        _append_markdown_example(master_md_path, example, existing_total + saved_count + 1)

                saved_count += 1
                with state["lock"]:
                    state["completed"] += 1
                    completed = state["completed"]
                    total = state["total"]
                    active = state["active"]
                    failed_count = state["failed"]
                    bar = _render_progress_bar(completed, total)
                if verbose:
                    print(f"\n{BOLD}{'#'*60}{RESET}")
                    print(
                        f"  {BOLD}[done {saved_count}/{len(jobs)}]{RESET} "
                        f"{bar} [{job['domain_key']}] {job['domain_position']}/{job['domain_total']}"
                    )
                    print(f"  {job['task'][:100]}")
                    print(f"  elapsed: {elapsed}s  active={active} failed={failed_count}")
                    if usage.get("requests"):
                        print(f"  {CYAN}[OpenAI cost]{RESET} {_format_usage_line(usage)}")
                    if errors:
                        print(f"  {YELLOW}[QA] Validation: {errors}{RESET}")
                    if output_path:
                        print(f"  {GREEN}Saved example -> {output_path.name}{RESET}")
                    print(f"{'#'*60}")
    finally:
        stop_event.set()
        if monitor_thread:
            monitor_thread.join(timeout=1)

    if verbose and usage_results:
        print(f"\n{GREEN}OpenAI run total:{RESET} {_format_usage_line(_merge_usage(usage_results))}")

    return all_results

# ---------------------------------------------------------------------------
# TUI helpers
# ---------------------------------------------------------------------------

def _pick_backend_and_model():
    from src.IV_inference.gemma4_integration import MODELS as HF_MODELS
    from src.IV_inference.ollama_integration import MODELS as OLLAMA_MODELS
    from src.IV_inference.openai_integration import (
        MODELS_DOCS_URL,
        MODEL_PICKER_DISPLAY_CAP,
        PRICING_DOCS_URL,
        fetch_text_generation_models,
        get_default_model_id,
        normalize_priced_model_id,
        price_brief,
    )

    print(f"\n{YELLOW}Backend:{RESET}")
    print("  [1] OpenAI API")
    print("  [2] Ollama local")
    print("  [3] Hugging Face transformers")
    backend = input("\n  Choice [1]: ").strip() or "1"

    if backend == "1":
        rows = fetch_text_generation_models()
        priced_rows = [row for row in rows if normalize_priced_model_id(row["id"])]
        shown = (priced_rows or rows)[:MODEL_PICKER_DISPLAY_CAP]
        default_id = shown[0]["id"] if shown else get_default_model_id()
        print(f"\n{YELLOW}OpenAI model:{RESET} {GREY}(5.5/5.4 priced flagship models for your key){RESET}")
        print(f"  {GREY}Models: {MODELS_DOCS_URL}{RESET}")
        print(f"  {GREY}Pricing: {PRICING_DOCS_URL}{RESET}")
        for i, row in enumerate(shown, 1):
            desc = price_brief(row["id"]) if normalize_priced_model_id(row["id"]) else (row.get("description") or "")[:44]
            default = f" {GREY}<- default{RESET}" if i == 1 else ""
            print(f"  [{i}] {row['id']:<26}  {desc}{default}")
        if priced_rows and len(rows) > len(priced_rows):
            print(f"  {GREY}Showing only priced 5.5/5.4 choices. Use [c] to type another live id.{RESET}")
        print("  [c] custom model id")
        raw = input(f"\n  Choice [1]: ").strip().lower() or "1"
        if raw == "c":
            return "openai", input("  Custom OpenAI model id: ").strip() or default_id
        idx = int(raw) - 1 if raw.isdigit() and 1 <= int(raw) <= len(shown) else 0
        model_alias = shown[idx]["id"] if shown else default_id
        return "openai", model_alias

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
    print(f"      {GREY}Generate prompts from exactly one domain only.{RESET}")
    print("  [2] multiple domains")
    print(f"      {GREY}You choose a few domains manually.{RESET}")
    print("  [3] all domains")
    print(f"      {GREY}Use every domain in seed_prompts.json. Biggest run.{RESET}")
    print("  [6] uncovered domains from existing dataset")
    print(f"      {GREY}Only domains missing from a reference dataset. Falls back if none are missing.{RESET}")
    print("  [7] lowest-count domains from existing dataset")
    print(f"      {GREY}Focus on the least-represented domains in a reference dataset.{RESET}")
    if rec:
        print("  [4] recommended fast small run")
        print(f"      {GREY}Quick smoke test. Smallest useful subset.{RESET}")
        print("  [5] recommended strong generalization core")
        print(f"      {GREY}Best medium-size subset for broad quality.{RESET}")
    mode = input("\n  Choice [1]: ").strip() or "1"

    if mode == "3":
        return domain_keys
    if mode == "4" and rec.get("fast_small_run"):
        return list(rec["fast_small_run"])
    if mode == "5" and rec.get("strong_generalization_core"):
        return list(rec["strong_generalization_core"])
    if mode in {"6", "7"}:
        ref_path = _pick_existing_dataset_file()
        if ref_path is None:
            return domain_keys

        counts = _count_by_domain(ref_path)
        missing = [key for key in domain_keys if counts.get(key, 0) == 0]

        print(f"\n{YELLOW}Domain coverage in {ref_path.name}:{RESET}")
        for i, key in enumerate(domain_keys, 1):
            label = seeds["domains"][key]["label"]
            print(f"  [{i}] {label:<45} {counts.get(key, 0)}")

        if mode == "6":
            if missing:
                print(f"\n{GREEN}Selecting uncovered domains:{RESET} {', '.join(missing)}")
                return missing
            print(f"\n{YELLOW}All domains already appear at least once in that dataset.{RESET}")
            raw = input("  Use the lowest-count domains instead? [Y/n]: ").strip().lower()
            if raw not in ("", "y", "yes"):
                return domain_keys

        min_count = min(counts.get(key, 0) for key in domain_keys)
        lowest = [key for key in domain_keys if counts.get(key, 0) == min_count]
        default = ",".join(str(domain_keys.index(key) + 1) for key in lowest[:4]) or "1"
        print(f"\n{YELLOW}Lowest-count domains:{RESET}")
        for key in lowest:
            print(f"  - {seeds['domains'][key]['label']} ({counts.get(key, 0)})")
        raw = input(f"\n  Pick multiple from lowest-count set (e.g. {default}) [{default}]: ").strip() or default
        picks = []
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                idx = int(part)
                if 1 <= idx <= len(domain_keys):
                    picks.append(domain_keys[idx - 1])
        filtered = [key for key in _dedupe_keep_order(picks) if key in lowest]
        return filtered or lowest

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
        result = _dedupe_keep_order(picks)
        return result or [domain_keys[0]]

    raw = input("\n  Choice [1]: ").strip() or "1"
    return [domain_keys[int(raw) - 1]] if raw.isdigit() and 1 <= int(raw) <= len(domain_keys) else [domain_keys[0]]


def _pick_pipeline_mode() -> str:
    print(f"\n{YELLOW}Pipeline architecture:{RESET}")
    print("  [1] simple")
    print(f"      {GREY}Faster. Good default for generating lots of training data.{RESET}")
    print("  [2] advanced")
    print(f"      {GREY}Slower but deeper traces. Use when quality matters more than speed.{RESET}")
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
    print(f"      {GREY}Recommended. Creates a fresh dataset file for this run.{RESET}")
    if existing:
        print("  [2] resume existing output file")
        print(f"      {GREY}Only use this if you want to continue adding into an older file.{RESET}")
    raw = input("\n  Choice [1]: ").strip() or "1"

    if raw == "2" and existing:
        print(f"\n{YELLOW}Existing datasets:{RESET}")
        for i, path in enumerate(existing, 1):
            print(f"  [{i}] {path.name}")
        pick = input("\n  Choice [1]: ").strip() or "1"
        idx = int(pick) - 1 if pick.isdigit() and 1 <= int(pick) <= len(existing) else 0
        return existing[idx], True

    return OUTPUT_DIR / _build_output_name(domain_keys, pipeline_mode=pipeline_mode), False


def _pick_parallel_workers(backend: str, pipeline_mode: str, model_alias: str) -> int:
    if backend == "hf":
        print(f"\n{GREY}Parallel workers are disabled for Hugging Face on local Mac. One in-process model is safer.{RESET}")
        return 1

    if backend == "openai":
        default_workers = 4 if pipeline_mode == "advanced" else 8
        print(f"\n{YELLOW}Parallel worker threads (OpenAI API):{RESET}")
        print(f"  {GREY}Use 1 for serial mode. Higher is faster but may hit rate limits.{RESET}")
        print(f"  {GREY}Recommended: 4 for advanced, 8 for simple. Reduce if you see rate-limit errors.{RESET}")
        raw = input(f"\n  Workers [{default_workers}]: ").strip() or str(default_workers)
        workers = int(raw) if raw.isdigit() else default_workers
        return max(1, workers)

    cpu_count = os.cpu_count() or 4
    if pipeline_mode == "advanced":
        default_workers = 1
    elif str(model_alias).lower().strip() == "e4b":
        default_workers = min(2, cpu_count)
    else:
        default_workers = min(4, cpu_count)

    print(f"\n{YELLOW}Parallel worker threads (Ollama only):{RESET}")
    print(f"  {GREY}Use 1 for serial mode (no threading).{RESET}")
    if pipeline_mode == "advanced":
        print(f"  {GREY}Advanced pipeline is heavy. Recommended: 1 worker on Mac, maybe 2 max.{RESET}")
    elif str(model_alias).lower().strip() == "e4b":
        print(f"  {GREY}E4B is heavier than E2B. Recommended: 1-2 workers on Mac.{RESET}")
    else:
        print(f"  {GREY}Simple + E2B usually handles 2-4 workers best.{RESET}")
    raw = input(f"\n  Workers [{default_workers}]: ").strip() or str(default_workers)
    workers = int(raw) if raw.isdigit() else default_workers
    return max(1, workers)


# ---------------------------------------------------------------------------
# Interactive TUI
# ---------------------------------------------------------------------------

def _tui():
    from src.IV_inference.gemma4_integration import load_gemma4
    from src.IV_inference.ollama_integration import load_ollama_gemma4
    from src.IV_inference.openai_integration import load_openai_generator, resolve_model_id

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
    print(f"  {GREY}Domain count:{RESET} {len(domain_keys)}")

    pipeline_mode = _pick_pipeline_mode()
    print(f"  {GREEN}Architecture:{RESET} {pipeline_mode}")

    smallest = min(len(seeds["domains"][k]["prompts"]) for k in domain_keys)
    default_limit = min(3, smallest)
    print(f"\n{YELLOW}How many prompts per domain? (max {smallest}) [{default_limit}]:{RESET}")
    print(f"  {GREY}Example: 48 with 4 selected domains = 192 total generation jobs.{RESET}")
    limit_raw = input("  > ").strip() or str(default_limit)
    limit_per_domain = int(limit_raw) if limit_raw.isdigit() else default_limit
    total_jobs = len(domain_keys) * limit_per_domain
    print(f"  {GREEN}Planned jobs:{RESET} {len(domain_keys)} domains x {limit_per_domain} prompts = {total_jobs}")

    output_path, resume = _pick_output_path(domain_keys, pipeline_mode=pipeline_mode)
    print(f"\n  {GREEN}Output file:{RESET} {output_path.name}")
    print(f"  {GREEN}Mode:{RESET} {'resume' if resume else 'new'}")

    backend, model_alias = _pick_backend_and_model()
    if backend == "openai":
        planned_jobs = _build_generation_jobs(
            domain_keys=domain_keys,
            limit_per_domain=limit_per_domain,
            output_path=output_path,
            resume=resume,
            verbose=False,
            pipeline_mode=pipeline_mode,
        )
        resolved_model_id = resolve_model_id(model_alias)
        projection = _estimate_openai_run_cost(resolved_model_id, planned_jobs, pipeline_mode)
        print(f"\n{YELLOW}OpenAI cost projection before run:{RESET}")
        print(f"  {GREY}Model:{RESET} {resolved_model_id}")
        print(f"  {GREY}Pending jobs:{RESET} {projection['jobs']}  approx calls={projection['calls']}")
        print(
            f"  {GREY}Estimated tokens:{RESET} "
            f"in={projection['input_tokens']:,}  out={projection['output_tokens']:,}"
        )
        print(f"  {BOLD}Estimated cost:{RESET} {_format_cost(projection['cost_usd'])}")
        print(f"  {GREY}Actual final cost will use OpenAI response.usage tokens.{RESET}")
        if input("\n  Continue with this OpenAI run? [Y/n]: ").strip().lower() in ("n", "no"):
            print(f"\n{YELLOW}Cancelled before any generation call.{RESET}")
            return

    if backend == "ollama":
        generate_fn = load_ollama_gemma4(model=model_alias, thinking=False, use_memory=False)
        worker_factory = lambda: load_ollama_gemma4(model=model_alias, thinking=False, use_memory=False)
    elif backend == "openai":
        generate_fn = load_openai_generator(model=model_alias, thinking=False, use_memory=False)
        worker_factory = lambda: load_openai_generator(model=model_alias, thinking=False, use_memory=False)
    else:
        generate_fn = load_gemma4(model=model_alias, thinking=False, use_memory=False)
        worker_factory = None

    parallel_workers = _pick_parallel_workers(backend, pipeline_mode, model_alias)

    if backend == "ollama":
        if pipeline_mode == "advanced" and parallel_workers > 2:
            print(f"\n{YELLOW}Warning:{RESET} advanced + {model_alias} + {parallel_workers} workers may be slower than serial on this Mac.")
        elif pipeline_mode == "advanced" and parallel_workers == 1:
            print(f"\n{GREEN}Using serial mode for advanced generation.{RESET} Best for stability and more predictable progress.")
        elif pipeline_mode == "simple" and parallel_workers == 1:
            print(f"\n{GREEN}Using serial mode.{RESET} Slower overall, but easiest to reason about.")
    elif backend == "openai":
        if parallel_workers > 8:
            print(f"\n{YELLOW}Warning:{RESET} {parallel_workers} OpenAI workers may hit rate limits. Resume mode will protect saved progress.")
        else:
            print(f"\n{GREEN}Using OpenAI parallel mode:{RESET} {parallel_workers} worker(s).")

    try:
        all_results = generate_many(
            domain_keys=domain_keys,
            generate_fn=generate_fn,
            limit_per_domain=limit_per_domain,
            verbose=True,
            output_path=output_path,
            resume=resume,
            pipeline_mode=pipeline_mode,
            parallel_workers=parallel_workers,
            worker_factory=worker_factory,
        )
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Stopped by user. Saved progress remains in:{RESET} {output_path.resolve()}")
        return

    current_total = len(_read_jsonl(output_path))
    valid = sum(1 for r in _read_jsonl(output_path) if not r.get("_meta", {}).get("validation_errors"))
    print(f"\n  {GREEN}Done:{RESET} {current_total} total saved, {valid} fully valid")
    usage_total = _merge_usage([
        r.get("_meta", {}).get("openai_usage") for r in all_results
        if r.get("_meta", {}).get("openai_usage")
    ])
    if usage_total.get("requests"):
        print(f"  {GREEN}Actual OpenAI cost this run:{RESET} {_format_usage_line(usage_total)}")

    dataset_for_formatting = output_path
    print(f"\n{YELLOW}Augment this raw dataset with Gemma 4 before SFT formatting? [y/N]{RESET}")
    if input("  > ").strip().lower() in ("y", "yes"):
        from src.II_dataGen.augment import augment_jsonl

        default_rounds = 6
        default_batch_size = 4 if backend == "hf" else 6

        print(f"\n{YELLOW}Max augmentation rounds [{default_rounds}]:{RESET}")
        rounds_raw = input("  > ").strip() or str(default_rounds)
        max_rounds = int(rounds_raw) if rounds_raw.isdigit() else default_rounds

        print(f"\n{YELLOW}Target new examples per round [{default_batch_size}]:{RESET}")
        batch_raw = input("  > ").strip() or str(default_batch_size)
        batch_size = int(batch_raw) if batch_raw.isdigit() else default_batch_size

        summary = augment_jsonl(
            input_path=output_path,
            generator=generate_fn,
            max_rounds=max_rounds,
            batch_size=batch_size,
            verbose=True,
        )
        dataset_for_formatting = summary["output_path"]

    print(f"\n{YELLOW}Convert this dataset to SFT training format and create train/eval/test splits? [y/N]{RESET}")
    if input("  > ").strip().lower() in ("y", "yes"):
        from src.II_dataGen.format_sft import process_jsonl
        process_jsonl(dataset_for_formatting)


if __name__ == "__main__":
    _tui()
