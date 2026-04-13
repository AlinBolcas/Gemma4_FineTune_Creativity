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

    start = time.time()
    try:
        example = run_pipeline(
            task=job["task"],
            generate_fn=local_generate_fn,
            domain=job["domain_label"],
            verbose=False,
        )
        elapsed = round(time.time() - start, 1)
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
        }
        if on_finish:
            on_finish(job, elapsed, False)
        return {"job": job, "example": example, "elapsed": elapsed, "errors": errors}
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
                all_results.append(example)

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
                    if errors:
                        print(f"  {YELLOW}[QA] Validation: {errors}{RESET}")
                    if output_path:
                        print(f"  {GREEN}Saved example -> {output_path.name}{RESET}")
                    print(f"{'#'*60}")
    finally:
        stop_event.set()
        if monitor_thread:
            monitor_thread.join(timeout=1)

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


def _pick_parallel_workers(backend: str, pipeline_mode: str, model_alias: str) -> int:
    if backend != "ollama":
        print(f"\n{GREY}Parallel workers are disabled for Hugging Face on local Mac. One in-process model is safer.{RESET}")
        return 1

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
        worker_factory = lambda: load_ollama_gemma4(model=model_alias, thinking=False, use_memory=False)
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
