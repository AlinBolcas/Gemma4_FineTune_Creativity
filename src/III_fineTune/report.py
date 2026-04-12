"""
report.py - Generate local training reports from saved runs.

Works with either:
1. data/output/training_runs/<run_name>/
2. data/output/models/<run_name>/checkpoint-*/trainer_state.json

Outputs:
- markdown summary
- loss / lr / grad_norm plots

Run:
    python src/III_fineTune/report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "data" / "output"
MODELS_DIR = OUTPUT_DIR / "models"
TRAINING_RUNS_DIR = OUTPUT_DIR / "training_runs"

CYAN = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; GREY = "\033[90m"
RED = "\033[91m"; BOLD = "\033[1m"; RESET = "\033[0m"


def _read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_name(path: Path) -> str:
    return path.name if path else "(missing)"


def _read_snapshot(run_dir: Path | None) -> dict:
    if not run_dir or not run_dir.exists():
        return {}
    for name in ("training_complete.json", "training_start.json"):
        path = run_dir / name
        if path.exists():
            try:
                return _read_json(path)
            except Exception:
                return {}
    return {}


def discover_report_targets() -> list[dict]:
    """
    Return possible report sources.

    Prefers explicit training_runs dirs, but also scans model checkpoint dirs so
    copied Kaggle outputs still work even if training_runs was not extracted.
    """
    targets = []

    # Explicit run dirs
    if TRAINING_RUNS_DIR.exists():
        for run_dir in sorted([p for p in TRAINING_RUNS_DIR.glob("*") if p.is_dir()]):
            if run_dir.name == "preflight":
                continue
            targets.append(
                {
                    "name": run_dir.name,
                    "run_dir": run_dir,
                    "model_dir": MODELS_DIR / run_dir.name,
                    "source": "training_runs",
                }
            )

    # Fallback from model checkpoints or bare adapter dirs
    if MODELS_DIR.exists():
        for model_dir in sorted([p for p in MODELS_DIR.glob("*") if p.is_dir()]):
            if any(t["name"] == model_dir.name for t in targets):
                continue
            checkpoint_dirs = sorted([p for p in model_dir.glob("checkpoint-*") if p.is_dir()])
            if checkpoint_dirs:
                targets.append(
                    {
                        "name": model_dir.name,
                        "run_dir": None,
                        "model_dir": model_dir,
                        "checkpoint_dir": checkpoint_dirs[-1],
                        "source": "checkpoint_only",
                    }
                )
                continue
            if any((model_dir / name).exists() for name in ("adapter_config.json", "tokenizer_config.json", "config.json")):
                targets.append(
                    {
                        "name": model_dir.name,
                        "run_dir": None,
                        "model_dir": model_dir,
                        "checkpoint_dir": None,
                        "source": "model_only",
                    }
                )

    return targets


def _pick_target() -> dict | None:
    targets = discover_report_targets()
    if not targets:
        print(f"{YELLOW}No training targets found under data/output/models or data/output/training_runs.{RESET}")
        return None

    print(f"\n{YELLOW}Available report targets:{RESET}")
    for i, t in enumerate(targets, 1):
        source = t["source"]
        print(f"  [{i}] {t['name']}  ({source})")

    raw = input("\n  Choice [1]: ").strip() or "1"
    idx = int(raw) - 1 if raw.isdigit() and 1 <= int(raw) <= len(targets) else 0
    return targets[idx]


def _load_metrics(target: dict) -> dict:
    """
    Load the best available metrics/log history from either training_runs or checkpoint.
    """
    run_dir = target.get("run_dir")
    model_dir = target.get("model_dir")
    checkpoint_dir = target.get("checkpoint_dir")

    result = {
        "name": target["name"],
        "train_metrics": {},
        "eval_metrics": {},
        "log_history": [],
        "trainer_state": {},
        "config": {},
        "source": target["source"],
        "run_dir": str(run_dir) if run_dir else "",
        "model_dir": str(model_dir) if model_dir else "",
    }

    if run_dir and run_dir.exists():
        snapshot = _read_snapshot(run_dir)
        result["config"] = snapshot.get("config", {}) if snapshot else {}
        train_metrics = run_dir / "train_metrics.json"
        eval_metrics = run_dir / "eval_metrics.json"
        log_history = run_dir / "log_history.json"
        trainer_state = run_dir / "trainer_state.json"

        if train_metrics.exists():
            result["train_metrics"] = _read_json(train_metrics)
        if eval_metrics.exists():
            result["eval_metrics"] = _read_json(eval_metrics)
        if log_history.exists():
            result["log_history"] = _read_json(log_history)
        if trainer_state.exists():
            result["trainer_state"] = _read_json(trainer_state)

    # Fallback to checkpoint trainer_state
    if not result["log_history"] and checkpoint_dir:
        state_path = checkpoint_dir / "trainer_state.json"
        if state_path.exists():
            trainer_state = _read_json(state_path)
            result["trainer_state"] = trainer_state
            result["log_history"] = trainer_state.get("log_history", [])

    return result


def _make_plots(metrics: dict, report_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"{YELLOW}matplotlib is not available. Skipping plots.{RESET}")
        print(f"{GREY}Install with: pip install matplotlib{RESET}")
        print(f"{GREY}Import error: {e}{RESET}")
        return []

    log_history = metrics.get("log_history", []) or []
    if not log_history:
        print(f"{YELLOW}No log history found. Skipping plots.{RESET}")
        return []

    report_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # Prepare series
    steps_loss, loss_vals = [], []
    steps_eval, eval_vals = [], []
    steps_lr, lr_vals = [], []
    steps_gn, gn_vals = [], []

    for row in log_history:
        step = row.get("step")
        if step is None:
            continue
        if row.get("loss") is not None:
            steps_loss.append(step)
            loss_vals.append(row["loss"])
        if row.get("eval_loss") is not None:
            steps_eval.append(step)
            eval_vals.append(row["eval_loss"])
        if row.get("learning_rate") is not None:
            steps_lr.append(step)
            lr_vals.append(row["learning_rate"])
        if row.get("grad_norm") is not None:
            steps_gn.append(step)
            gn_vals.append(row["grad_norm"])

    if loss_vals:
        plt.figure(figsize=(10, 5))
        plt.plot(steps_loss, loss_vals, marker="o", label="train loss")
        if eval_vals:
            plt.plot(steps_eval, eval_vals, marker="s", label="eval loss")
        plt.title("Training / Eval Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        loss_path = report_dir / "loss.png"
        plt.savefig(loss_path, bbox_inches="tight", dpi=160)
        plt.close()
        generated.append(loss_path)

    if lr_vals:
        plt.figure(figsize=(10, 4))
        plt.plot(steps_lr, lr_vals, marker=".")
        plt.title("Learning Rate")
        plt.xlabel("Step")
        plt.ylabel("LR")
        plt.grid(True)
        lr_path = report_dir / "learning_rate.png"
        plt.savefig(lr_path, bbox_inches="tight", dpi=160)
        plt.close()
        generated.append(lr_path)

    if gn_vals:
        plt.figure(figsize=(10, 4))
        plt.plot(steps_gn, gn_vals, marker=".")
        plt.title("Gradient Norm")
        plt.xlabel("Step")
        plt.ylabel("Grad Norm")
        plt.grid(True)
        gn_path = report_dir / "grad_norm.png"
        plt.savefig(gn_path, bbox_inches="tight", dpi=160)
        plt.close()
        generated.append(gn_path)

    return generated


def _make_markdown(metrics: dict, report_dir: Path) -> Path:
    train_metrics = metrics.get("train_metrics", {}) or {}
    eval_metrics = metrics.get("eval_metrics", {}) or {}
    trainer_state = metrics.get("trainer_state", {}) or {}
    log_history = metrics.get("log_history", []) or []
    config = metrics.get("config", {}) or {}
    model_cfg = config.get("model", {}) or {}
    data_cfg = config.get("data", {}) or {}
    training_cfg = config.get("training", {}) or {}

    lines = [
        f"# Fine-Tune Report: {metrics['name']}",
        "",
        f"**Source:** {metrics['source']}",
        f"**Model dir:** `{metrics.get('model_dir', '')}`",
    ]
    if metrics.get("run_dir"):
        lines.append(f"**Run dir:** `{metrics.get('run_dir', '')}`")
    lines.extend(["", "## Summary", ""])

    if config:
        lines.extend(["", "## Config", ""])
        if model_cfg:
            lines.append(f"- `model.alias`: {model_cfg.get('alias', 'n/a')}")
            lines.append(f"- `model.hf_model_id`: {model_cfg.get('hf_model_id', 'n/a')}")
        if training_cfg:
            lines.append(f"- `training.backend`: {training_cfg.get('backend', 'n/a')}")
            lines.append(f"- `training.output_dir`: {training_cfg.get('output_dir', 'n/a')}")
            lines.append(f"- `training.max_seq_length`: {training_cfg.get('max_seq_length', 'n/a')}")
        if data_cfg:
            lines.append(f"- `data.train_path`: {data_cfg.get('train_path', 'n/a')}")
            lines.append(f"- `data.eval_path`: {data_cfg.get('eval_path', 'n/a')}")
            lines.append(f"- `data.test_path`: {data_cfg.get('test_path', 'n/a')}")

    if trainer_state:
        lines.append(f"- `global_step`: {trainer_state.get('global_step', 'n/a')}")
        lines.append(f"- `max_steps`: {trainer_state.get('max_steps', 'n/a')}")
        lines.append(f"- `num_train_epochs`: {trainer_state.get('num_train_epochs', 'n/a')}")
        lines.append(f"- `best_metric`: {trainer_state.get('best_metric', 'n/a')}")

    if train_metrics:
        lines.extend(["", "## Train Metrics", ""])
        for k, v in train_metrics.items():
            lines.append(f"- `{k}`: {v}")

    if eval_metrics:
        lines.extend(["", "## Eval Metrics", ""])
        for k, v in eval_metrics.items():
            lines.append(f"- `{k}`: {v}")

    if log_history:
        lines.extend(["", "## Last Logged Steps", "", "| Step | Loss | Eval Loss | LR | Grad Norm |", "|---|---:|---:|---:|---:|"])
        for row in log_history[-10:]:
            lines.append(
                f"| {row.get('step','')} | {row.get('loss','')} | {row.get('eval_loss','')} | "
                f"{row.get('learning_rate','')} | {row.get('grad_norm','')} |"
            )

    lines.extend(
        [
            "",
            "## Plot Files",
            "",
            "- `loss.png`",
            "- `learning_rate.png`",
            "- `grad_norm.png`",
            "",
        ]
    )

    out_path = report_dir / "report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def generate_report(target: dict) -> dict:
    metrics = _load_metrics(target)
    report_dir = OUTPUT_DIR / "reports" / metrics["name"]
    report_dir.mkdir(parents=True, exist_ok=True)

    plots = _make_plots(metrics, report_dir)
    report_md = _make_markdown(metrics, report_dir)

    summary = {
        "name": metrics["name"],
        "source": metrics["source"],
        "report_dir": str(report_dir),
        "report_md": str(report_md),
        "plots": [str(p) for p in plots],
        "has_log_history": bool(metrics.get("log_history")),
    }

    with open(report_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def _tui():
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"  {BOLD}Fine-Tune Report{RESET}")
    print("  plots + markdown from training outputs")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

    target = _pick_target()
    if not target:
        print(f"{GREY}Tip: download your Kaggle run into data/output/models and ideally data/output/training_runs too.{RESET}")
        return

    summary = generate_report(target)
    print(f"\n{GREEN}Report dir:{RESET} {summary['report_dir']}")
    print(f"{GREEN}Markdown:{RESET}  {summary['report_md']}")
    if summary["plots"]:
        print(f"{GREEN}Plots:{RESET}")
        for p in summary["plots"]:
            print(f"  - {p}")
    else:
        print(f"{YELLOW}No plots were generated.{RESET}")


if __name__ == "__main__":
    _tui()
