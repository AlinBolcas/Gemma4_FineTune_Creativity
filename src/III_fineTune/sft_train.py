"""
sft_train.py - Config-driven SFT workflow for Gemma 4.

Local Mac:
- preflight your train/eval/test JSONL
- preview formatting and token lengths
- save portable run configs

Cloud / GPU:
- use the same saved config to run Unsloth or transformers training

Run:
    python src/III_fineTune/sft_train.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_INPUT_DIR = REPO_ROOT / "data" / "input"
TRAIN_DIR = DATA_INPUT_DIR / "train"
EVAL_DIR = DATA_INPUT_DIR / "eval"
TEST_DIR = DATA_INPUT_DIR / "test"
CONFIG_DIR = REPO_ROOT / "src" / "III_fineTune"
RUN_CONFIG_DIR = CONFIG_DIR / "configs" / "runs"
OUTPUT_DIR = REPO_ROOT / "data" / "output"
MODELS_DIR = OUTPUT_DIR / "models"
TRAINING_RUNS_DIR = OUTPUT_DIR / "training_runs"
PREFLIGHT_DIR = TRAINING_RUNS_DIR / "preflight"

CYAN = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; GREY = "\033[90m"
RED = "\033[91m"; BOLD = "\033[1m"; RESET = "\033[0m"


def _ensure_dirs():
    RUN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _run_output_dir(run_name: str) -> Path:
    path = TRAINING_RUNS_DIR / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_run_snapshot(run_name: str, config: dict, stage: str, extra: dict | None = None) -> Path:
    run_dir = _run_output_dir(run_name)
    snapshot = {
        "run_name": run_name,
        "stage": stage,
        "saved_at": datetime.now().isoformat(),
        "config": config,
    }
    if extra:
        snapshot["extra"] = extra
    out_path = run_dir / f"{stage}.json"
    _write_json(out_path, snapshot)
    return out_path


def _safe_json_dump(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _read_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    if not path or not path.exists():
        return items
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _pick(prompt: str, options: list[str], default: int = 1) -> str:
    print(f"\n{YELLOW}{prompt}{RESET}")
    for i, item in enumerate(options, 1):
        mark = f" {GREY}<- default{RESET}" if i == default else ""
        print(f"  [{i}] {item}{mark}")
    raw = input(f"\n  Choice [{default}]: ").strip() or str(default)
    if raw.isdigit() and 1 <= int(raw) <= len(options):
        return options[int(raw) - 1]
    return options[default - 1]


def _ask(prompt: str, default: str = "") -> str:
    shown = f" [{default}]" if default else ""
    raw = input(f"\n  {prompt}{shown}: ").strip()
    return raw or default


def _confirm(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = input(f"\n  {prompt} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def discover_dataset_bundles() -> list[dict]:
    """
    Discover split datasets by matching:
      train/<base>_train.jsonl
      eval/<base>_eval.jsonl
      test/<base>_test.jsonl
      input/<base>_sft.jsonl
    """
    bundles = []
    for train_path in sorted(TRAIN_DIR.glob("*_train.jsonl")):
        base = train_path.name[:-len("_train.jsonl")]
        eval_path = EVAL_DIR / f"{base}_eval.jsonl"
        test_path = TEST_DIR / f"{base}_test.jsonl"
        combined_path = DATA_INPUT_DIR / f"{base}_sft.jsonl"

        bundle = {
            "base": base,
            "combined_path": combined_path if combined_path.exists() else None,
            "train_path": train_path,
            "eval_path": eval_path if eval_path.exists() else None,
            "test_path": test_path if test_path.exists() else None,
            "train_count": len(_read_jsonl(train_path)),
            "eval_count": len(_read_jsonl(eval_path)) if eval_path.exists() else 0,
            "test_count": len(_read_jsonl(test_path)) if test_path.exists() else 0,
        }
        bundles.append(bundle)
    return bundles


def _resolve_unsloth_model_id(alias: str, hf_model_id: str) -> str:
    from src.IV_inference.gemma4_integration import MODELS, resolve_model_id
    if alias in MODELS:
        return "unsloth/" + resolve_model_id(alias).split("/", 1)[1]
    if hf_model_id.startswith("google/"):
        return "unsloth/" + hf_model_id.split("/", 1)[1]
    return hf_model_id


def build_run_config(bundle: dict, model_alias: str = "e2b", backend: str = "unsloth", run_name: str | None = None) -> dict:
    from src.IV_inference.gemma4_integration import resolve_model_id

    hf_model_id = resolve_model_id(model_alias)
    unsloth_model_id = _resolve_unsloth_model_id(model_alias, hf_model_id)
    run_name = run_name or f"{bundle['base']}_{model_alias}_{backend}"

    return {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(),
        "notes": "Generated by src/III_fineTune/sft_train.py",
        "data": {
            "combined_path": str(bundle["combined_path"]) if bundle.get("combined_path") else "",
            "train_path": str(bundle["train_path"]) if bundle.get("train_path") else "",
            "eval_path": str(bundle["eval_path"]) if bundle.get("eval_path") else "",
            "test_path": str(bundle["test_path"]) if bundle.get("test_path") else "",
        },
        "model": {
            "alias": model_alias,
            "hf_model_id": hf_model_id,
            "unsloth_model_id": unsloth_model_id,
            "chat_template": "gemma-4",
            "thinking": False,
        },
        "training": {
            "backend": backend,
            "output_dir": f"data/output/models/{run_name}",
            "load_in_4bit": True,
            "full_finetuning": False,
            "max_seq_length": 4096,
            "num_train_epochs": 3,
            "max_steps": 0,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "warmup_steps": 5,
            "weight_decay": 0.001,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.0,
            "seed": 3407,
            "report_to": "none",
        },
        "runtime": {
            "target": "cloud",
            "preflight_sample_count": 3,
            "post_train_eval": False,
        },
    }


def save_run_config(config: dict) -> Path:
    _ensure_dirs()
    path = RUN_CONFIG_DIR / f"{config['run_name']}.json"
    _write_json(path, config)
    return path


def load_run_config(path: Path) -> dict:
    return _read_json(path)


def _select_bundle() -> dict:
    bundles = discover_dataset_bundles()
    if not bundles:
        raise FileNotFoundError("No dataset bundles found in data/input/train, eval, test")

    print(f"\n{YELLOW}Available dataset bundles:{RESET}")
    for i, bundle in enumerate(bundles, 1):
        print(
            f"  [{i}] {bundle['base']}  "
            f"train={bundle['train_count']} eval={bundle['eval_count']} test={bundle['test_count']}"
        )
    raw = input("\n  Choice [1]: ").strip() or "1"
    idx = int(raw) - 1 if raw.isdigit() and 1 <= int(raw) <= len(bundles) else 0
    return bundles[idx]


def _select_existing_config() -> Path:
    _ensure_dirs()
    configs = sorted(RUN_CONFIG_DIR.glob("*.json"))
    if not configs:
        raise FileNotFoundError("No saved run configs found yet.")
    print(f"\n{YELLOW}Saved run configs:{RESET}")
    for i, path in enumerate(configs, 1):
        print(f"  [{i}] {path.name}")
    raw = input("\n  Choice [1]: ").strip() or "1"
    idx = int(raw) - 1 if raw.isdigit() and 1 <= int(raw) <= len(configs) else 0
    return configs[idx]


def _build_config_tui() -> Path:
    from src.IV_inference.gemma4_integration import MODELS

    bundle = _select_bundle()
    print(f"\n{GREY}Hackathon final: pick e4b. Local smoke: e2b is lighter.{RESET}")
    alias = _pick("Model alias:", list(MODELS.keys()), default=1)
    backend = _pick("Training backend:", ["unsloth", "transformers"], default=1)
    run_name = _ask("Run name", f"{bundle['base']}_{alias}_{backend}")

    config = build_run_config(bundle=bundle, model_alias=alias, backend=backend, run_name=run_name)

    config["training"]["max_seq_length"] = int(_ask("Max sequence length", str(config["training"]["max_seq_length"])))
    config["training"]["num_train_epochs"] = int(_ask("Epochs", str(config["training"]["num_train_epochs"])))
    config["training"]["per_device_train_batch_size"] = int(
        _ask("Batch size", str(config["training"]["per_device_train_batch_size"]))
    )
    config["training"]["gradient_accumulation_steps"] = int(
        _ask("Gradient accumulation", str(config["training"]["gradient_accumulation_steps"]))
    )
    config["training"]["learning_rate"] = float(_ask("Learning rate", str(config["training"]["learning_rate"])))
    config["runtime"]["preflight_sample_count"] = int(
        _ask("Preflight sample count", str(config["runtime"]["preflight_sample_count"]))
    )

    path = save_run_config(config)
    print(f"\n{GREEN}Saved config:{RESET} {path.resolve()}")
    return path


def run_preflight(config: dict) -> dict:
    """
    Local-safe dataset and tokenizer preflight.
    Loads processor/tokenizer only, never launches real training.
    """
    from transformers import AutoProcessor

    train_path = Path(config["data"]["train_path"]) if config["data"]["train_path"] else None
    eval_path = Path(config["data"]["eval_path"]) if config["data"]["eval_path"] else None
    test_path = Path(config["data"]["test_path"]) if config["data"]["test_path"] else None

    train_examples = _read_jsonl(train_path) if train_path else []
    eval_examples = _read_jsonl(eval_path) if eval_path else []
    test_examples = _read_jsonl(test_path) if test_path else []

    if not train_examples:
        raise ValueError("Train split is empty or missing.")

    hf_model_id = config["model"]["hf_model_id"]
    sample_n = int(config["runtime"].get("preflight_sample_count", 3))
    sample_examples = train_examples[:sample_n]

    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"  {BOLD}Local Preflight{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"  run_name: {config['run_name']}")
    print(f"  model:    {hf_model_id}")
    print(f"  backend:  {config['training']['backend']}")
    print(f"  train:    {len(train_examples)}")
    print(f"  eval:     {len(eval_examples)}")
    print(f"  test:     {len(test_examples)}")

    processor = AutoProcessor.from_pretrained(hf_model_id)

    lengths = []
    system_counts = 0
    for ex in sample_examples:
        messages = ex["messages"]
        if messages and messages[0].get("role") == "system":
            system_counts += 1
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        tokenized = processor(text=text, return_tensors="pt")
        lengths.append(int(tokenized["input_ids"].shape[-1]))

    summary = {
        "run_name": config["run_name"],
        "model": hf_model_id,
        "backend": config["training"]["backend"],
        "train_count": len(train_examples),
        "eval_count": len(eval_examples),
        "test_count": len(test_examples),
        "sample_count": len(sample_examples),
        "sample_token_min": min(lengths) if lengths else 0,
        "sample_token_avg": round(mean(lengths), 1) if lengths else 0,
        "sample_token_max": max(lengths) if lengths else 0,
        "max_seq_length": config["training"]["max_seq_length"],
        "system_prompt_rows_in_sample": system_counts,
    }

    print(f"\n{GREEN}Token lengths:{RESET}")
    print(f"  min={summary['sample_token_min']}  avg={summary['sample_token_avg']}  max={summary['sample_token_max']}")
    if summary["sample_token_max"] > summary["max_seq_length"]:
        print(f"  {RED}Warning:{RESET} sample exceeds max_seq_length")
    else:
        print(f"  {GREEN}OK:{RESET} sample fits max_seq_length")

    print(f"\n{GREEN}Training row style:{RESET}")
    print(f"  system rows in sample: {system_counts}/{len(sample_examples)}")
    print(f"  thinking flag in config: {config['model']['thinking']}")
    print(f"  chat template: {config['model']['chat_template']}")

    preflight_path = PREFLIGHT_DIR / f"{config['run_name']}_preflight.json"
    _write_json(preflight_path, summary)
    _save_run_snapshot(config["run_name"], config, "preflight", summary)
    print(f"\n{GREEN}Saved preflight:{RESET} {preflight_path.resolve()}")
    return summary


def _format_dataset_for_unsloth(examples: list[dict], tokenizer):
    from datasets import Dataset

    def format_example(ex):
        text = tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        ).removeprefix("<bos>")
        return {"text": text}

    return Dataset.from_list(examples).map(format_example)


def _format_dataset_for_transformers(examples: list[dict], processor):
    from datasets import Dataset

    def format_example(ex):
        text = processor.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    return Dataset.from_list(examples).map(format_example)


def _train_unsloth(config: dict):
    if not torch.cuda.is_available():
        print(f"\n{YELLOW}Unsloth training is intended for CUDA GPUs.{RESET}")
        print("  Use this config on Colab / Kaggle. Local Mac should use preflight only.")
        return

    try:
        from unsloth import FastModel
        from unsloth.chat_templates import get_chat_template, train_on_responses_only
        from trl import SFTTrainer, SFTConfig
    except Exception as e:
        print(f"\n{YELLOW}Unsloth is not available in this environment.{RESET}")
        print(f"  Import error: {e}")
        print("  Use this config on Colab / Kaggle where Unsloth is installed.")
        return

    train_examples = _read_jsonl(Path(config["data"]["train_path"]))
    eval_examples = _read_jsonl(Path(config["data"]["eval_path"])) if config["data"]["eval_path"] else []
    model_id = config["model"]["unsloth_model_id"]
    output_dir = str(REPO_ROOT / config["training"]["output_dir"])
    run_dir = _run_output_dir(config["run_name"])
    _save_run_snapshot(config["run_name"], config, "training_start", {
        "backend": "unsloth",
        "train_count": len(train_examples),
        "eval_count": len(eval_examples),
        "output_dir": output_dir,
    })

    print(f"\nLoading {model_id} with Unsloth...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_id,
        dtype=None,
        max_seq_length=config["training"]["max_seq_length"],
        load_in_4bit=config["training"]["load_in_4bit"],
        full_finetuning=config["training"]["full_finetuning"],
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=config["training"]["lora_r"],
        lora_alpha=config["training"]["lora_alpha"],
        lora_dropout=config["training"]["lora_dropout"],
        bias="none",
        random_state=config["training"]["seed"],
    )

    tokenizer = get_chat_template(tokenizer, chat_template=config["model"]["chat_template"])
    train_dataset = _format_dataset_for_unsloth(train_examples, tokenizer)
    eval_dataset = _format_dataset_for_unsloth(eval_examples, tokenizer) if eval_examples else None

    sft_args = {
        "dataset_text_field": "text",
        "per_device_train_batch_size": config["training"]["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
        "warmup_steps": config["training"]["warmup_steps"],
        "learning_rate": config["training"]["learning_rate"],
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": config["training"]["weight_decay"],
        "lr_scheduler_type": "linear",
        "seed": config["training"]["seed"],
        "report_to": config["training"]["report_to"],
        "output_dir": output_dir,
    }
    if config["training"]["max_steps"] and int(config["training"]["max_steps"]) > 0:
        sft_args["max_steps"] = int(config["training"]["max_steps"])
    else:
        sft_args["num_train_epochs"] = int(config["training"]["num_train_epochs"])

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(**sft_args),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    print("\nStarting Unsloth training...")
    train_result = trainer.train()
    train_metrics = getattr(train_result, "metrics", {}) or {}

    # Save adapter immediately after training so post-train eval issues never lose the run.
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"{GREEN}Saved model:{RESET} {output_dir}")

    eval_metrics = {}
    if eval_dataset is not None and config["runtime"].get("post_train_eval", False):
        try:
            eval_metrics = trainer.evaluate()
        except Exception as e:
            eval_metrics = {"eval_skipped": True, "error": str(e)}
            print(f"{YELLOW}Post-train eval skipped:{RESET} {e}")
    elif eval_dataset is not None:
        eval_metrics = {"eval_skipped": True, "reason": "post_train_eval disabled in config"}

    _safe_json_dump(run_dir / "train_metrics.json", train_metrics)
    _safe_json_dump(run_dir / "eval_metrics.json", eval_metrics)
    _safe_json_dump(run_dir / "log_history.json", getattr(trainer.state, "log_history", []))
    try:
        trainer.state.save_to_json(str(run_dir / "trainer_state.json"))
    except Exception:
        _safe_json_dump(run_dir / "trainer_state.json", getattr(trainer.state, "__dict__", {}))
    _save_run_snapshot(config["run_name"], config, "training_complete", {
        "backend": "unsloth",
        "model_output_dir": output_dir,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    })
    print(f"{GREEN}Saved stats:{RESET} {run_dir}")


def _train_transformers(config: dict):
    if not torch.cuda.is_available():
        print(f"\n{YELLOW}Transformers QLoRA training is not practical on this local Mac setup.{RESET}")
        print("  Use preflight locally, then run the same config on a CUDA machine.")
        return

    try:
        from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
        import gc
    except Exception as e:
        print(f"\n{RED}Training dependencies missing:{RESET} {e}")
        return

    train_examples = _read_jsonl(Path(config["data"]["train_path"]))
    eval_examples = _read_jsonl(Path(config["data"]["eval_path"])) if config["data"]["eval_path"] else []
    model_id = config["model"]["hf_model_id"]
    output_dir = str(REPO_ROOT / config["training"]["output_dir"])
    run_dir = _run_output_dir(config["run_name"])
    _save_run_snapshot(config["run_name"], config, "training_start", {
        "backend": "transformers",
        "train_count": len(train_examples),
        "eval_count": len(eval_examples),
        "output_dir": output_dir,
    })

    processor = AutoProcessor.from_pretrained(model_id)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config["training"]["lora_r"],
        lora_alpha=config["training"]["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=config["training"]["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    train_dataset = _format_dataset_for_transformers(train_examples, processor)
    eval_dataset = _format_dataset_for_transformers(eval_examples, processor) if eval_examples else None

    train_args = {
        "output_dir": output_dir,
        "per_device_train_batch_size": config["training"]["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
        "learning_rate": config["training"]["learning_rate"],
        "weight_decay": config["training"]["weight_decay"],
        "warmup_steps": config["training"]["warmup_steps"],
        "logging_steps": 1,
        "save_strategy": "epoch",
        "bf16": True,
        "max_grad_norm": 0.3,
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit",
        "report_to": config["training"]["report_to"],
    }
    if config["training"]["max_steps"] and int(config["training"]["max_steps"]) > 0:
        train_args["max_steps"] = int(config["training"]["max_steps"])
        train_args["num_train_epochs"] = 1
    else:
        train_args["num_train_epochs"] = int(config["training"]["num_train_epochs"])

    trainer = SFTTrainer(
        model=model,
        args=TrainingArguments(**train_args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("\nStarting transformers QLoRA training...")
    train_result = trainer.train()
    train_metrics = getattr(train_result, "metrics", {}) or {}

    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"{GREEN}Saved model:{RESET} {output_dir}")

    eval_metrics = {}
    if eval_dataset is not None and config["runtime"].get("post_train_eval", False):
        try:
            eval_metrics = trainer.evaluate()
        except Exception as e:
            eval_metrics = {"eval_skipped": True, "error": str(e)}
            print(f"{YELLOW}Post-train eval skipped:{RESET} {e}")
    elif eval_dataset is not None:
        eval_metrics = {"eval_skipped": True, "reason": "post_train_eval disabled in config"}

    _safe_json_dump(run_dir / "train_metrics.json", train_metrics)
    _safe_json_dump(run_dir / "eval_metrics.json", eval_metrics)
    _safe_json_dump(run_dir / "log_history.json", getattr(trainer.state, "log_history", []))
    try:
        trainer.state.save_to_json(str(run_dir / "trainer_state.json"))
    except Exception:
        _safe_json_dump(run_dir / "trainer_state.json", getattr(trainer.state, "__dict__", {}))
    _save_run_snapshot(config["run_name"], config, "training_complete", {
        "backend": "transformers",
        "model_output_dir": output_dir,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    })
    print(f"{GREEN}Saved stats:{RESET} {run_dir}")

    gc.collect()
    torch.cuda.empty_cache()


def train_from_config(config: dict):
    backend = config["training"]["backend"]
    if backend == "unsloth":
        _train_unsloth(config)
        return
    if backend == "transformers":
        _train_transformers(config)
        return
    raise ValueError(f"Unknown backend: {backend}")


def _print_cloud_instructions(config_path: Path, config: dict):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"  {BOLD}Cloud / GPU Handoff{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"  Config:   {config_path.resolve()}")
    print(f"  Backend:  {config['training']['backend']}")
    print(f"  Model:    {config['model']['hf_model_id']}")
    print(f"  Train:    {config['data']['train_path']}")
    print(f"  Eval:     {config['data']['eval_path'] or '(none)'}")
    print(f"  Output:   {config['training']['output_dir']}")
    print("\n  Recommended:")
    print("  1. Prefer Kaggle first since the competition lives there")
    print("  2. Copy this repo or at minimum the config + data/input/ splits to Kaggle / Colab")
    print("  3. Install Unsloth there")
    print("  4. Run this same script and choose 'Train from existing config'")
    print("  5. Trained adapters save to data/output/models/<run_name>")
    print("  6. Metrics / logs save to data/output/training_runs/<run_name>")


def _tui():
    from src.IV_inference.gemma4_integration import MODELS

    _ensure_dirs()
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"  {BOLD}Gemma 4 Fine-Tune{RESET}")
    print("  local preflight + portable config + cloud training")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

    actions = [
        "Build new run config",
        "Local preflight existing config",
        "Train from existing config",
        "Show dataset bundles",
        "Exit",
    ]
    choice = _pick("What do you want to do?", actions, default=1)

    if choice == "Show dataset bundles":
        bundles = discover_dataset_bundles()
        if not bundles:
            print(f"\n{YELLOW}No dataset bundles found yet.{RESET}")
        else:
            print(f"\n{YELLOW}Dataset bundles:{RESET}")
            for bundle in bundles:
                print(
                    f"  - {bundle['base']}  "
                    f"train={bundle['train_count']} eval={bundle['eval_count']} test={bundle['test_count']}"
                )
        return

    if choice == "Build new run config":
        config_path = _build_config_tui()
        config = load_run_config(config_path)
        if _confirm("Run local preflight now?", True):
            run_preflight(config)
        _print_cloud_instructions(config_path, config)
        return

    if choice == "Local preflight existing config":
        config_path = _select_existing_config()
        config = load_run_config(config_path)
        run_preflight(config)
        return

    if choice == "Train from existing config":
        config_path = _select_existing_config()
        config = load_run_config(config_path)
        print(f"\n{YELLOW}Run local preflight before training? [Y/n]{RESET}")
        if input("  > ").strip().lower() not in ("n", "no"):
            run_preflight(config)
        train_from_config(config)
        return


if __name__ == "__main__":
    _tui()
