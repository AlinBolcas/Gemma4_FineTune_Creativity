"""
sft_train.py - Gemma 4 SFT training script for Colab / Kaggle.

This is the code-based entrypoint for fine-tuning Gemma 4 on our
full-loop creative reasoning traces.

Run on Colab/Kaggle with GPU. NOT for local Mac execution.

Usage:
    python sft_train.py --data path/to/sft_data.jsonl --model google/gemma-4-E4B-it
"""

import json
import argparse
import torch
import gc
from pathlib import Path


def load_sft_data(path: str) -> list[dict]:
    """Load chat-format JSONL training data."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    print(f"Loaded {len(examples)} training examples from {path}")
    return examples


def train(
    data_path: str,
    model_id: str = "google/gemma-4-E4B-it",
    output_dir: str = "./gemma4-creative-tuned",
    max_steps: int = 0,
    num_epochs: int = 3,
    use_unsloth: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    learning_rate: float = 2e-4,
    batch_size: int = 1,
    grad_accum: int = 4,
    max_seq_length: int = 4096,
):
    """
    Fine-tune Gemma 4 on creative reasoning traces.

    Supports two paths:
      1. Unsloth (preferred on Colab/Kaggle for speed + VRAM savings)
      2. Standard transformers + peft + trl (fallback)
    """
    examples = load_sft_data(data_path)

    if use_unsloth:
        _train_unsloth(
            examples, model_id, output_dir,
            max_steps, num_epochs,
            lora_r, lora_alpha,
            learning_rate, batch_size, grad_accum,
            max_seq_length,
        )
    else:
        _train_standard(
            examples, model_id, output_dir,
            max_steps, num_epochs,
            lora_r, lora_alpha,
            learning_rate, batch_size, grad_accum,
        )


# ---------------------------------------------------------------------------
# Path 1: Unsloth (Colab/Kaggle preferred)
# ---------------------------------------------------------------------------

def _train_unsloth(
    examples, model_id, output_dir,
    max_steps, num_epochs,
    lora_r, lora_alpha,
    learning_rate, batch_size, grad_accum,
    max_seq_length,
):
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    print(f"Loading {model_id} with Unsloth...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_id,
        dtype=None,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )

    # Apply Gemma 4 chat template (non-thinking, since our traces are visible)
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    # Format dataset
    def format_example(ex):
        text = tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        ).removeprefix("<bos>")
        return {"text": text}

    dataset = Dataset.from_list(examples).map(format_example)
    print(f"Dataset formatted: {len(dataset)} examples")

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=5,
            num_train_epochs=num_epochs if max_steps == 0 else 1,
            max_steps=max_steps if max_steps > 0 else -1,
            learning_rate=learning_rate,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
            output_dir=output_dir,
        ),
    )

    # Train only on assistant responses
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    print("Starting training...")
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


# ---------------------------------------------------------------------------
# Path 2: Standard transformers + peft (fallback)
# ---------------------------------------------------------------------------

def _train_standard(
    examples, model_id, output_dir,
    max_steps, num_epochs,
    lora_r, lora_alpha,
    learning_rate, batch_size, grad_accum,
):
    from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset
    from trl import SFTTrainer

    print(f"Loading {model_id} with QLoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Format dataset
    def format_example(ex):
        text = processor.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(examples).map(format_example)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs if max_steps == 0 else 1,
        max_steps=max_steps if max_steps > 0 else -1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 on creative reasoning traces")
    parser.add_argument("--data", required=True, help="Path to SFT JSONL")
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--output", default="./gemma4-creative-tuned")
    parser.add_argument("--unsloth", action="store_true", help="Use Unsloth backend")
    parser.add_argument("--max-steps", type=int, default=0, help="Max steps (0 = use epochs)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    args = parser.parse_args()

    train(
        data_path=args.data,
        model_id=args.model,
        output_dir=args.output,
        max_steps=args.max_steps,
        num_epochs=args.epochs,
        use_unsloth=args.unsloth,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
