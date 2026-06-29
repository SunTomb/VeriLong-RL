"""LoRA SFT for Qwen2.5-7B-Instruct on VeriLong-RL SFT chat data.

Runs on a single A40 (bf16 LoRA, no bitsandbytes/trl dependency). Loss is
computed only on the assistant span; system/user tokens are masked so the model
is trained to produce the Evidence/Steps/Answer target, not to echo the prompt.

Cluster usage (pin one low-memory GPU first):

    nvidia-smi
    export CUDA_VISIBLE_DEVICES=2
    source /NAS/yesh/VeriLong-RL/activate.sh
    python experiments/sft/train_sft_lora.py --config experiments/sft/configs/sft_v1.yaml

Use --smoke to validate config parsing and data shaping without loading the
model or touching a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("sft config must be a mapping")
    return config


def load_examples(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def build_tokenized_dataset(examples, tokenizer, max_seq_len):
    """Tokenize chat examples, masking everything before the assistant span.

    Returns (features, dropped) where features is a list of dicts with
    input_ids/attention_mask/labels and dropped counts examples whose full
    sequence exceeded max_seq_len (skipped rather than truncating the target).
    """

    features: list[dict[str, list[int]]] = []
    dropped = 0
    for example in examples:
        messages = example["messages"]
        prompt_messages = messages[:-1]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        # Assistant-only loss requires prompt_ids to be a token-level prefix of
        # full_ids. This holds for the Qwen2.5 chat template (add_generation_prompt
        # ends exactly where the assistant content begins), but assert it so a
        # future template/tokenizer change fails loudly instead of silently
        # masking the wrong span.
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"prompt is not a token prefix of full sequence for task "
                f"{example.get('metadata', {}).get('task_id', '?')}; "
                "assistant-loss masking would be incorrect"
            )
        if len(full_ids) > max_seq_len:
            dropped += 1
            continue

        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        features.append(
            {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }
        )
    return features, dropped


def run(config_path: Path, smoke: bool) -> dict[str, Any]:
    config = load_config(config_path)
    examples = load_examples(Path(config["train_path"]))

    if smoke:
        # No GPU / no model load: only validate config + data are well-formed.
        for example in examples[:5]:
            roles = [m["role"] for m in example["messages"]]
            assert roles == ["system", "user", "assistant"], roles
        return {"smoke": True, "examples": len(examples), "config": str(config_path)}

    import torch  # noqa: PLC0415
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    features, dropped = build_tokenized_dataset(examples, tokenizer, int(config["max_seq_len"]))
    if not features:
        raise SystemExit("no training features after length filtering; raise max_seq_len")

    def collate(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        pad_id = tokenizer.pad_token_id
        input_ids, attention, labels = [], [], []
        for item in batch:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad)
            attention.append(item["attention_mask"] + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if config.get("gradient_checkpointing"):
        # With a frozen base (LoRA), checkpointing only saves memory if the
        # input embeddings produce grad-requiring activations and reentrant
        # autograd is disabled; otherwise backward still materializes full
        # activations and OOMs on long sequences.
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    lora = config["lora"]
    peft_config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=float(config["num_train_epochs"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        warmup_ratio=float(config.get("warmup_ratio", 0.0)),
        logging_steps=int(config["logging_steps"]),
        save_strategy=config["save_strategy"],
        bf16=True,
        optim=config.get("optim", "adamw_torch"),
        seed=int(config["seed"]),
        report_to=[],
    )

    trainer = Trainer(model=model, args=args, train_dataset=features, data_collator=collate)
    train_result = trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    metrics = train_result.metrics
    log_path = Path(config["output_dir"]) / "train_log.json"
    log_path.write_text(
        json.dumps(
            {"metrics": metrics, "dropped_over_length": dropped, "trained_examples": len(features)},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "trained_examples": len(features),
        "dropped_over_length": dropped,
        "train_loss": metrics.get("train_loss"),
        "output_dir": config["output_dir"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT for VeriLong-RL.")
    parser.add_argument("--config", default="experiments/sft/configs/sft_v1.yaml")
    parser.add_argument("--smoke", action="store_true", help="Validate config/data without GPU or model load.")
    args = parser.parse_args()

    summary = run(Path(args.config), smoke=args.smoke)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
