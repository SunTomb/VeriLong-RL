"""GRPO training for VeriLong-RL: RL on top of v1-gold SFT with programmatic reward.

Continues training the v1 LoRA adapter (base frozen) using trl's GRPOTrainer.
Rollouts are scored by the same programmatic reward as the benchmark scorer.
Run on the cluster `verilong_rl` env (trl 0.14.0 + vllm 0.6.6.post1).
Unlike the SFT script's local --smoke dry-run, GRPO --smoke still loads the model and trains 2 real steps, so it requires the cluster GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.rlvr.data import load_grpo_dataset
from experiments.rlvr.reward import make_reward_fn


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"config at {path} must be a YAML mapping, got {type(config).__name__}")
    return config


def run(config_path: Path, smoke: bool) -> dict[str, Any]:
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from trl import GRPOConfig, GRPOTrainer  # noqa: PLC0415

    config = load_config(config_path)

    num_generations = 2 if smoke else int(config["num_generations"])
    max_steps = 2 if smoke else int(config["max_steps"])
    use_vllm = False if smoke else bool(config.get("use_vllm", False))
    max_prompt_length = 2048 if smoke else int(config["max_prompt_length"])
    # In smoke mode force a tiny batch (= G) so one prompt's group fits a step.
    # Otherwise honor the config, but GRPO requires the per-device batch to be a
    # whole number of groups, so it must be divisible by num_generations.
    if smoke:
        per_device_bs = num_generations
    else:
        per_device_bs = int(config["per_device_train_batch_size"])
        if per_device_bs % num_generations != 0:
            raise SystemExit(
                f"per_device_train_batch_size ({per_device_bs}) must be divisible by "
                f"num_generations ({num_generations})"
            )

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    # With vllm, rollout runs on cuda:1, so the policy model must stay on cuda:0
    # only (device_map="auto" would otherwise shard across both and collide with
    # vllm). Without vllm (smoke), "auto" is fine on the single visible card.
    device_map = {"": 0} if use_vllm else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"], torch_dtype=torch.bfloat16, device_map=device_map,
        # SDPA fused attention avoids materializing the O(batch*seq^2) 4D causal
        # mask that eager attention clones. With GRPO's batch == num_generations
        # of 14K-token sequences, that mask alone is ~23GB and OOMs; SDPA removes
        # it without touching sequence length (so gold evidence stays intact).
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, config["init_adapter"], is_trainable=True)
    model.config.use_cache = False
    # GRPO holds policy + generation state, so it is even more memory-hungry than
    # SFT. Gradient checkpointing only saves memory with a frozen base when the
    # input embeddings produce grad-requiring activations and reentrant autograd
    # is disabled (same fix as train_sft_lora.py); without it the 7B OOMs even at
    # G=2 on an 80GB card.
    model.enable_input_require_grads()

    dataset = load_grpo_dataset(
        Path(config["tasks_path"]), config["family"],
        int(config["max_context_tokens"]), config["split"],
    )
    if smoke:
        dataset = dataset.select(range(min(4, len(dataset))))

    grpo_config = GRPOConfig(
        output_dir=config["output_dir"],
        num_generations=num_generations,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        beta=float(config["beta"]),
        max_prompt_length=max_prompt_length,
        max_completion_length=int(config["max_completion_length"]),
        temperature=float(config["temperature"]),
        max_steps=max_steps,
        logging_steps=int(config["logging_steps"]),
        save_steps=int(config["save_steps"]),
        bf16=bool(config.get("bf16", True)),
        seed=int(config["seed"]),
        use_vllm=use_vllm,
        vllm_device=config.get("vllm_device", "auto") if use_vllm else "auto",
        vllm_gpu_memory_utilization=float(config.get("vllm_gpu_memory_utilization", 0.45)),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=make_reward_fn(),
        args=grpo_config,
        train_dataset=dataset,
    )
    train_result = trainer.train()
    trainer.save_model(config["output_dir"])

    summary = {
        "train_runtime": train_result.metrics.get("train_runtime"),
        "metrics": train_result.metrics,  # full trl metrics dict (reward key name varies by version)
        "steps": max_steps,
        "num_generations": num_generations,
        "smoke": smoke,
        "dataset_size": len(dataset),
        "output_dir": config["output_dir"],
    }
    log_path = Path(config["output_dir"]) / "grpo_train_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training for VeriLong-RL RLVR.")
    parser.add_argument("--config", default="experiments/rlvr/configs/grpo_v1.yaml")
    parser.add_argument("--smoke", action="store_true", help="Tiny cluster smoke run (G=2, 2 steps, no vllm, 4 samples) to validate the GRPO pipeline on GPU before a full run.")
    args = parser.parse_args()
    summary = run(Path(args.config), smoke=args.smoke)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
