# SFT Warmup (Candidate B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SFT-warmup Qwen2.5-7B-Instruct on VeriLong-RL pilot train data so it follows the Evidence/Steps/Answer format with disciplined citations, and produce an honest base-vs-SFT comparison on the dev split.

**Architecture:** A local, testable data-conversion step turns the pilot train split into chat-format SFT examples whose assistant target is assembled from programmatic gold (Evidence IDs + Answer always gold; Steps from `expected_steps`). A cluster LoRA training step (transformers Trainer + peft, bf16, no new packages) produces an adapter. The existing HF eval path gains `--adapter` so base and SFT are scored by the identical parser/metrics pipeline. Step A (this plan) is pure-gold; Step B (API-distilled Steps) is a later increment.

**Tech Stack:** Python 3.11+, transformers 4.46.3, peft 0.18.1, accelerate, torch 2.5.1+cu121, pytest. Cluster: Tang-2/Tang-3 A40 48GB, shared conda `gmsra`, NAS `/NAS/yesh`.

---

## 0. Scope and constraints

- Phase 1 only. No Core/Full scale-up, no Phase 2/3.
- Do NOT install packages on the cluster (no trl, no bitsandbytes). Use transformers `Trainer` + peft LoRA in bf16.
- Do NOT modify the shared `gmsra` env.
- Assistant SFT target: Evidence IDs and Answer come ONLY from programmatic gold; Steps come from `expected_steps`. No API distillation in this plan (that is Step B, a separate later plan).
- Reward/metric/parser code is NOT changed.
- GPU is shared: `nvidia-smi` first, pick the lowest-memory card, pin `CUDA_VISIBLE_DEVICES`, `pkill` own process when done.
- Never fabricate metrics. The comparison table comes only from real runs.
- Commits require explicit user approval. Each task lists an authorized commit command; do not run it unless the user approves committing.
- Code sync to cluster: `git archive HEAD | scp` to `/tmp`, extract to `/NAS/yesh/VeriLong-RL` (no rsync on Windows client). Single changed files may be scp'd directly.

---

## 1. File structure map

### New files
- `experiments/sft/__init__.py` — package marker.
- `experiments/sft/build_sft_data.py` — convert pilot train split → SFT chat JSONL. Pure local, unit-tested.
- `experiments/sft/train_sft_lora.py` — LoRA bf16 training with transformers Trainer + peft. Runs on cluster; supports `--smoke` for a no-GPU import/shape check.
- `experiments/sft/configs/sft_v1.yaml` — hyperparameters (lora r/alpha/dropout, lr, epochs, max_seq_len, seed, target paths).
- `experiments/sft/README.md` — how to build data, train on cluster, and eval base-vs-SFT.
- `tests/benchmark/test_sft_data.py` — assert SFT examples round-trip: assistant text parses, format_valid, citation_f1=1.0, zero distractor/stale.

### Modified files
- `experiments/eval_open_source/run_hf_eval.py` — add optional `--adapter` to load a peft adapter on top of the base model.

---

## 2. Implementation tasks

### Task 0: Prepare SFT feature branch

**Files:** none changed.

- [ ] **Step 1: Confirm current branch and clean state**

Run:

```bash
git status --short
git branch --show-current
```

Expected: current branch is `feature/hard-difficulty-system` with the hard-difficulty work committed; only untracked non-project dirs (`.claude/`, `docs/`, etc.) may show.

- [ ] **Step 2: Create the SFT branch off current HEAD**

The SFT data build reads `data/pilot/tasks.jsonl`, which is committed on the current branch, so branch off it.

Run:

```bash
git switch -c feature/sft-warmup
```

Expected: `Switched to a new branch 'feature/sft-warmup'`.

---

### Task 1: SFT data conversion (local, TDD)

**Files:**
- Create: `experiments/sft/__init__.py`
- Create: `experiments/sft/build_sft_data.py`
- Create: `tests/benchmark/test_sft_data.py`

- [ ] **Step 1: Write failing SFT data tests**

Create `tests/benchmark/test_sft_data.py`:

```python
from benchmark.generator.retrieval import generate_retrieval_task
from benchmark.generator.multihop import generate_multihop_task
from benchmark.generator.temporal import generate_temporal_task
from benchmark.parser.output_parser import parse_model_output
from benchmark.metrics.citation import citation_scores
from benchmark.metrics.format import format_scores
from experiments.sft.build_sft_data import build_sft_example


def _assistant_text(example):
    assert example["messages"][0]["role"] == "system"
    assert example["messages"][1]["role"] == "user"
    assert example["messages"][2]["role"] == "assistant"
    return example["messages"][2]["content"]


def _assert_self_consistent(task):
    example = build_sft_example(task)
    text = _assistant_text(example)
    parsed = parse_model_output(text, valid_evidence_ids=task.evidence_ids())
    fmt = format_scores(parsed, task)
    cit = citation_scores(
        pred_evidence_ids=parsed.pred_evidence_ids,
        gold_evidence_ids=task.gold_evidence_ids,
        distractor_evidence_ids=task.distractor_evidence_ids,
        stale_evidence_ids=task.stale_evidence_ids,
        valid_evidence_ids=task.evidence_ids(),
    )
    assert fmt.format_valid == 1.0
    assert cit.f1 == 1.0
    assert cit.distractor_citation_rate == 0.0
    assert cit.stale_citation_rate == 0.0
    assert parsed.pred_answer == task.gold_answer
    assert example["metadata"]["task_id"] == task.id
    assert example["metadata"]["task_family"] == task.task_family


def test_retrieval_sft_example_is_self_consistent():
    _assert_self_consistent(
        generate_retrieval_task(task_id="vlr_pilot_000001", seed=1, target_context_tokens=8000)
    )


def test_multihop_sft_example_is_self_consistent():
    _assert_self_consistent(
        generate_multihop_task(task_id="vlr_pilot_000002", seed=2, hop_count=3, target_context_tokens=8000)
    )


def test_temporal_sft_example_is_self_consistent():
    _assert_self_consistent(
        generate_temporal_task(task_id="vlr_pilot_000003", seed=3, update_count=2, target_context_tokens=8000)
    )


def test_hard_examples_are_self_consistent():
    _assert_self_consistent(
        generate_retrieval_task(
            task_id="vlr_pilot_000010", seed=10, target_context_tokens=16000,
            difficulty="hard", distractor_strength="adversarial", distractor_count=12,
            evidence_position="random",
        )
    )
    _assert_self_consistent(
        generate_temporal_task(
            task_id="vlr_pilot_000040", seed=40, update_count=4, stale_count=8,
            target_context_tokens=16000, difficulty="hard", evidence_position="mixed",
        )
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
PYTHONPATH=. python -m pytest tests/benchmark/test_sft_data.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.sft.build_sft_data'`.

- [ ] **Step 3: Create the package marker**

Create `experiments/sft/__init__.py` (empty file).

- [ ] **Step 4: Implement `build_sft_data.py`**

Create `experiments/sft/build_sft_data.py`:

```python
"""Convert the VeriLong-RL pilot train split into SFT chat examples.

The assistant target is assembled from programmatic gold so it is, by
construction, a perfect reference under the existing parser/metrics:

- Evidence: the task's gold_evidence_ids (never distractor/stale).
- Steps: the task's expected_steps, numbered.
- Answer: the task's gold_answer.

This pure-gold target teaches the format and citation discipline without
risking the over-citation habit a model-distilled target could introduce.
The system/user messages reuse the exact eval prompt so training matches
evaluation. API-distilled Steps are a later increment (Step B), not here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.schemas.task import VeriLongTask
from experiments.eval_api.run_api_eval import SYSTEM_PROMPT, build_user_prompt


def build_assistant_text(task: VeriLongTask) -> str:
    evidence_line = "Evidence: " + ", ".join(task.gold_evidence_ids)
    step_lines = [f"{i}. {step}" for i, step in enumerate(task.expected_steps, start=1)]
    return "\n".join([evidence_line, "Steps:", *step_lines, f"Answer: {task.gold_answer}"])


def build_sft_example(task: VeriLongTask) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(task)},
            {"role": "assistant", "content": build_assistant_text(task)},
        ],
        "metadata": {
            "task_id": task.id,
            "task_family": task.task_family,
            "difficulty": task.difficulty,
            "split": task.metadata.split,
            "target_context_tokens": task.metadata.target_context_tokens,
        },
    }


def build_sft_dataset(tasks_path: Path, out_path: Path, split: str | None) -> dict[str, Any]:
    written = 0
    by_family: dict[str, int] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tasks_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            task = VeriLongTask.model_validate(json.loads(line))
            if split is not None and task.metadata.split != split:
                continue
            example = build_sft_example(task)
            dst.write(json.dumps(example, ensure_ascii=False) + "\n")
            written += 1
            by_family[task.task_family] = by_family.get(task.task_family, 0) + 1
    return {"written": written, "by_family": by_family, "out": str(out_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VeriLong-RL SFT chat data from pilot tasks.")
    parser.add_argument("--tasks", default="data/pilot/tasks.jsonl")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="data/sft/train.jsonl")
    args = parser.parse_args()

    summary = build_sft_dataset(Path(args.tasks), Path(args.out), split=args.split)
    print(f"written={summary['written']} by_family={summary['by_family']} out={summary['out']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run:

```bash
PYTHONPATH=. python -m pytest tests/benchmark/test_sft_data.py -q
```

Expected: PASS (4 tests).

- [ ] **Step 6: Build the real SFT train dataset and sanity-check counts**

Run:

```bash
PYTHONPATH=. python experiments/sft/build_sft_data.py --tasks data/pilot/tasks.jsonl --split train --out data/sft/train.jsonl
```

Expected: `written=840 by_family={'anti_distractor_retrieval': 280, 'multi_hop_reasoning': 280, 'temporal_update': 280} out=data/sft/train.jsonl`.

- [ ] **Step 7: Authorized commit command**

Only after the user authorizes commits:

```bash
git add experiments/sft/__init__.py experiments/sft/build_sft_data.py tests/benchmark/test_sft_data.py
git commit -m "feat: build SFT chat data from pilot gold"
```

---

### Task 2: LoRA training script + config (cluster runner, smoke-testable)

**Files:**
- Create: `experiments/sft/configs/sft_v1.yaml`
- Create: `experiments/sft/train_sft_lora.py`

- [ ] **Step 1: Create the training config**

Create `experiments/sft/configs/sft_v1.yaml`:

```yaml
base_model: Qwen/Qwen2.5-7B-Instruct
train_path: data/sft/train.jsonl
output_dir: /NAS/yesh/VeriLong-RL/checkpoints/qwen2_5_7b_sft_v1
seed: 20260627
max_seq_len: 8192
num_train_epochs: 3
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 0.0002
warmup_ratio: 0.03
logging_steps: 5
save_strategy: epoch
gradient_checkpointing: true
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

- [ ] **Step 2: Implement `train_sft_lora.py`**

Create `experiments/sft/train_sft_lora.py`:

```python
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
        model.gradient_checkpointing_enable()
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
        warmup_ratio=float(config["warmup_ratio"]),
        logging_steps=int(config["logging_steps"]),
        save_strategy=config["save_strategy"],
        bf16=True,
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
```

- [ ] **Step 3: Smoke-test the training script locally (no GPU)**

Build a tiny dataset first so the smoke check has input, then run smoke mode:

```bash
PYTHONPATH=. python experiments/sft/build_sft_data.py --tasks data/pilot/tasks.jsonl --split train --out data/sft/train.jsonl
PYTHONPATH=. python experiments/sft/train_sft_lora.py --config experiments/sft/configs/sft_v1.yaml --smoke
```

Expected: JSON like `{"smoke": true, "examples": 840, "config": "experiments/sft/configs/sft_v1.yaml"}` and exit 0. No model download, no GPU use.

- [ ] **Step 4: Authorized commit command**

Only after the user authorizes commits:

```bash
git add experiments/sft/train_sft_lora.py experiments/sft/configs/sft_v1.yaml
git commit -m "feat: add Qwen LoRA SFT training script and config"
```

---

### Task 3: Add adapter loading to HF eval

**Files:**
- Modify: `experiments/eval_open_source/run_hf_eval.py`

- [ ] **Step 1: Add `--adapter` argument and pass it through**

In `experiments/eval_open_source/run_hf_eval.py`, change the `run(...)` signature and `_load_generator(...)` to accept an optional adapter path.

Replace the `run(` signature line:

```python
def run(
    tasks_path: Path,
    out_path: Path,
    model_path: str,
    split: str | None,
    limit: int | None,
    stratify: bool,
    max_new_tokens: int,
    fake: bool,
    adapter_path: str | None = None,
) -> dict[str, Any]:
```

Replace the generator-construction line inside `run`:

```python
    model_name = Path(model_path).name if not fake else "fake-open-source"
    if not fake and adapter_path:
        model_name = f"{model_name}+{Path(adapter_path).name}"
    generate = None if fake else _load_generator(model_path, max_new_tokens, adapter_path)
```

- [ ] **Step 2: Load the adapter in `_load_generator`**

Replace the `_load_generator` signature and the model-load block:

```python
def _load_generator(model_path: str, max_new_tokens: int, adapter_path: str | None = None):
    """Load the HF model once and return a closure that generates text."""

    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if adapter_path:
        from peft import PeftModel  # noqa: PLC0415

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
```

(The rest of `_load_generator` — the `generate` closure — is unchanged.)

- [ ] **Step 3: Add the CLI flag and thread it into `run`**

In `main()`, add the argument after `--model`:

```python
    parser.add_argument("--adapter", default=None, help="Optional peft LoRA adapter path to load on top of --model.")
```

And pass it in the `run(...)` call:

```python
    summary = run(
        tasks_path=Path(args.tasks),
        out_path=Path(args.out),
        model_path=args.model,
        split=args.split,
        limit=args.limit,
        stratify=args.stratify,
        max_new_tokens=args.max_new_tokens,
        fake=args.fake,
        adapter_path=args.adapter,
    )
```

- [ ] **Step 4: Verify wiring with the offline fake path (no GPU)**

Run:

```bash
PYTHONPATH=. python experiments/eval_open_source/run_hf_eval.py --tasks data/pilot/tasks.jsonl --split dev --limit 6 --stratify --fake --out results/raw/open_source/_wire_fake.jsonl
```

Expected: prints `model=fake-open-source tasks=6 ...` and writes the file. Then remove it:

```bash
rm -f results/raw/open_source/_wire_fake.jsonl
```

- [ ] **Step 5: Authorized commit command**

Only after the user authorizes commits:

```bash
git add experiments/eval_open_source/run_hf_eval.py
git commit -m "feat: support peft adapter in HF eval"
```

---

### Task 4: Cluster training and base-vs-SFT evaluation

**Files:** none in-repo; produces artifacts on cluster + `results/raw/open_source/` outputs.

This task runs real GPU work. Do it only after Tasks 1-3 pass locally.

- [ ] **Step 1: Pick a free GPU**

Run:

```bash
ssh Tang-2-Wu 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'
ssh Tang-3-Wu 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'
```

Pick the lowest-memory-used card on whichever host is freer; note `HOST` and `GPU`.

- [ ] **Step 2: Sync code and SFT data to the cluster**

From the repo root (Windows client), package the committed tree and the built SFT data and copy it over:

```bash
git archive --format=tar HEAD -o /tmp/verilong_sft.tar
scp /tmp/verilong_sft.tar <HOST>:/tmp/verilong_sft.tar
scp data/sft/train.jsonl <HOST>:/tmp/sft_train.jsonl
ssh <HOST> 'cd /NAS/yesh/VeriLong-RL && tar -xf /tmp/verilong_sft.tar && mkdir -p data/sft && cp /tmp/sft_train.jsonl data/sft/train.jsonl'
```

(`data/sft/train.jsonl` is regenerated locally in Task 1 Step 6; if it is gitignored it will not be in the archive, hence the explicit scp.)

- [ ] **Step 3: Run LoRA training on the pinned GPU**

```bash
ssh <HOST> 'cd /NAS/yesh/VeriLong-RL && source activate.sh && export CUDA_VISIBLE_DEVICES=<GPU> && nohup python experiments/sft/train_sft_lora.py --config experiments/sft/configs/sft_v1.yaml > /NAS/yesh/VeriLong-RL/sft_v1_train.log 2>&1 &'
```

Monitor:

```bash
ssh <HOST> 'tail -n 30 /NAS/yesh/VeriLong-RL/sft_v1_train.log'
```

Expected at completion: `train_log.json` written under the output dir, with a finite `train_loss` and a `dropped_over_length` count (16K-context tasks dropped at max_seq_len 8192 are expected and acceptable for v1).

- [ ] **Step 4: Evaluate base model on dev (stratified)**

```bash
ssh <HOST> 'cd /NAS/yesh/VeriLong-RL && source activate.sh && export CUDA_VISIBLE_DEVICES=<GPU> && python experiments/eval_open_source/run_hf_eval.py --tasks data/pilot/tasks.jsonl --split dev --limit 30 --stratify --model Qwen/Qwen2.5-7B-Instruct --out results/raw/open_source/qwen_base_dev30.jsonl'
```

- [ ] **Step 5: Evaluate SFT adapter on the same dev subset**

```bash
ssh <HOST> 'cd /NAS/yesh/VeriLong-RL && source activate.sh && export CUDA_VISIBLE_DEVICES=<GPU> && python experiments/eval_open_source/run_hf_eval.py --tasks data/pilot/tasks.jsonl --split dev --limit 30 --stratify --model Qwen/Qwen2.5-7B-Instruct --adapter /NAS/yesh/VeriLong-RL/checkpoints/qwen2_5_7b_sft_v1 --out results/raw/open_source/qwen_sft_dev30.jsonl'
```

- [ ] **Step 6: Score both on the cluster and copy results back**

```bash
ssh <HOST> 'cd /NAS/yesh/VeriLong-RL && source activate.sh && python scripts/score_outputs.py --tasks data/pilot/tasks.jsonl --outputs results/raw/open_source/qwen_base_dev30.jsonl --scored results/raw/open_source/qwen_base_dev30_scored.jsonl --summary results/raw/open_source/qwen_base_dev30_summary.json && python scripts/score_outputs.py --tasks data/pilot/tasks.jsonl --outputs results/raw/open_source/qwen_sft_dev30.jsonl --scored results/raw/open_source/qwen_sft_dev30_scored.jsonl --summary results/raw/open_source/qwen_sft_dev30_summary.json'
scp '<HOST>:/NAS/yesh/VeriLong-RL/results/raw/open_source/qwen_*_dev30_*.json*' results/raw/open_source/
```

- [ ] **Step 7: Free the GPU**

```bash
ssh <HOST> 'pkill -f train_sft_lora.py; pkill -f run_hf_eval.py; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
```

(The training process should already have exited; this guards against leftovers.)

---

### Task 5: Base-vs-SFT comparison and memory

**Files:** none in-repo (analysis + memory).

- [ ] **Step 1: Compute the per-family comparison from the real scored files**

Run locally after results are copied back:

```bash
PYTHONPATH=. python - <<'PY'
import json, collections
from pathlib import Path
def load(p): return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
def mean(xs): return sum(xs)/len(xs) if xs else 0.0
runs = {
  "base": "results/raw/open_source/qwen_base_dev30_scored.jsonl",
  "sft":  "results/raw/open_source/qwen_sft_dev30_scored.jsonl",
}
for name, p in runs.items():
    rows = load(p)
    print(f"\n== {name} (n={len(rows)}) ==")
    for f in ["answer_normalized_match","format_valid","citation_precision","citation_recall","distractor_citation_rate","stale_citation_rate","reward_total"]:
        print(f"  {f}: {mean([r[f] for r in rows]):.3f}")
    byfam = collections.defaultdict(list)
    for r in rows: byfam[r["task_family"]].append(r)
    for fam in sorted(byfam):
        fr = byfam[fam]
        print(f"  [{fam}] reward={mean([r['reward_total'] for r in fr]):.3f} cit_prec={mean([r['citation_precision'] for r in fr]):.3f} distr={mean([r['distractor_citation_rate'] for r in fr]):.3f} stale={mean([r['stale_citation_rate'] for r in fr]):.3f}")
PY
```

- [ ] **Step 2: Report the honest result**

State base vs SFT deltas, focusing on the hypotheses: SFT should improve format validity and citation discipline (precision up, distractor/over-citation down), and possibly temporal stale-rate. If SFT does NOT improve or regresses, report that plainly with the numbers and a brief cause analysis (e.g. dropped 16K examples, target too uniform). Do not fabricate improvement.

- [ ] **Step 3: Update memory**

Create `sft-warmup-plan` memory (auto-memory path) recording: data = pure-gold pilot train 840 (8K trained, 16K dropped at max_seq_len 8192, with count), LoRA bf16 hyperparameters, the base-vs-SFT result table, and what Step B (API-distilled Steps) would add. Add the index line to `MEMORY.md`. Cross-link `[[hard-difficulty-system]]` and `[[pilot-eval-findings-gemini]]`.

---

## 3. Verification commands (local, Tasks 1-3)

```bash
PYTHONPATH=. python -m pytest tests/benchmark/test_sft_data.py -q
PYTHONPATH=. python experiments/sft/build_sft_data.py --tasks data/pilot/tasks.jsonl --split train --out data/sft/train.jsonl
PYTHONPATH=. python experiments/sft/train_sft_lora.py --config experiments/sft/configs/sft_v1.yaml --smoke
PYTHONPATH=. python -m pytest -q
```

Expected: SFT data tests pass; `written=840`; smoke prints `examples=840`; full suite stays green.

---

## 4. Self-review

### Spec coverage
- SFT data conversion (pure gold, eval-aligned prompt): Task 1.
- LoRA bf16 training without trl/bitsandbytes, assistant-only loss: Task 2.
- Adapter eval through the existing scorer: Task 3.
- Real cluster training + base-vs-SFT eval: Task 4.
- Honest comparison + memory: Task 5.
- Step B (API-distilled Steps) is explicitly deferred to a later plan.

### Placeholder scan
No placeholders. Cluster `<HOST>`/`<GPU>` are runtime selections made in Task 4 Step 1 from real `nvidia-smi` output, not fabricated values.

### Type/consistency
- `build_sft_example` / `build_assistant_text` / `build_sft_dataset` names are consistent across Task 1 code and tests.
- SFT example shape `{messages:[system,user,assistant], metadata}` matches the plan's Task 8 Step 2 format in the Phase 1 plan.
- `run_hf_eval.run(...)` gains `adapter_path` used consistently in `_load_generator`.
- Assistant target uses `gold_evidence_ids` + `expected_steps` + `gold_answer`, which the existing parser/format/citation metrics score as perfect by construction (verified against `step_bounds`: retrieval 1∈[1,2], multihop/temporal within bounds).
