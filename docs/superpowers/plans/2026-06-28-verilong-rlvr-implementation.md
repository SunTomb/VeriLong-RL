# VeriLong-RL RLVR (GRPO) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 v1-gold SFT checkpoint 之上叠加一轮 GRPO（trl 0.14.0 + vllm），用复用现有评测打分链路的程序化 reward 强化 retrieval family 的引用纪律，验证 RQ3（RLVR 改善 citation precision / overcitation）。

**Architecture:** 新增 `experiments/rlvr/` 包（reward / data / train_grpo / config），与 `experiments/sft/` 平级。reward 回调复用从 `scripts/score_outputs.py` 抽出的 `score_output_record`（保证 RL 优化目标 = 评测口径）。GRPOTrainer 继续训练 v1 的 LoRA adapter（base 冻结），vllm 做 rollout，导出 v1+RL adapter，在 hard retrieval 子集上与 v1-gold 同口径对比。

**Tech Stack:** Python 3.10, trl 0.14.0 (GRPOTrainer/GRPOConfig), vllm 0.6.6.post1, peft 0.18.1, transformers 4.46.3, torch 2.5.1+cu121 — 全部在集群 `verilong_rl` conda env（非 gmsra）。本地仅跑无 GPU 单测（pytest）。

**关键约束（来自 CLAUDE.md / 设计 spec）：**
- 只做 Phase 1 retrieval 8K，不碰 multihop/temporal/32K/Phase 2-3。
- reward 纯程序化，禁止在线 LLM judge。
- 集群 GPU 共享：`nvidia-smi` 先看，`CUDA_VISIBLE_DEVICES` 锁卡，用完 `pkill`。
- 不污染 gmsra；RLVR 用 `verilong_rl` env。
- 不提交到 master，在 feature 分支；commit 需用户批准（plan 内 commit 步骤指本地提交到 feature 分支）。

**起点状态：** master 已含 SFT/hard 全部代码（67 tests 绿）。v1-gold adapter 在 `/NAS/yesh/VeriLong-RL/checkpoints/qwen2_5_7b_sft_v1`。hard retrieval 评测子集需从 `data/pilot/hard_eval_sft_16k.jsonl` 过滤出 retrieval。

---

## File Structure

| 文件 | 创建/修改 | 职责 |
|---|---|---|
| `benchmark/reward/score.py` | 创建 | 从 `scripts/score_outputs.py` 抽出 `score_output_record` + `_error_type`，成为可被复用的打分核心。 |
| `scripts/score_outputs.py` | 修改 | 改为从 `benchmark.reward.score` import，删除重复定义，CLI 行为不变。 |
| `experiments/rlvr/__init__.py` | 创建 | 空包标记。 |
| `experiments/rlvr/reward.py` | 创建 | `make_reward_fn()` → trl reward 回调，复用 `score_output_record`。 |
| `experiments/rlvr/data.py` | 创建 | `load_grpo_prompts(...)` → HF Dataset（prompt + task_json 列）。 |
| `experiments/rlvr/train_grpo.py` | 创建 | GRPO 训练编排 + `--smoke`。 |
| `experiments/rlvr/configs/grpo_v1.yaml` | 创建 | GRPO 超参。 |
| `tests/benchmark/test_rlvr_reward.py` | 创建 | reward 回调单测 + 与 score_output_record 同尺一致性。 |
| `tests/benchmark/test_rlvr_data.py` | 创建 | data.py prompt/列构造单测。 |

## Task 1: 抽出 `score_output_record` 到 `benchmark/reward/score.py`

**目的：** 让 RLVR reward 能复用评测打分核心，又不依赖 `scripts/`（脚本目录不是包，import 不干净）。保持 `score_outputs.py` CLI 与现有测试行为完全不变。

**Files:**
- Create: `benchmark/reward/score.py`
- Modify: `scripts/score_outputs.py`
- Test: `tests/benchmark/test_reward.py`（已存在；新增一个回归用例）

- [ ] **Step 1: 写回归测试，锁定抽取前后 score_output_record 行为一致**

在 `tests/benchmark/test_reward.py` 末尾追加：

```python
def test_score_output_record_importable_from_benchmark_reward():
    # 抽取后必须能从新位置 import，且对一个 gold-perfect 输出打满 reward。
    from benchmark.reward.score import score_output_record
    from benchmark.generator.retrieval import generate_retrieval_task

    task = generate_retrieval_task(task_id="vlr_pilot_000001", seed=1, target_context_tokens=8000)
    gold_ids = ", ".join(task.gold_evidence_ids)
    output_text = (
        f"Evidence: {gold_ids}\nSteps:\n1. {task.gold_evidence_ids[0]} states the fact.\n"
        f"Answer: {task.gold_answer}"
    )
    scored = score_output_record(task, {"task_id": task.id, "output_text": output_text})
    assert scored["citation_precision"] == 1.0
    assert scored["answer_normalized_match"] == 1.0
    assert scored["reward_total"] > 0.8
```

- [ ] **Step 2: 运行测试，确认失败（模块尚不存在）**

Run: `python -m pytest tests/benchmark/test_reward.py::test_score_output_record_importable_from_benchmark_reward -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'benchmark.reward.score'`

- [ ] **Step 3: 创建 `benchmark/reward/score.py`，移入 score_output_record 与 _error_type**

把 `scripts/score_outputs.py` 中的 `score_output_record` 和 `_error_type` 原样移入新文件，连同它们需要的 import：

```python
from typing import Any

from benchmark.metrics.answer import exact_match, normalized_match
from benchmark.metrics.citation import citation_scores
from benchmark.metrics.format import format_scores
from benchmark.parser.output_parser import parse_model_output
from benchmark.reward.programmatic import compute_reward
from benchmark.schemas.task import VeriLongTask


def score_output_record(task: VeriLongTask, output_record: dict[str, Any]) -> dict[str, Any]:
    output_text = str(output_record.get("output_text", output_record.get("output", "")))
    parsed = parse_model_output(output_text, valid_evidence_ids=task.evidence_ids())
    citation = citation_scores(
        pred_evidence_ids=parsed.pred_evidence_ids,
        gold_evidence_ids=task.gold_evidence_ids,
        distractor_evidence_ids=task.distractor_evidence_ids,
        stale_evidence_ids=task.stale_evidence_ids,
        valid_evidence_ids=task.evidence_ids(),
    )
    format_breakdown = format_scores(parsed, task)
    answer_score = normalized_match(parsed.pred_answer, task.gold_answer)
    reward = compute_reward(
        answer_score=answer_score,
        citation_f1=citation.f1,
        reasoning_score=format_breakdown.step_count_valid,
        format_score=format_breakdown.format_valid,
        distractor_rate=citation.distractor_citation_rate,
        stale_rate=citation.stale_citation_rate,
        invalid_rate=citation.invalid_citation_rate,
    )
    return {
        "task_id": task.id,
        "model": output_record.get("model", "unknown"),
        "task_family": task.task_family,
        "difficulty": task.difficulty,
        "answer_exact_match": exact_match(parsed.pred_answer, task.gold_answer),
        "answer_normalized_match": answer_score,
        "format_valid": format_breakdown.format_valid,
        "step_count_valid": format_breakdown.step_count_valid,
        "citation_precision": citation.precision,
        "citation_recall": citation.recall,
        "citation_f1": citation.f1,
        "all_gold_evidence_recall": citation.all_gold_evidence_recall,
        "distractor_citation_rate": citation.distractor_citation_rate,
        "stale_citation_rate": citation.stale_citation_rate,
        "invalid_citation_rate": citation.invalid_citation_rate,
        "overcitation_rate": citation.overcitation_rate,
        "reward_total": reward.total,
        "reward_components": reward.components,
        "parsed": parsed.model_dump(mode="json"),
        "error_type": _error_type(
            parsed.error_flags,
            format_breakdown.step_count_valid,
            citation.distractor_citation_rate,
            citation.stale_citation_rate,
        ),
    }


def _error_type(
    error_flags: list[str],
    step_count_valid: float,
    distractor_rate: float,
    stale_rate: float,
) -> str | None:
    if error_flags:
        return "format_or_invalid_citation"
    if step_count_valid < 1.0:
        return "invalid_step_count"
    if distractor_rate > 0:
        return "distractor_citation"
    if stale_rate > 0:
        return "stale_citation"
    return None
```

- [ ] **Step 4: 改 `scripts/score_outputs.py` 从新模块 import，删除重复定义**

删除 `score_outputs.py` 中的 `score_output_record` 和 `_error_type` 函数体，在顶部 import 区把这两个的本地定义替换为：

```python
from benchmark.reward.score import score_output_record
```

（`_error_type` 仅被 `score_output_record` 使用，移走后 `score_outputs.py` 不再需要它。保留 `score_outputs.py` 里 `main`、`_load_tasks`、`SUMMARY_FIELDS` 不动。注意删除因移走而不再使用的 import：`exact_match`/`normalized_match`/`citation_scores`/`format_scores`/`parse_model_output`/`compute_reward` 若 `main` 不再直接用则删除，避免 lint 警告；`_load_tasks` 仍需 `VeriLongTask`。）

- [ ] **Step 5: 运行新测试 + 全量回归，确认绿**

Run: `python -m pytest tests/benchmark/test_reward.py tests/benchmark/test_pilot_eval.py -v`
Expected: 全部 PASS（含新用例；现有打分相关测试不破）。

Run: `python -m pytest -q`
Expected: 之前 67 passed → 现在 68 passed（新增 1 用例）。

- [ ] **Step 6: Commit**

```bash
git add benchmark/reward/score.py scripts/score_outputs.py tests/benchmark/test_reward.py
git commit -m "refactor: extract score_output_record to benchmark.reward.score for reuse"
```

## Task 2: RLVR reward 回调 `experiments/rlvr/reward.py`

**目的：** 把 `score_output_record` 包成 trl GRPOTrainer 期望的 `reward_funcs` 签名 `fn(prompts, completions, **kwargs) -> list[float]`。kwargs 中按 dataset 列名透传 `task_json`（每个 task 的 JSON 字符串）。

**Files:**
- Create: `experiments/rlvr/__init__.py`（空文件）
- Create: `experiments/rlvr/reward.py`
- Test: `tests/benchmark/test_rlvr_reward.py`

- [ ] **Step 1: 创建空包标记**

创建 `experiments/rlvr/__init__.py`，内容为空（0 字节）。

- [ ] **Step 2: 写失败测试**

创建 `tests/benchmark/test_rlvr_reward.py`：

```python
import json

from benchmark.generator.retrieval import generate_retrieval_task
from benchmark.reward.score import score_output_record
from experiments.rlvr.reward import make_reward_fn


def _task_and_json():
    task = generate_retrieval_task(task_id="vlr_pilot_000001", seed=1, target_context_tokens=8000)
    return task, task.model_dump_json()


def test_reward_fn_matches_score_output_record_for_gold_perfect():
    task, task_json = _task_and_json()
    gold_ids = ", ".join(task.gold_evidence_ids)
    text = (
        f"Evidence: {gold_ids}\nSteps:\n1. {task.gold_evidence_ids[0]} states the fact.\n"
        f"Answer: {task.gold_answer}"
    )
    reward_fn = make_reward_fn()
    got = reward_fn(prompts=["ignored"], completions=[text], task_json=[task_json])
    expected = score_output_record(task, {"output_text": text})["reward_total"]
    assert abs(got[0] - expected) < 1e-6


def test_reward_fn_penalizes_distractor_citation():
    task, task_json = _task_and_json()
    distractor = task.distractor_evidence_ids[0]
    good = f"Evidence: {task.gold_evidence_ids[0]}\nSteps:\n1. ok.\nAnswer: {task.gold_answer}"
    bad = f"Evidence: {task.gold_evidence_ids[0]}, {distractor}\nSteps:\n1. ok.\nAnswer: {task.gold_answer}"
    reward_fn = make_reward_fn()
    rewards = reward_fn(prompts=["x", "x"], completions=[good, bad], task_json=[task_json, task_json])
    assert rewards[0] > rewards[1]  # 引入 distractor 必须降低 reward


def test_reward_fn_accepts_chat_style_completion():
    # vllm/trl 在 conversational 模式下 completion 是 [{"role","content"}] 列表
    task, task_json = _task_and_json()
    text = f"Evidence: {task.gold_evidence_ids[0]}\nSteps:\n1. ok.\nAnswer: {task.gold_answer}"
    reward_fn = make_reward_fn()
    chat = [[{"role": "assistant", "content": text}]]
    got = reward_fn(prompts=["x"], completions=chat, task_json=[task_json])
    assert got[0] > 0.0
```

- [ ] **Step 3: 运行测试，确认失败**

Run: `python -m pytest tests/benchmark/test_rlvr_reward.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'experiments.rlvr.reward'`

- [ ] **Step 4: 实现 `experiments/rlvr/reward.py`**

```python
"""Programmatic reward callback for GRPO, reusing the eval scoring core.

The reward a completion receives during RL is *identical* to the reward_total
the benchmark scorer would assign, so the RL objective and the reported metric
are the same ruler. We do not re-implement any scoring here.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from benchmark.reward.score import score_output_record
from benchmark.schemas.task import VeriLongTask


def _completion_text(completion: Any) -> str:
    """trl passes a plain string (text mode) or a chat list (conversational)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion)


def make_reward_fn() -> Callable[..., list[float]]:
    def reward_fn(prompts=None, completions=None, task_json=None, **_kwargs) -> list[float]:
        rewards: list[float] = []
        for i, completion in enumerate(completions):
            task = VeriLongTask.model_validate(json.loads(task_json[i]))
            text = _completion_text(completion)
            scored = score_output_record(task, {"output_text": text})
            rewards.append(float(scored["reward_total"]))
        return rewards

    # trl reads __name__ for logging; give it a stable, meaningful name.
    reward_fn.__name__ = "programmatic_citation_reward"
    return reward_fn
```

- [ ] **Step 5: 运行测试，确认通过**

Run: `python -m pytest tests/benchmark/test_rlvr_reward.py -v`
Expected: 3 passed。

- [ ] **Step 6: Commit**

```bash
git add experiments/rlvr/__init__.py experiments/rlvr/reward.py tests/benchmark/test_rlvr_reward.py
git commit -m "feat: add GRPO programmatic reward callback reusing eval scorer"
```

## Task 3: GRPO prompt 数据集 `experiments/rlvr/data.py`

**目的：** 从 pilot tasks 过滤出 retrieval + 指定 context 的 split，构造 HF `Dataset`，每行含 `prompt`（chat 形式：system + user，与 SFT/eval 同一 prompt）和 `task_json`（透传给 reward）。

**Files:**
- Create: `experiments/rlvr/data.py`
- Test: `tests/benchmark/test_rlvr_data.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/benchmark/test_rlvr_data.py`：

```python
import json

from experiments.rlvr.data import build_prompt_messages, iter_grpo_rows
from benchmark.generator.retrieval import generate_retrieval_task
from experiments.eval_api.run_api_eval import SYSTEM_PROMPT


def test_build_prompt_messages_uses_system_and_user():
    task = generate_retrieval_task(task_id="vlr_pilot_000001", seed=1, target_context_tokens=8000)
    messages = build_prompt_messages(task)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1]["role"] == "user"
    assert task.question in messages[1]["content"]


def test_iter_grpo_rows_filters_family_and_context(tmp_path):
    # build a tiny tasks.jsonl with mixed family/context/split
    keep = generate_retrieval_task(task_id="vlr_pilot_000001", seed=1, target_context_tokens=8000)
    keep.metadata.split = "train"
    drop_ctx = generate_retrieval_task(task_id="vlr_pilot_000002", seed=2, target_context_tokens=16000)
    drop_ctx.metadata.split = "train"
    path = tmp_path / "tasks.jsonl"
    path.write_text(keep.model_dump_json() + "\n" + drop_ctx.model_dump_json() + "\n", encoding="utf-8")

    rows = list(iter_grpo_rows(path, family="anti_distractor_retrieval", max_context_tokens=8000, split="train"))
    assert len(rows) == 1
    row = rows[0]
    assert set(row.keys()) == {"prompt", "task_json"}
    assert isinstance(row["prompt"], list)  # chat messages
    # task_json must round-trip to the same task id
    assert json.loads(row["task_json"])["id"] == "vlr_pilot_000001"
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `python -m pytest tests/benchmark/test_rlvr_data.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'experiments.rlvr.data'`

- [ ] **Step 3: 实现 `experiments/rlvr/data.py`**

```python
"""Build the GRPO prompt dataset from VeriLong-RL pilot tasks.

Each row carries the chat-style `prompt` (system + user, the exact eval/SFT
prompt) and `task_json` (the task serialized) so the reward callback can score
each rollout against programmatic gold. Filtering keeps a single family and a
single context length to scope the first GRPO run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from benchmark.schemas.task import VeriLongTask
from experiments.eval_api.run_api_eval import SYSTEM_PROMPT, build_user_prompt


def build_prompt_messages(task: VeriLongTask) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(task)},
    ]


def iter_grpo_rows(
    tasks_path: Path,
    family: str,
    max_context_tokens: int,
    split: str,
) -> Iterator[dict[str, Any]]:
    with Path(tasks_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            task = VeriLongTask.model_validate(json.loads(line))
            if task.task_family != family:
                continue
            if task.metadata.split != split:
                continue
            if task.metadata.target_context_tokens != max_context_tokens:
                continue
            yield {"prompt": build_prompt_messages(task), "task_json": task.model_dump_json()}


def load_grpo_dataset(
    tasks_path: Path,
    family: str,
    max_context_tokens: int,
    split: str,
):
    """Return a HF Dataset. Imported lazily so unit tests need no `datasets`."""
    from datasets import Dataset  # noqa: PLC0415

    rows = list(iter_grpo_rows(tasks_path, family, max_context_tokens, split))
    if not rows:
        raise SystemExit(
            f"no tasks matched family={family!r} split={split!r} ctx={max_context_tokens}"
        )
    return Dataset.from_list(rows)
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `python -m pytest tests/benchmark/test_rlvr_data.py -v`
Expected: 2 passed。

- [ ] **Step 5: Commit**

```bash
git add experiments/rlvr/data.py tests/benchmark/test_rlvr_data.py
git commit -m "feat: add GRPO prompt dataset builder (retrieval/context filter)"
```

## Task 4: GRPO 配置文件 `experiments/rlvr/configs/grpo_v1.yaml`

**Files:**
- Create: `experiments/rlvr/configs/grpo_v1.yaml`

- [ ] **Step 1: 创建配置文件**

```yaml
base_model: Qwen/Qwen2.5-7B-Instruct
# v1-gold SFT LoRA adapter (RL 起点). GRPO 继续训练这个 adapter, base 冻结.
init_adapter: /NAS/yesh/VeriLong-RL/checkpoints/qwen2_5_7b_sft_v1
output_dir: /NAS/yesh/VeriLong-RL/checkpoints/qwen2_5_7b_sft_v1_grpo
tasks_path: data/pilot/tasks.jsonl
family: anti_distractor_retrieval
split: train
max_context_tokens: 8000
seed: 20260628

# GRPO 超参 (设计 spec §6)
num_generations: 8          # 每 prompt 采样数 G (group size)
beta: 0.04                  # KL 到 v1 参考策略, 温和防退化
learning_rate: 0.000001     # 1e-6, RL 远小于 SFT 2e-4
max_prompt_length: 14336    # retrieval 8K ≈ 14K token (SFT 同口径)
max_completion_length: 256  # Evidence/Steps/Answer 输出短
per_device_train_batch_size: 8   # 必须能被 num_generations 整除 (trl 要求): 8/8=1 prompt/step
gradient_accumulation_steps: 4
temperature: 0.7            # rollout 需多样性 (评测是贪心)
max_steps: 100             # 闭环验证, 看曲线再延
logging_steps: 1
save_steps: 25
bf16: true

# vllm rollout (use_vllm=True 时 trl 用 vllm_gpu_memory_utilization 控制显存)
use_vllm: true
vllm_gpu_memory_utilization: 0.45
```

**说明（给实现者）：** trl 0.14 `GRPOConfig` 要求 `per_device_train_batch_size` 能被 `num_generations` 整除（每个 prompt 的 G 个 generation 在同一 batch 内构成 group）。这里 8/8=1，即每 step 1 个 prompt、8 个 completion。`use_vllm=True` 时，trl 0.14 在**主进程所在 GPU 之外**起 vllm；2 卡布局通过启动时 `CUDA_VISIBLE_DEVICES` + trl 的 vllm 参数控制（见 Task 6 运行命令）。

- [ ] **Step 2: Commit**

```bash
git add experiments/rlvr/configs/grpo_v1.yaml
git commit -m "feat: add GRPO v1 config (retrieval 8K, G=8, vllm)"
```

## Task 5: GRPO 训练脚本 `experiments/rlvr/train_grpo.py`

**目的：** 编排 —— 读 config、加载 base+v1 adapter、构造 dataset、配 GRPOConfig、跑 GRPOTrainer、导出 adapter。含 `--smoke`（极小规模、关 vllm）便于先验证管线。

**Files:**
- Create: `experiments/rlvr/train_grpo.py`
- Test: 无新单测（依赖 GPU/trl，靠 Task 6 的集群 smoke 验证；本地仅 `--help` 可解析）

- [ ] **Step 1: 实现脚本骨架（config 读取 + dataset + 模型加载）**

创建 `experiments/rlvr/train_grpo.py`：

```python
"""GRPO training for VeriLong-RL: RL on top of v1-gold SFT with programmatic reward.

Continues training the v1 LoRA adapter (base frozen) using trl's GRPOTrainer.
Rollouts are scored by the same programmatic reward as the benchmark scorer.
Run on the cluster `verilong_rl` env (trl 0.14.0 + vllm 0.6.6.post1).
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
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


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
    per_device_bs = num_generations  # 1 prompt/step; must be divisible by num_generations

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"], torch_dtype=torch.bfloat16, device_map="auto"
    )
    # Load v1-gold adapter as trainable (is_trainable=True so GRPO updates it).
    model = PeftModel.from_pretrained(model, config["init_adapter"], is_trainable=True)
    model.config.use_cache = False

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
        vllm_gpu_memory_utilization=float(config.get("vllm_gpu_memory_utilization", 0.45)),
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
        "train_reward": train_result.metrics.get("train_reward", train_result.metrics.get("reward")),
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
    parser.add_argument("--smoke", action="store_true", help="Tiny run (G=2, 2 steps, no vllm) to validate the pipeline.")
    args = parser.parse_args()
    summary = run(Path(args.config), smoke=args.smoke)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 本地验证 `--help` 可解析（不触发 GPU import）**

注意：`run()` 内部才 import torch/trl，故 `--help` 在本地（无 trl）也能跑。
Run: `python experiments/rlvr/train_grpo.py --help`
Expected: 打印 usage，含 `--config` 和 `--smoke`，exit 0。

- [ ] **Step 3: 本地验证 config 能被 yaml 解析**

Run: `python -c "from experiments.rlvr.train_grpo import load_config; c=load_config('experiments/rlvr/configs/grpo_v1.yaml'); print(c['family'], c['num_generations'])"`
Expected: `anti_distractor_retrieval 8`

- [ ] **Step 4: Commit**

```bash
git add experiments/rlvr/train_grpo.py
git commit -m "feat: add GRPO training script with --smoke pipeline check"
```

## Task 6: 集群冒烟 + 正式 GRPO 训练

**目的：** 在 `verilong_rl` env 先 smoke 验证管线，再正式训练，导出 v1+RL adapter。**全部在集群执行**（本地无 GPU/trl）。

**前置：** 把最新代码同步到集群 `/NAS/yesh/VeriLong-RL`（`git archive HEAD | scp` 或单文件 scp，见 [[cluster-gpu-usage-tang-song]]）。

- [ ] **Step 1: 同步代码到集群**

本地（repo root）：
```bash
git archive --format=tar HEAD | ssh Song-3-Wu "cd /NAS/yesh/VeriLong-RL && tar -xf - "
```
（或对改动的少数文件逐个 scp。注意 activate.sh 若重新解包需 `sed -i 's/\r$//'`。）

- [ ] **Step 2: 选空卡（2 张，Song-3 优先）**

```bash
ssh Song-3-Wu 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits'
```
Expected: 选两张显存占用最低、util≈0 的卡（记为 $TRAIN_GPU、$VLLM_GPU）。

- [ ] **Step 3: 冒烟（单卡、关 vllm、G=2、2 步）**

```bash
ssh Song-3-Wu 'cd /NAS/yesh/VeriLong-RL && source /NAS/yesh/miniconda3/etc/profile.d/conda.sh && conda activate verilong_rl && export CUDA_VISIBLE_DEVICES=<TRAIN_GPU> && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python experiments/rlvr/train_grpo.py --config experiments/rlvr/configs/grpo_v1.yaml --smoke 2>&1 | tail -40'
```
Expected: 加载 base+adapter → dataset 4 条 → GRPOTrainer 跑 2 步 → 打印 summary JSON（含 train_reward）、无 Traceback、`grpo_train_log.json` 写出。

冒烟若报 trl API 不匹配（如 `processing_class` 参数名、reward_funcs 签名），按报错对齐 trl 0.14 实际 API 后再试——这是冒烟的目的。常见点：trl 0.14 `GRPOTrainer` 用 `processing_class`（非 `tokenizer`）；reward 回调收到的 kwargs 列名 = dataset 列名（`task_json`）。

- [ ] **Step 4: 正式训练（2 卡 + vllm，后台）**

```bash
ssh Song-3-Wu 'cd /NAS/yesh/VeriLong-RL && source /NAS/yesh/miniconda3/etc/profile.d/conda.sh && conda activate verilong_rl && export CUDA_VISIBLE_DEVICES=<TRAIN_GPU>,<VLLM_GPU> && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && nohup python experiments/rlvr/train_grpo.py --config experiments/rlvr/configs/grpo_v1.yaml > /NAS/yesh/VeriLong-RL/grpo_train.log 2>&1 & disown; sleep 10; tail -n 20 /NAS/yesh/VeriLong-RL/grpo_train.log'
```
Expected: 进程起、模型加载、vllm 初始化、开始按 step 打 reward。

- [ ] **Step 5: 监控训练（盯 reward 曲线 + OOM/Traceback）**

```bash
ssh Song-3-Wu 'grep -E "reward|loss|OutOfMemory|Traceback" /NAS/yesh/VeriLong-RL/grpo_train.log | tail -30'
```
Expected: reward 随 step 有变化（理想是上升趋势）；100 步后 `grpo_train_log.json` 写出、adapter 存到 `checkpoints/qwen2_5_7b_sft_v1_grpo`。
失败处理：OOM → 降 `vllm_gpu_memory_utilization` 或 `num_generations` 或 `max_prompt_length`；被抢占 → 换更空的卡（见 [[cluster-gpu-usage-tang-song]]，Song-3 大卡不易被挤）。

- [ ] **Step 6: 确认 adapter 落盘**

```bash
ssh Song-3-Wu 'ls -la /NAS/yesh/VeriLong-RL/checkpoints/qwen2_5_7b_sft_v1_grpo/adapter_model.safetensors /NAS/yesh/VeriLong-RL/checkpoints/qwen2_5_7b_sft_v1_grpo/grpo_train_log.json'
```
Expected: adapter (~160MB) + train log 都在。

（本任务无 commit —— 产物是 checkpoint，不入 git。代码已在前序任务提交。）

## Task 7: v1+RL 评测 + 四/五方对比 + memory

**目的：** 在 hard retrieval 子集上评 v1+RL，与 v1-gold 同口径对比，回答 RQ3。

**Files:**
- Create: `data/pilot/hard_eval_retrieval_16k.jsonl`（hard 16K 中 retrieval 子集；从已有 `hard_eval_sft_16k.jsonl` 过滤）
- 评测产物：`results/raw/open_source/qwen_sft_v1_grpo_*`（gitignored）

- [ ] **Step 1: 构造 hard retrieval 评测子集（集群）**

```bash
ssh Song-3-Wu 'cd /NAS/yesh/VeriLong-RL && source /NAS/yesh/miniconda3/etc/profile.d/conda.sh && conda activate verilong_rl && python -c "
import json
rows=[json.loads(l) for l in open(\"data/pilot/hard_eval_sft_16k.jsonl\") if l.strip()]
ret=[d for d in rows if d[\"task_family\"]==\"anti_distractor_retrieval\"]
open(\"data/pilot/hard_eval_retrieval_16k.jsonl\",\"w\").write(\"\".join(json.dumps(d)+\"\n\" for d in ret))
print(\"retrieval hard16k:\", len(ret))
"'
```
Expected: `retrieval hard16k: 9`（hard_eval_sft_16k 每族 9）。

- [ ] **Step 2: 评 v1+RL（hard retrieval）**

```bash
ssh Song-3-Wu 'cd /NAS/yesh/VeriLong-RL && source /NAS/yesh/miniconda3/etc/profile.d/conda.sh && conda activate verilong_rl && export CUDA_VISIBLE_DEVICES=<GPU> && export TRANSFORMERS_OFFLINE=1 && python experiments/eval_open_source/run_hf_eval.py --tasks data/pilot/hard_eval_retrieval_16k.jsonl --model Qwen/Qwen2.5-7B-Instruct --adapter checkpoints/qwen2_5_7b_sft_v1_grpo --out results/raw/open_source/qwen_sft_v1_grpo_hard_ret.jsonl && python scripts/score_outputs.py --tasks data/pilot/hard_eval_retrieval_16k.jsonl --outputs results/raw/open_source/qwen_sft_v1_grpo_hard_ret.jsonl --scored results/raw/open_source/qwen_sft_v1_grpo_hard_ret_scored.jsonl --summary results/raw/open_source/qwen_sft_v1_grpo_hard_ret_summary.json && echo GRPO_EVAL_DONE'
```
Expected: `GRPO_EVAL_DONE`，summary 写出。

- [ ] **Step 3: 评 v1-gold 同口径 baseline（hard retrieval）**

同 Step 2，但 `--adapter checkpoints/qwen2_5_7b_sft_v1`，输出 `qwen_sft_v1_hard_ret_*`。（v1 之前评的是 hard 全族 27 条；此处单独评 retrieval 9 条以与 v1+RL 同口径对比。）

- [ ] **Step 4: 打印对比表**

```bash
ssh Song-3-Wu 'cd /NAS/yesh/VeriLong-RL && source /NAS/yesh/miniconda3/etc/profile.d/conda.sh && conda activate verilong_rl && python -c "
import json
for tag,f in [(\"v1-gold\",\"qwen_sft_v1_hard_ret\"),(\"v1+RL\",\"qwen_sft_v1_grpo_hard_ret\")]:
    d=json.load(open(f\"results/raw/open_source/{f}_summary.json\"))
    print(tag, \"reward=%.3f\"%d[\"reward_total_mean\"], \"cit_prec=%.3f\"%d[\"citation_precision_mean\"], \"overcite=%.3f\"%d[\"overcitation_rate_mean\"], \"distractor=%.3f\"%d[\"distractor_citation_rate_mean\"], \"ans=%.3f\"%d[\"answer_normalized_match_mean\"])
"'
```
Expected: 两行对比。判读：v1+RL 的 cit_prec↑ 或 overcite↓ 且 ans 不降 → RQ3 正向；持平 → 饱和结论；下降 → 记录退化分析。

- [ ] **Step 5: 更新 memory**

把 RLVR 结果写入 `C:\Users\ysh20\.claude\projects\D--USTC-2026Summer---\memory\`：
- 更新 [[sft-warmup-plan]] 或新建 `rlvr-grpo-plan.md`：起点 v1、GRPO 配置（G=8/beta=0.04/lr 1e-6/100 步）、训练 reward 曲线走向、v1 vs v1+RL hard retrieval 对比真实数字、RQ3 结论。
- 更新 `MEMORY.md` 索引一行。

- [ ] **Step 6: Commit 代码与数据子集（feature 分支）**

```bash
git add data/pilot/hard_eval_retrieval_16k.jsonl
git commit -m "chore: add hard retrieval eval subset for GRPO comparison"
```
（结果文件 results/raw 已 gitignored，不提交。）

---

## Self-Review 检查（写计划后自查，已完成）

- **Spec 覆盖**：起点 v1✓(T4)、复用 reward✓(T1-2)、retrieval 8K✓(T3-4)、vllm 2卡✓(T4,T6)、hard 对比判据✓(T7)、smoke 回退✓(T5,T6)、reward 同尺一致性测试✓(T2)。全 spec 要求有对应 task。
- **占位扫描**：无 TBD/TODO；每个 code step 有完整代码；命令有 expected。
- **类型一致性**：`make_reward_fn`/`score_output_record`/`iter_grpo_rows`/`load_grpo_dataset`/`build_prompt_messages` 签名跨 task 一致；reward 回调列名 `task_json` 与 data.py 产出列名一致；GRPOConfig 字段名对齐 trl 0.14（smoke 阶段验证并按需修正）。
- **已知不确定性**（诚实标注，靠 Task 6 smoke 收敛）：trl 0.14 GRPOTrainer 的确切参数名（`processing_class` vs `tokenizer`）、use_vllm 的 2 卡布局机制——这些在集群 smoke 第一次实跑时对齐，计划已把 smoke 设为正式训练前的强制门。





