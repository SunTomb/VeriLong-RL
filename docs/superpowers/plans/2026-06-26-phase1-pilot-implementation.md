# Phase 1 Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成 VeriLong-RL Phase 1 的 Pilot 闭环：可生成、可校验、可解析、可评测、可展示的 1K–2K synthetic multi-document evidence benchmark，并为 Core/SFT/RLVR/Demo 扩展保留清晰接口。

**Architecture:** 先以 Python benchmark package 打通数据 schema、三类规则生成器、parser、metrics、reward 与 pilot eval；再用文件型 JSONL/JSON 结果驱动 FastAPI + React dashboard。训练与 RLVR 脚本只在 Pilot 闭环验证后接入，避免在数据与指标不稳定时占用 GPU 或扩大工程面。

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, Typer, JSONL, FastAPI, React + Vite + TypeScript, Recharts/ECharts（二选一，默认 Recharts）, Qwen2.5-7B/Instruct + LoRA/QLoRA, GRPO/RLVR on Tang/Song GPU cluster.

---

## 0. 当前理解与阶段门控

### 项目目标

VeriLong-RL 的定位是 **A verifiable long-context benchmark for evidence-grounded reasoning and RLVR**。项目不是单纯 survey，而是要在固定时间内形成：

1. 可验证 long-context benchmark/dataset；
2. answer/citation/reasoning/error taxonomy 分解评测；
3. API frontier models 与开源 7B line 的离线对比；
4. Qwen2.5-7B / Qwen2.5-7B-Instruct 的 SFT warmup 与小规模 RLVR/GRPO 尝试；
5. Benchmark Portal + live API demo；
6. 最终汇报材料：survey motivation → gap → benchmark design → experiments → demo。

### 阶段门控

- **Phase 1 是唯一不能牺牲的主线**：synthetic multi-document evidence benchmark。
- Phase 1 核心任务族：
  - `anti_distractor_retrieval`
  - `multi_hop_reasoning`
  - `temporal_update`
- Phase 2 真实论文/技术报告任务、Phase 3 repo-level 任务只能在 Phase 1 完成后询问用户是否继续。
- 数据规模必须 **Pilot 1K–2K → Core 10K–20K → Full 50K+**，不能一开始跳 Full。
- 模型输出格式固定为：

```text
Evidence: E03, E17
Steps:
1. E03 establishes the initial condition.
2. E17 updates or combines the relevant fact.
3. Therefore the answer is ...
Answer: ...
```

### 当前状态

- 设计文档已完成：`docs/superpowers/specs/2026-06-26-verilong-rl-design.md`。
- 本地记忆已迁移：`memory/MEMORY.md`、`memory/long-context-experiment-scope.md`、`memory/cluster-resources.md`、`memory/project-decisions.md`。
- GPU/模型资源已知：Tang-1/2/3-Wu 各 8×A40，Song-3-Wu 8×A100；NAS `/NAS/yesh`；HF cache 包含 Qwen2.5-7B/Instruct、Llama、Self-RAG、Contriever、MiniLM。
- 当前还没有完整工程脚手架；本计划批准前不创建 benchmark/backend/frontend 完整代码目录。

---

## 1. Milestone 1：Pilot 闭环优先级

### Milestone 1 目标

在本地先完成 1K–2K pilot tasks 的闭环：

1. 三个任务族均能规则生成样本；
2. 每个样本有唯一 gold answer、gold evidence IDs、distractor/stale IDs、expected steps、metadata；
3. 样本可通过程序校验；
4. 模型输出 parser 可解析 Answer / Evidence / Steps；
5. metrics 和 reward 可对模拟预测、人工样例、API 输出统一评分；
6. pilot eval 能读取 JSONL、运行 baseline 或 cached outputs、生成 summary JSON；
7. demo 所需 case JSON 能从 pilot 结果导出。

### 第一批应该创建的目录与文件

只在计划批准后创建以下首批文件；这是 Milestone 1 所需的最小工程面：

```text
benchmark/
  __init__.py
  schemas/
    __init__.py
    task.py
    prediction.py
    metrics.py
  generator/
    __init__.py
    common.py
    retrieval.py
    multihop.py
    temporal.py
    build_pilot.py
  validators/
    __init__.py
    task_validator.py
  parser/
    __init__.py
    output_parser.py
  metrics/
    __init__.py
    answer.py
    citation.py
    format.py
    aggregate.py
  reward/
    __init__.py
    programmatic.py
  eval/
    __init__.py
    run_pilot_eval.py
scripts/
  generate_pilot.py
  validate_pilot.py
  parse_outputs.py
  score_outputs.py
tests/
  benchmark/
    test_schema.py
    test_generators.py
    test_validator.py
    test_parser.py
    test_metrics.py
    test_reward.py
data/
  pilot/
    README.md
results/
  pilot/
    README.md
configs/
  pilot.yaml
pyproject.toml
README.md
```

Milestone 1 暂不创建 `backend/`、`frontend/`、`experiments/sft/`、`experiments/rlvr/` 的完整实现，只在本计划后续任务中定义何时进入。

---

## 2. File Structure Map

### Benchmark core

- `benchmark/schemas/task.py`：Pydantic 数据结构，定义 `VeriLongTask`、`EvidenceDocument`、`TaskMetadata`、枚举字段。
- `benchmark/schemas/prediction.py`：解析后的模型输出结构 `ParsedPrediction` 与 parser error flags。
- `benchmark/schemas/metrics.py`：单样本 metric 与聚合 summary 结构。
- `benchmark/generator/common.py`：实体、证据 ID、文档拼接、split 分配、context token 近似工具。
- `benchmark/generator/retrieval.py`：anti-distractor retrieval 规则生成。
- `benchmark/generator/multihop.py`：multi-hop evidence reasoning 规则生成。
- `benchmark/generator/temporal.py`：conflict / temporal update 规则生成。
- `benchmark/generator/build_pilot.py`：按配置混合三个任务族并写 JSONL。
- `benchmark/validators/task_validator.py`：唯一答案、证据 ID、角色、metadata、parser-consumability 校验。
- `benchmark/parser/output_parser.py`：解析 Evidence / Steps / Answer 格式，不丢弃失败输出。
- `benchmark/metrics/answer.py`：exact/normalized/type-aware answer scoring。
- `benchmark/metrics/citation.py`：precision/recall/F1、distractor/stale/invalid/overcitation。
- `benchmark/metrics/format.py`：format validity 与 error flags。
- `benchmark/metrics/aggregate.py`：按 task family、difficulty、context length、position 等维度聚合。
- `benchmark/reward/programmatic.py`：程序化 reward，初始权重来自设计文档。
- `benchmark/eval/run_pilot_eval.py`：读取 tasks 与 outputs，产出 scored JSONL、summary JSON、case export。

### Scripts/config/data/results

- `configs/pilot.yaml`：Pilot 数据规模、任务族比例、难度分布、seed、目标输出路径。
- `scripts/generate_pilot.py`：CLI 包装 `build_pilot.py`。
- `scripts/validate_pilot.py`：CLI 校验 JSONL。
- `scripts/parse_outputs.py`：CLI 解析模型输出 JSONL。
- `scripts/score_outputs.py`：CLI 评分并聚合。
- `data/pilot/tasks.jsonl`：生成后文件，不手写。
- `results/pilot/*.jsonl|*.json`：评测后文件，不伪造实验值。

### Later milestones

- `experiments/eval_api/`：API 调用、batch/queue/cache、provider clients。
- `experiments/eval_open_source/`：Qwen/Llama/Contriever/MiniLM 推理与 baseline。
- `experiments/sft/`：SFT 数据转换、LoRA/QLoRA 配置、训练脚本。
- `experiments/rlvr/`：GRPO/RLVR reward adapter 与小规模训练。
- `backend/`：FastAPI 读取 dataset/results/cached live examples。
- `frontend/`：React Benchmark Portal。
- `docs/presentation/`：图表、case study、demo script、讲稿。

---

## 3. Task Breakdown

### Task 1: Project Python baseline and schemas

**Files:**
- Create: `pyproject.toml`
- Create: `benchmark/__init__.py`
- Create: `benchmark/schemas/__init__.py`
- Create: `benchmark/schemas/task.py`
- Create: `benchmark/schemas/prediction.py`
- Create: `benchmark/schemas/metrics.py`
- Create: `tests/benchmark/test_schema.py`

- [ ] **Step 1: Write failing schema tests**

```python
from benchmark.schemas.task import EvidenceDocument, TaskMetadata, VeriLongTask


def test_task_requires_gold_evidence_in_documents():
    task = VeriLongTask(
        id="vlr_pilot_000001",
        task_family="anti_distractor_retrieval",
        difficulty="easy",
        question="Which access code is assigned to Project Orion?",
        documents=[
            EvidenceDocument(doc_id="D01", evidence_id="E01", text="Project Orion uses access code A17.", role="gold"),
            EvidenceDocument(doc_id="D02", evidence_id="E02", text="Project Oriole uses access code B42.", role="distractor"),
        ],
        gold_answer="A17",
        gold_evidence_ids=["E01"],
        distractor_evidence_ids=["E02"],
        stale_evidence_ids=[],
        expected_steps=["Use E01 to identify Project Orion's access code."],
        metadata=TaskMetadata(target_context_tokens=8000, evidence_position="front", distractor_strength="lexical"),
    )
    assert task.gold_evidence_ids == ["E01"]
    assert task.documents_by_evidence_id()["E01"].role == "gold"
```

- [ ] **Step 2: Run schema test to verify it fails**

Run:

```bash
python -m pytest tests/benchmark/test_schema.py -q
```

Expected: import error for `benchmark.schemas.task` because code has not been created.

- [ ] **Step 3: Implement minimal schemas**

Implement exact enum-like literals:

```python
TaskFamily = Literal["anti_distractor_retrieval", "multi_hop_reasoning", "temporal_update"]
EvidenceRole = Literal["gold", "distractor", "stale", "neutral"]
Difficulty = Literal["easy", "medium", "hard"]
```

`VeriLongTask` must include the JSONL fields from the design doc and helper methods:

```python
def evidence_ids(self) -> set[str]
def documents_by_evidence_id(self) -> dict[str, EvidenceDocument]
def gold_documents(self) -> list[EvidenceDocument]
```

- [ ] **Step 4: Run tests**

Run:

```bash
python -m pytest tests/benchmark/test_schema.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml benchmark tests/benchmark/test_schema.py
git commit -m "feat: define VeriLong task schemas"
```

### Task 2: Deterministic generators for three task families

**Files:**
- Create: `benchmark/generator/__init__.py`
- Create: `benchmark/generator/common.py`
- Create: `benchmark/generator/retrieval.py`
- Create: `benchmark/generator/multihop.py`
- Create: `benchmark/generator/temporal.py`
- Create: `tests/benchmark/test_generators.py`

- [ ] **Step 1: Write generator tests**

```python
from benchmark.generator.retrieval import generate_retrieval_task
from benchmark.generator.multihop import generate_multihop_task
from benchmark.generator.temporal import generate_temporal_task


def assert_common_task_invariants(task):
    evidence_ids = task.evidence_ids()
    assert task.gold_answer
    assert set(task.gold_evidence_ids).issubset(evidence_ids)
    assert set(task.distractor_evidence_ids).issubset(evidence_ids)
    assert set(task.stale_evidence_ids).issubset(evidence_ids)
    assert 1 <= len(task.expected_steps) <= 4


def test_retrieval_generator_has_gold_and_distractors():
    task = generate_retrieval_task(task_id="vlr_pilot_000001", seed=1, target_context_tokens=8000)
    assert task.task_family == "anti_distractor_retrieval"
    assert_common_task_invariants(task)
    assert len(task.gold_evidence_ids) >= 1
    assert len(task.distractor_evidence_ids) >= 1


def test_multihop_generator_has_multiple_gold_evidence():
    task = generate_multihop_task(task_id="vlr_pilot_000002", seed=2, hop_count=3, target_context_tokens=8000)
    assert task.task_family == "multi_hop_reasoning"
    assert_common_task_invariants(task)
    assert len(task.gold_evidence_ids) == 3


def test_temporal_generator_marks_stale_evidence():
    task = generate_temporal_task(task_id="vlr_pilot_000003", seed=3, update_count=1, target_context_tokens=8000)
    assert task.task_family == "temporal_update"
    assert_common_task_invariants(task)
    assert len(task.stale_evidence_ids) >= 1
```

- [ ] **Step 2: Run tests to verify failure**

```bash
python -m pytest tests/benchmark/test_generators.py -q
```

Expected: import errors for generator modules.

- [ ] **Step 3: Implement deterministic common utilities**

`common.py` must provide:

```python
def make_rng(seed: int) -> random.Random
def evidence_id(index: int) -> str  # E01, E02, ...
def doc_id(index: int) -> str       # D01, D02, ...
def approximate_tokens(text: str) -> int
def pad_neutral_documents(documents, target_context_tokens, rng) -> list[EvidenceDocument]
```

Neutral padding should use deterministic synthetic filler paragraphs that do not mention gold entities.

- [ ] **Step 4: Implement three minimal generators**

Rules:

- Retrieval: one target entity, one correct attribute, at least four lexically similar distractors for medium/hard.
- Multihop: chain like entity → protocol → condition/rule → answer; `hop_count` controls gold evidence count.
- Temporal: old fact + update fact + stale checklist; answer comes from latest/update evidence.

Each generator returns `VeriLongTask`, not raw dict.

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/benchmark/test_generators.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add benchmark/generator tests/benchmark/test_generators.py
git commit -m "feat: add deterministic pilot task generators"
```

### Task 3: Task validator and pilot JSONL builder

**Files:**
- Create: `benchmark/validators/__init__.py`
- Create: `benchmark/validators/task_validator.py`
- Create: `benchmark/generator/build_pilot.py`
- Create: `configs/pilot.yaml`
- Create: `scripts/generate_pilot.py`
- Create: `scripts/validate_pilot.py`
- Create: `data/pilot/README.md`
- Test: `tests/benchmark/test_validator.py`

- [ ] **Step 1: Write validator tests**

```python
import pytest
from benchmark.generator.retrieval import generate_retrieval_task
from benchmark.validators.task_validator import validate_task


def test_valid_generated_task_passes_validation():
    task = generate_retrieval_task(task_id="vlr_pilot_000001", seed=11, target_context_tokens=8000)
    report = validate_task(task)
    assert report.valid is True
    assert report.errors == []


def test_missing_gold_evidence_fails_validation():
    task = generate_retrieval_task(task_id="vlr_pilot_000001", seed=11, target_context_tokens=8000)
    task.gold_evidence_ids = ["E99"]
    report = validate_task(task)
    assert report.valid is False
    assert "missing_gold_evidence_id:E99" in report.errors
```

- [ ] **Step 2: Run validator tests to verify failure**

```bash
python -m pytest tests/benchmark/test_validator.py -q
```

Expected: import error for validator.

- [ ] **Step 3: Implement validation report**

`validate_task(task)` must check:

```text
- unique sample id non-empty
- all document evidence_id values are unique
- gold_evidence_ids exist and refer to role=gold documents
- distractor_evidence_ids exist and do not overlap gold
- stale_evidence_ids exist and do not overlap gold
- gold_answer non-empty
- expected_steps length is 1–4 for generated labels; parser target later enforces 2–4 on model output
- task_family-specific metadata exists
```

- [ ] **Step 4: Implement pilot config**

Initial `configs/pilot.yaml`:

```yaml
seed: 20260626
output_path: data/pilot/tasks.jsonl
size: 1200
task_mix:
  anti_distractor_retrieval: 400
  multi_hop_reasoning: 400
  temporal_update: 400
target_context_tokens:
  - 8000
  - 16000
difficulty:
  easy: 0.30
  medium: 0.50
  hard: 0.20
splits:
  train: 0.70
  dev: 0.10
  test: 0.20
extra_splits:
  judge_subset_size: 60
  live_demo_subset_size: 30
```

- [ ] **Step 5: Implement CLI scripts**

Commands:

```bash
python scripts/generate_pilot.py --config configs/pilot.yaml
python scripts/validate_pilot.py data/pilot/tasks.jsonl
```

`generate_pilot.py` writes JSONL only after every generated task passes validation. `validate_pilot.py` exits non-zero when any task fails.

- [ ] **Step 6: Run generation smoke test**

```bash
python scripts/generate_pilot.py --config configs/pilot.yaml
python scripts/validate_pilot.py data/pilot/tasks.jsonl
```

Expected:

```text
generated=1200 valid=1200 output=data/pilot/tasks.jsonl
validated=1200 valid=1200 invalid=0
```

- [ ] **Step 7: Commit**

```bash
git add benchmark/validators benchmark/generator/build_pilot.py configs scripts data/pilot/README.md tests/benchmark/test_validator.py
git commit -m "feat: build validated pilot dataset"
```

### Task 4: Output parser

**Files:**
- Create: `benchmark/parser/__init__.py`
- Create: `benchmark/parser/output_parser.py`
- Test: `tests/benchmark/test_parser.py`
- Create: `scripts/parse_outputs.py`

- [ ] **Step 1: Write parser tests**

```python
from benchmark.parser.output_parser import parse_model_output


def test_parse_valid_answer_evidence_steps():
    text = """Evidence: E03, E17
Steps:
1. E03 establishes the initial condition.
2. E17 updates the relevant fact.
3. Therefore the answer is Team Delta.
Answer: Team Delta
"""
    parsed = parse_model_output(text, valid_evidence_ids={"E03", "E17", "E21"})
    assert parsed.format_valid is True
    assert parsed.pred_evidence_ids == ["E03", "E17"]
    assert parsed.pred_answer == "Team Delta"
    assert len(parsed.pred_steps) == 3
    assert parsed.error_flags == []


def test_parse_missing_answer_records_error():
    text = "Evidence: E03\nSteps:\n1. E03 supports it."
    parsed = parse_model_output(text, valid_evidence_ids={"E03"})
    assert parsed.format_valid is False
    assert "missing_answer" in parsed.error_flags


def test_parse_invalid_evidence_records_error():
    text = "Evidence: E99\nSteps:\n1. E99 supports it.\nAnswer: X"
    parsed = parse_model_output(text, valid_evidence_ids={"E01"})
    assert parsed.format_valid is False
    assert "invalid_evidence_id:E99" in parsed.error_flags
```

- [ ] **Step 2: Run parser tests to verify failure**

```bash
python -m pytest tests/benchmark/test_parser.py -q
```

Expected: import error for parser.

- [ ] **Step 3: Implement parser behavior**

Parser rules:

- Accept `Evidence:` line with comma/space separated IDs matching `E\d+`.
- Accept `Steps:` followed by numbered lines until `Answer:`.
- Extract `Answer:` as all remaining text after marker; trim whitespace.
- `format_valid` is true only when answer exists, evidence exists, step count is 2–4, and no invalid evidence IDs exist.
- Keep `unparsed_text` always.
- Do not throw on malformed model output.

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/benchmark/test_parser.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmark/parser scripts/parse_outputs.py tests/benchmark/test_parser.py
git commit -m "feat: parse evidence-grounded model outputs"
```

### Task 5: Metrics and programmatic reward

**Files:**
- Create: `benchmark/metrics/__init__.py`
- Create: `benchmark/metrics/answer.py`
- Create: `benchmark/metrics/citation.py`
- Create: `benchmark/metrics/format.py`
- Create: `benchmark/metrics/aggregate.py`
- Create: `benchmark/reward/__init__.py`
- Create: `benchmark/reward/programmatic.py`
- Test: `tests/benchmark/test_metrics.py`
- Test: `tests/benchmark/test_reward.py`
- Create: `scripts/score_outputs.py`

- [ ] **Step 1: Write metrics tests**

```python
from benchmark.metrics.answer import normalized_match
from benchmark.metrics.citation import citation_scores


def test_normalized_match_ignores_case_and_articles():
    assert normalized_match("The Team Delta", "team delta") == 1.0


def test_citation_scores_penalize_distractor_and_stale():
    scores = citation_scores(
        pred_evidence_ids=["E01", "E02", "E03"],
        gold_evidence_ids=["E01", "E04"],
        distractor_evidence_ids=["E02"],
        stale_evidence_ids=["E03"],
        valid_evidence_ids={"E01", "E02", "E03", "E04"},
    )
    assert scores.precision == 1 / 3
    assert scores.recall == 1 / 2
    assert scores.distractor_citation_rate == 1 / 3
    assert scores.stale_citation_rate == 1 / 3
```

- [ ] **Step 2: Write reward tests**

```python
from benchmark.reward.programmatic import compute_reward


def test_reward_rewards_answer_and_gold_citation():
    reward = compute_reward(
        answer_score=1.0,
        citation_f1=1.0,
        reasoning_score=0.8,
        format_score=1.0,
        distractor_rate=0.0,
        stale_rate=0.0,
        invalid_rate=0.0,
    )
    assert reward.total > 0.8


def test_reward_penalizes_distractor_and_invalid_citations():
    clean = compute_reward(1.0, 1.0, 0.8, 1.0, 0.0, 0.0, 0.0)
    noisy = compute_reward(1.0, 0.5, 0.8, 1.0, 0.5, 0.0, 0.5)
    assert noisy.total < clean.total
```

- [ ] **Step 3: Run tests to verify failure**

```bash
python -m pytest tests/benchmark/test_metrics.py tests/benchmark/test_reward.py -q
```

Expected: import errors for metrics/reward modules.

- [ ] **Step 4: Implement answer/citation/format metrics**

Metrics must include at least:

```text
answer_exact_match
answer_normalized_match
format_valid
step_count_valid
citation_precision
citation_recall
citation_f1
all_gold_evidence_recall
distractor_citation_rate
stale_citation_rate
invalid_citation_rate
overgitation_rate
```

Note: use field name `overcitation_rate` in code; do not use the typo above.

- [ ] **Step 5: Implement reward formula**

Initial weights from design doc:

```python
DEFAULT_REWARD_WEIGHTS = {
    "answer": 0.40,
    "citation": 0.25,
    "reasoning": 0.20,
    "format": 0.10,
    "distractor": 0.15,
    "stale": 0.15,
    "invalid": 0.10,
}
```

Formula:

```text
R = 0.40 * R_answer
  + 0.25 * R_citation
  + 0.20 * R_reasoning
  + 0.10 * R_format
  - 0.15 * P_distractor
  - 0.15 * P_stale
  - 0.10 * P_invalid
```

Return both `total` and component breakdown.

- [ ] **Step 6: Run tests**

```bash
python -m pytest tests/benchmark/test_metrics.py tests/benchmark/test_reward.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add benchmark/metrics benchmark/reward scripts/score_outputs.py tests/benchmark/test_metrics.py tests/benchmark/test_reward.py
git commit -m "feat: score pilot outputs with evidence metrics"
```

### Task 6: Pilot eval loop with synthetic/cached model outputs

**Files:**
- Create: `benchmark/eval/__init__.py`
- Create: `benchmark/eval/run_pilot_eval.py`
- Create: `results/pilot/README.md`
- Modify: `scripts/score_outputs.py`

- [ ] **Step 1: Define output JSONL contract**

Model output JSONL rows must be:

```json
{
  "task_id": "vlr_pilot_000001",
  "model": "oracle_format_baseline",
  "output_text": "Evidence: E01\nSteps:\n1. E01 states the relevant fact.\n2. Therefore the answer is A17.\nAnswer: A17",
  "metadata": {"source": "synthetic_oracle"}
}
```

- [ ] **Step 2: Add oracle/corrupted baselines**

`run_pilot_eval.py` should support two no-API local baselines:

```text
oracle_format_baseline: uses gold answer + gold evidence in required format
corrupted_distractor_baseline: answers gold answer but cites one distractor when available
```

These are not experimental model results; they are parser/metrics smoke tests.

- [ ] **Step 3: Run pilot eval smoke test**

```bash
python scripts/generate_pilot.py --config configs/pilot.yaml
python -m benchmark.eval.run_pilot_eval \
  --tasks data/pilot/tasks.jsonl \
  --baseline oracle_format_baseline \
  --out-dir results/pilot/oracle_smoke
python -m benchmark.eval.run_pilot_eval \
  --tasks data/pilot/tasks.jsonl \
  --baseline corrupted_distractor_baseline \
  --out-dir results/pilot/corrupted_smoke
```

Expected:

```text
oracle_smoke summary has format_valid_rate=1.0 and citation_f1_mean=1.0
corrupted_smoke summary has distractor_citation_rate_mean > 0.0
```

- [ ] **Step 4: Export demo case JSON**

Write:

```text
results/pilot/oracle_smoke/cases_for_demo.json
results/pilot/corrupted_smoke/cases_for_demo.json
```

Each case must include question, documents, gold evidence IDs, distractor/stale IDs, model output, parsed output, metric breakdown, and error type.

- [ ] **Step 5: Commit**

```bash
git add benchmark/eval results/pilot/README.md scripts/score_outputs.py
git commit -m "feat: run pilot evaluation smoke tests"
```

### Task 7: API eval adapter design and Claude constraints

**Files:**
- Create: `experiments/eval_api/README.md`
- Create later after approval: `experiments/eval_api/claude_client.py`
- Create later after approval: `experiments/eval_api/run_api_eval.py`

- [ ] **Step 1: Document API constraints before coding**

Use these Claude defaults in the implementation:

```text
Default strong Claude model: claude-opus-4-8
Adaptive thinking: thinking={"type": "adaptive"}
No budget_tokens on Opus 4.8/4.7/Fable 5
No temperature/top_p/top_k on Opus 4.8/4.7/Fable 5
Use streaming for long input or large max_tokens
Use output_config.format for structured outputs; do not use deprecated output_format
Do not use assistant prefill to force JSON or output shape
Offline batch eval should use queue/retry/cache; live demo should stay small and have cached fallback
```

- [ ] **Step 2: API eval request format**

Prompt must include:

```text
You are solving a VeriLong-RL evidence-grounded long-context task.
Return exactly this format:
Evidence: E01, E02
Steps:
1. One short sentence grounded in cited evidence.
2. One short sentence grounded in cited evidence.
Answer: final answer only
```

Do not ask the model for hidden chain-of-thought. Steps are short evidence-grounded rationale only.

- [ ] **Step 3: API cache design**

Cache key fields:

```text
task_id
model
prompt_version
task_hash
```

Cache file path:

```text
results/raw/api_cache/{model}/{task_id}.json
```

- [ ] **Step 4: Pilot API eval entry gate**

Only run API eval after Tasks 1–6 pass. First run must be a small dev subset:

```bash
python experiments/eval_api/run_api_eval.py \
  --tasks data/pilot/tasks.jsonl \
  --split dev \
  --limit 30 \
  --model claude-opus-4-8 \
  --out results/raw/api/claude_opus_4_8_pilot_dev30.jsonl
```

Expected: output file exists, no fabricated metrics, parser/score script can process it.

### Task 8: Open-source eval, SFT, and RLVR plan gates

**Files:**
- Create later: `experiments/eval_open_source/README.md`
- Create later: `experiments/sft/README.md`
- Create later: `experiments/rlvr/README.md`

- [ ] **Step 1: Open-source eval gate**

Only start after pilot parser/metrics are stable. First model:

```text
Qwen/Qwen2.5-7B-Instruct
```

Run on 8K/16K pilot dev/test subset first; 32K/64K are evaluation stretch, not initial training requirement.

- [ ] **Step 2: SFT data conversion gate**

SFT target format uses the fixed Answer/Evidence/Steps output. Training examples are:

```json
{
  "messages": [
    {"role": "system", "content": "You answer VeriLong-RL tasks with cited evidence IDs."},
    {"role": "user", "content": "Question...\n\nContext..."},
    {"role": "assistant", "content": "Evidence: E01\nSteps:\n1. ...\n2. ...\nAnswer: ..."}
  ],
  "metadata": {"task_id": "vlr_train_000001", "task_family": "multi_hop_reasoning"}
}
```

- [ ] **Step 3: SFT first run**

Default training line:

```text
Qwen2.5-7B-Instruct + LoRA/QLoRA
sequence length: 8K/16K first
cluster: Tang A40 or Song A100 depending availability
NAS path: /NAS/yesh
```

Do not start GPU training until pilot data and metrics pass local tests.

- [ ] **Step 4: RLVR first run**

Default RLVR/GRPO starts from SFT checkpoint and only on retrieval family 8K/16K smoke test. Reward is programmatic; LLM judge is not online main reward.

- [ ] **Step 5: Failure fallback**

If RLVR is unstable, preserve:

```text
SFT results
reward analysis
oracle/corrupted baseline sanity checks
API vs Qwen-Instruct comparison
case studies showing failure modes
```

This still supports a complete benchmark + experiments + demo story.

### Task 9: Backend/frontend demo plan

**Files:**
- Create later: `backend/README.md`
- Create later: `frontend/README.md`

- [ ] **Step 1: Backend minimal endpoints**

FastAPI initial endpoints:

```text
GET  /api/summary
GET  /api/leaderboard
GET  /api/results/breakdown
GET  /api/cases
POST /api/tasks/generate
POST /api/live/run
GET  /api/training/curves
```

Milestone 1 only needs read-only local results plus `POST /api/tasks/generate` using generator. `POST /api/live/run` can initially return cached examples until API adapter is verified.

- [ ] **Step 2: Frontend minimal pages**

React pages in implementation order:

```text
Home
Dataset
CaseViewer
Leaderboard
Analysis
Training
LiveDemo
```

Milestone 1 frontend minimum is `Home + Dataset + CaseViewer` driven by `cases_for_demo.json`; no fake leaderboard values.

- [ ] **Step 3: Live demo stability contract**

Live demo defaults:

```text
context length: 4K/8K/16K only
model: API model only
open-source trained model: cached/offline results only
timeout: configured server-side
fallback: cached demo case
API key: server env only, never exposed to frontend
```

### Task 10: Presentation artifacts

**Files:**
- Create later: `docs/presentation/outline.md`
- Create later: `docs/presentation/demo_script.md`
- Create later: `results/figures/README.md`

- [ ] **Step 1: Produce pilot figures from real outputs only**

Allowed first figures after Task 6:

```text
pilot task distribution by family/difficulty/context length
oracle vs corrupted baseline metric sanity chart
case screenshots from generated tasks
parser failure examples if any
```

Do not fill leaderboard `value` placeholders until actual model runs exist.

- [ ] **Step 2: Presentation narrative**

Use this order:

```text
Survey Motivation
Gap: answer-only long-context evaluation hides evidence-use failures
VeriLong-RL Design
Pilot/Core/Full data pipeline
Metrics and reward decomposition
Experiments: API + Qwen + SFT + RLVR
Dashboard/live demo
Takeaways
```

- [ ] **Step 3: Demo script**

Demo script must include two paths:

```text
Happy path: generate small task → API call → parse → evidence alignment
Fallback path: load cached run → show same UI and explain API timeout/cost guard
```

---

## 4. Verification commands for Milestone 1

After implementing Tasks 1–6, run:

```bash
python -m pytest tests/benchmark -q
python scripts/generate_pilot.py --config configs/pilot.yaml
python scripts/validate_pilot.py data/pilot/tasks.jsonl
python -m benchmark.eval.run_pilot_eval --tasks data/pilot/tasks.jsonl --baseline oracle_format_baseline --out-dir results/pilot/oracle_smoke
python -m benchmark.eval.run_pilot_eval --tasks data/pilot/tasks.jsonl --baseline corrupted_distractor_baseline --out-dir results/pilot/corrupted_smoke
```

Expected Milestone 1 evidence:

```text
All benchmark tests pass.
Pilot JSONL has 1200 valid tasks.
Oracle baseline summary has citation_f1_mean=1.0 and format_valid_rate=1.0.
Corrupted baseline summary has non-zero distractor/stale penalties where applicable.
Demo case export exists and includes at least one case per task family.
```

---

## 5. Core / Full expansion gates

Proceed to Core only after Milestone 1 passes:

1. Increase dataset to 10K–20K.
2. Add LLM rewrite/paraphrase with strict preservation of answer/evidence IDs.
3. Add deterministic re-validation after rewrite.
4. Add judge subset audit with fixed schema and cached judge outputs.
5. Add API eval on balanced dev/test subsets.
6. Add Qwen2.5-7B-Instruct eval.
7. Add SFT conversion and short training.

Proceed to Full only after Core results and dashboard are stable:

1. 50K+ generation.
2. Hard split.
3. RLVR/GRPO runs.
4. Final analysis figures.
5. Public-facing benchmark packaging.

Phase 2/3 remain blocked until Phase 1 complete and user explicitly approves.

---

## 6. Self-review

### Spec coverage

- Dataset schema/generator: Tasks 1–3.
- Parser/metrics/reward: Tasks 4–5.
- Pilot eval: Task 6.
- API eval constraints: Task 7.
- SFT/RLVR gates: Task 8.
- Backend/frontend demo: Task 9.
- Presentation artifacts: Task 10.
- Phase gate constraints: Sections 0, 1, 5.

### Placeholder scan

This plan intentionally avoids fabricated results and does not use unresolved implementation placeholders. Experimental metric values must be produced by actual runs.

### Type consistency

Canonical task family strings are:

```text
anti_distractor_retrieval
multi_hop_reasoning
temporal_update
```

Canonical output parser fields are:

```text
pred_answer
pred_evidence_ids
pred_steps
format_valid
unparsed_text
error_flags
```

Canonical metric field spelling is `overcitation_rate`.
