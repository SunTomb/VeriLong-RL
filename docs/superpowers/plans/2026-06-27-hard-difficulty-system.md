# Hard Difficulty System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable hard-difficulty generation system for VeriLong-RL Phase 1 synthetic tasks so strong-model evaluation has a controllable reward gradient without expanding to Core/Full or Phase 2/3.

**Architecture:** Keep the existing generator modules as the task-family boundary and add a small profile resolver that maps `difficulty` plus YAML overrides to concrete generator kwargs. `build_pilot.py` becomes responsible for deterministic difficulty assignment and profile resolution; individual generators remain deterministic, schema-returning functions. The existing parser/metrics/reward formula stays unchanged so any reward change comes from task difficulty, not metric drift.

**Tech Stack:** Python 3.11+, Pydantic v2 schemas, PyYAML config loading, pytest, JSONL pilot data.

---

## 0. Scope and constraints

This plan is only for Phase 1 synthetic multi-document evidence tasks:

- Allowed task families: `anti_distractor_retrieval`, `multi_hop_reasoning`, `temporal_update`.
- Do not add Phase 2 real-paper tasks or Phase 3 repo-level tasks.
- Do not change the fixed model output format: `Evidence:` / `Steps:` / `Answer:`.
- Do not change reward weights or parser semantics in this plan.
- Do not fabricate model metrics. Model reward curves come only after real API/Open-source eval runs.
- Keep all split assignment stratified by task family.
- Commits require explicit user approval. Each task below includes an authorized commit command for later use; do not run it unless the user has approved committing.

---

## 1. File structure map

### New files

- `benchmark/generator/profiles.py`
  - Owns default difficulty profiles.
  - Validates family/difficulty names.
  - Merges per-difficulty YAML generation overrides.
  - Returns kwargs that are safe to pass to the family generator.

- `configs/hard.yaml`
  - Generates a 180-task all-hard pilot slice.
  - Uses 16K/32K target contexts only.
  - Keeps split ratios and extra subsets stratified through existing `build_pilot.py` logic.

### Modified files

- `benchmark/generator/retrieval.py`
  - Add `difficulty`, `distractor_count`, and `evidence_position` parameters.
  - Generate multiple hard distractors with lexically/adversarially similar project names.
  - Record `distractor_count` in `metadata.extra`.

- `benchmark/generator/multihop.py`
  - Support `hop_count` values 2, 3, 4, and 5.
  - Add `irrelevant_rule_count`, `conflicting_rule_count`, and `difficulty` parameters.
  - Generate a deterministic evidence chain whose gold evidence count equals `hop_count`.
  - Record irrelevant/conflicting counts in `metadata.extra`.

- `benchmark/generator/temporal.py`
  - Add `difficulty`, `stale_count`, and `evidence_position` parameters.
  - Generate an update chain where the latest update is the only gold evidence.
  - Mark old records, intermediate updates, and legacy copies as stale.
  - Record `stale_count` and `latest_update_index` in `metadata.extra`.

- `benchmark/generator/build_pilot.py`
  - Use config `difficulty` distribution to assign task difficulty deterministically.
  - Resolve profile kwargs before calling a generator.
  - Preserve existing stratified train/dev/test and extra subset behavior.

- `tests/benchmark/test_generators.py`
  - Add hard retrieval, variable-hop multihop, and hard temporal tests.
  - Update the old test that expected 2/4-hop multihop to raise.

- `tests/benchmark/test_build_pilot.py`
  - Add tests for deterministic difficulty distribution and hard generation overrides.

---

## 2. Implementation tasks

### Task 0: Prepare implementation branch

**Files:**
- No code files changed.

- [ ] **Step 1: Check working tree status**

Run:

```bash
git status --short
```

Expected: existing untracked project files may appear, but no unexpected modified tracked files from this task.

- [ ] **Step 2: Create and switch to a feature branch**

Run:

```bash
git switch -c feature/hard-difficulty-system
```

Expected: output like:

```text
Switched to a new branch 'feature/hard-difficulty-system'
```

If the branch already exists, run:

```bash
git switch feature/hard-difficulty-system
```

Expected: output like:

```text
Switched to branch 'feature/hard-difficulty-system'
```

---

### Task 1: Add difficulty profile resolver

**Files:**
- Create: `benchmark/generator/profiles.py`
- Modify: `tests/benchmark/test_build_pilot.py`

- [ ] **Step 1: Write failing profile tests**

Append these tests to `tests/benchmark/test_build_pilot.py`:

```python
from benchmark.generator.profiles import resolve_generation_kwargs


def test_resolve_generation_kwargs_uses_hard_defaults():
    kwargs = resolve_generation_kwargs(
        task_family="anti_distractor_retrieval",
        difficulty="hard",
        generation_config={},
    )

    assert kwargs["difficulty"] == "hard"
    assert kwargs["distractor_count"] >= 8
    assert kwargs["distractor_strength"] == "adversarial"
    assert kwargs["evidence_position"] == "random"


def test_resolve_generation_kwargs_applies_yaml_override():
    kwargs = resolve_generation_kwargs(
        task_family="multi_hop_reasoning",
        difficulty="hard",
        generation_config={
            "hard": {
                "multi_hop_reasoning": {
                    "hop_count": 5,
                    "irrelevant_rule_count": 8,
                    "conflicting_rule_count": 2,
                }
            }
        },
    )

    assert kwargs == {
        "difficulty": "hard",
        "hop_count": 5,
        "irrelevant_rule_count": 8,
        "conflicting_rule_count": 2,
    }
```

- [ ] **Step 2: Run profile tests to verify they fail**

Run:

```bash
python -m pytest tests/benchmark/test_build_pilot.py::test_resolve_generation_kwargs_uses_hard_defaults tests/benchmark/test_build_pilot.py::test_resolve_generation_kwargs_applies_yaml_override -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark.generator.profiles'`.

- [ ] **Step 3: Implement `benchmark/generator/profiles.py`**

Create `benchmark/generator/profiles.py` with this content:

```python
from copy import deepcopy
from typing import Any

from benchmark.schemas.task import Difficulty, TaskFamily


_GENERATOR_KWARGS = {
    "anti_distractor_retrieval": {
        "difficulty",
        "distractor_strength",
        "distractor_count",
        "evidence_position",
    },
    "multi_hop_reasoning": {
        "difficulty",
        "hop_count",
        "irrelevant_rule_count",
        "conflicting_rule_count",
    },
    "temporal_update": {
        "difficulty",
        "update_count",
        "stale_count",
        "evidence_position",
    },
}


_DEFAULT_PROFILES: dict[Difficulty, dict[TaskFamily, dict[str, Any]]] = {
    "easy": {
        "anti_distractor_retrieval": {
            "difficulty": "easy",
            "distractor_strength": "lexical",
            "distractor_count": 1,
            "evidence_position": "front",
        },
        "multi_hop_reasoning": {
            "difficulty": "easy",
            "hop_count": 2,
            "irrelevant_rule_count": 1,
            "conflicting_rule_count": 0,
        },
        "temporal_update": {
            "difficulty": "easy",
            "update_count": 1,
            "stale_count": 2,
            "evidence_position": "front",
        },
    },
    "medium": {
        "anti_distractor_retrieval": {
            "difficulty": "medium",
            "distractor_strength": "semantic",
            "distractor_count": 4,
            "evidence_position": "mixed",
        },
        "multi_hop_reasoning": {
            "difficulty": "medium",
            "hop_count": 3,
            "irrelevant_rule_count": 3,
            "conflicting_rule_count": 1,
        },
        "temporal_update": {
            "difficulty": "medium",
            "update_count": 2,
            "stale_count": 4,
            "evidence_position": "mixed",
        },
    },
    "hard": {
        "anti_distractor_retrieval": {
            "difficulty": "hard",
            "distractor_strength": "adversarial",
            "distractor_count": 12,
            "evidence_position": "random",
        },
        "multi_hop_reasoning": {
            "difficulty": "hard",
            "hop_count": 5,
            "irrelevant_rule_count": 8,
            "conflicting_rule_count": 2,
        },
        "temporal_update": {
            "difficulty": "hard",
            "update_count": 4,
            "stale_count": 8,
            "evidence_position": "mixed",
        },
    },
}


def resolve_generation_kwargs(
    task_family: TaskFamily,
    difficulty: Difficulty,
    generation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if difficulty not in _DEFAULT_PROFILES:
        raise ValueError(f"unsupported_difficulty:{difficulty}")
    if task_family not in _DEFAULT_PROFILES[difficulty]:
        raise ValueError(f"unsupported_task_family:{task_family}")

    kwargs = deepcopy(_DEFAULT_PROFILES[difficulty][task_family])
    overrides = (generation_config or {}).get(difficulty, {}).get(task_family, {})
    if not isinstance(overrides, dict):
        raise ValueError(f"generation override must be a mapping for {difficulty}/{task_family}")
    kwargs.update(overrides)

    allowed = _GENERATOR_KWARGS[task_family]
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise ValueError(f"unsupported_generation_kwargs:{task_family}:{','.join(unknown)}")

    return kwargs
```

- [ ] **Step 4: Run profile tests to verify they pass**

Run:

```bash
python -m pytest tests/benchmark/test_build_pilot.py::test_resolve_generation_kwargs_uses_hard_defaults tests/benchmark/test_build_pilot.py::test_resolve_generation_kwargs_applies_yaml_override -q
```

Expected: PASS.

- [ ] **Step 5: Authorized commit command**

Only after the user explicitly authorizes commits, run:

```bash
git add benchmark/generator/profiles.py tests/benchmark/test_build_pilot.py
git commit -m "feat: add generator difficulty profiles"
```

---

### Task 2: Extend retrieval generator for hard distractors

**Files:**
- Modify: `benchmark/generator/retrieval.py`
- Modify: `tests/benchmark/test_generators.py`

- [ ] **Step 1: Write failing retrieval hard test**

Append this test to `tests/benchmark/test_generators.py`:

```python

def test_retrieval_hard_has_many_distractors():
    task = generate_retrieval_task(
        task_id="vlr_pilot_000010",
        seed=10,
        target_context_tokens=16000,
        difficulty="hard",
        distractor_strength="adversarial",
        distractor_count=12,
        evidence_position="random",
    )

    assert task.difficulty == "hard"
    assert task.metadata.distractor_strength == "adversarial"
    assert task.metadata.evidence_position == "random"
    assert task.metadata.extra["distractor_count"] == 12
    assert len(task.distractor_evidence_ids) == 12
    assert set(task.distractor_evidence_ids).issubset(task.evidence_ids())
    assert set(task.gold_evidence_ids).isdisjoint(task.distractor_evidence_ids)
```

- [ ] **Step 2: Run retrieval hard test to verify it fails**

Run:

```bash
python -m pytest tests/benchmark/test_generators.py::test_retrieval_hard_has_many_distractors -q
```

Expected: FAIL with `TypeError: generate_retrieval_task() got an unexpected keyword argument 'difficulty'`.

- [ ] **Step 3: Replace retrieval generator implementation**

Replace the full contents of `benchmark/generator/retrieval.py` with:

```python
from benchmark.generator.common import doc_id, evidence_id, make_rng, pad_neutral_documents
from benchmark.schemas.task import Difficulty, EvidenceDocument, TaskMetadata, VeriLongTask


_ENTITIES = ["Orion", "Lyra", "Vega", "Atlas", "Nova", "Mira"]
_ATTRIBUTES = ["A17", "C29", "H41", "M63", "Q88", "Z05", "B72", "K19"]
_WINDOWS = ["current verification window", "archive review window", "regional audit window", "legacy intake window"]


def generate_retrieval_task(
    task_id: str,
    seed: int,
    target_context_tokens: int = 8000,
    difficulty: Difficulty = "easy",
    distractor_strength: str = "lexical",
    distractor_count: int = 1,
    evidence_position: str = "front",
) -> VeriLongTask:
    if distractor_count < 1:
        raise ValueError("distractor_count must be at least 1")

    rng = make_rng(seed)
    entity = rng.choice(_ENTITIES)
    answer = rng.choice(_ATTRIBUTES)

    documents = [
        EvidenceDocument(
            doc_id=doc_id(1),
            evidence_id=evidence_id(1),
            text=f"Project {entity} uses access code {answer} for the current verification window.",
            role="gold",
        )
    ]

    for offset in range(distractor_count):
        index = offset + 2
        distractor_answer = rng.choice([attribute for attribute in _ATTRIBUTES if attribute != answer])
        distractor_entity = _distractor_entity(entity=entity, offset=offset, strength=distractor_strength)
        window = _WINDOWS[(offset + 1) % len(_WINDOWS)]
        text = (
            f"Project {distractor_entity} uses access code {distractor_answer} for the {window}. "
            f"This memorandum is not the current verification record for Project {entity}."
        )
        documents.append(
            EvidenceDocument(
                doc_id=doc_id(index),
                evidence_id=evidence_id(index),
                text=text,
                role="distractor",
            )
        )

    if evidence_position in {"mixed", "random"}:
        rng.shuffle(documents)
        documents = _renumber_documents(documents)

    gold_ids = [document.evidence_id for document in documents if document.role == "gold"]
    distractor_ids = [document.evidence_id for document in documents if document.role == "distractor"]
    documents = pad_neutral_documents(documents, target_context_tokens, rng)

    return VeriLongTask(
        id=task_id,
        task_family="anti_distractor_retrieval",
        difficulty=difficulty,
        question=f"Which access code is assigned to Project {entity} for the current verification window?",
        documents=documents,
        gold_answer=answer,
        gold_evidence_ids=gold_ids,
        distractor_evidence_ids=distractor_ids,
        stale_evidence_ids=[],
        expected_steps=[f"Use {gold_ids[0]} to identify Project {entity}'s current access code."],
        metadata=TaskMetadata(
            target_context_tokens=target_context_tokens,
            evidence_position=evidence_position,
            distractor_strength=distractor_strength,
            extra={"distractor_count": distractor_count},
        ),
    )


def _distractor_entity(entity: str, offset: int, strength: str) -> str:
    lexical_forms = [
        f"{entity} Annex",
        f"{entity} Archive",
        f"{entity} East",
        f"{entity} Review",
    ]
    adversarial_forms = [
        f"{entity} Annex",
        f"{entity}-Current",
        f"{entity} Verification Annex",
        f"{entity} Regional",
        f"{entity} Legacy",
        f"{entity} Operations",
    ]
    forms = adversarial_forms if strength == "adversarial" else lexical_forms
    return forms[offset % len(forms)]


def _renumber_documents(documents: list[EvidenceDocument]) -> list[EvidenceDocument]:
    return [
        EvidenceDocument(
            doc_id=doc_id(index),
            evidence_id=evidence_id(index),
            text=document.text,
            role=document.role,
        )
        for index, document in enumerate(documents, start=1)
    ]
```

- [ ] **Step 4: Run retrieval tests**

Run:

```bash
python -m pytest tests/benchmark/test_generators.py::test_retrieval_generator_has_gold_and_distractors tests/benchmark/test_generators.py::test_retrieval_hard_has_many_distractors -q
```

Expected: PASS.

- [ ] **Step 5: Authorized commit command**

Only after the user explicitly authorizes commits, run:

```bash
git add benchmark/generator/retrieval.py tests/benchmark/test_generators.py
git commit -m "feat: generate hard retrieval distractors"
```

---

### Task 3: Extend multihop generator to 2-5 hops

**Files:**
- Modify: `benchmark/generator/multihop.py`
- Modify: `tests/benchmark/test_generators.py`

- [ ] **Step 1: Replace old unsupported-hop test with variable-hop tests**

In `tests/benchmark/test_generators.py`, replace this old test:

```python
@pytest.mark.parametrize("hop_count", [2, 4])
def test_multihop_generator_requires_supported_hop_count(hop_count):
    with pytest.raises(ValueError):
        generate_multihop_task(task_id="vlr_pilot_000004", seed=4, hop_count=hop_count, target_context_tokens=8000)
```

with these tests:

```python
@pytest.mark.parametrize("hop_count", [2, 3, 4, 5])
def test_multihop_generator_supports_two_to_five_hops(hop_count):
    task = generate_multihop_task(
        task_id=f"vlr_pilot_00002{hop_count}",
        seed=20 + hop_count,
        hop_count=hop_count,
        target_context_tokens=8000,
    )

    assert task.task_family == "multi_hop_reasoning"
    assert task.metadata.hop_count == hop_count
    assert len(task.gold_evidence_ids) == hop_count
    assert set(task.gold_evidence_ids).issubset(task.evidence_ids())


def test_multihop_hard_records_irrelevant_and_conflicting_rules():
    task = generate_multihop_task(
        task_id="vlr_pilot_000030",
        seed=30,
        hop_count=5,
        target_context_tokens=16000,
        difficulty="hard",
        irrelevant_rule_count=8,
        conflicting_rule_count=2,
    )

    assert task.difficulty == "hard"
    assert task.metadata.hop_count == 5
    assert task.metadata.extra["irrelevant_rule_count"] == 8
    assert task.metadata.extra["conflicting_rule_count"] == 2
    assert len(task.gold_evidence_ids) == 5
    assert len(task.distractor_evidence_ids) == 10
```

Keep the `import pytest` line because the new test still uses `pytest.mark.parametrize`.

- [ ] **Step 2: Run multihop variable-hop tests to verify they fail**

Run:

```bash
python -m pytest tests/benchmark/test_generators.py::test_multihop_generator_supports_two_to_five_hops tests/benchmark/test_generators.py::test_multihop_hard_records_irrelevant_and_conflicting_rules -q
```

Expected: FAIL because `hop_count=2` and `hop_count=4` currently raise `ValueError`, and hard kwargs are unsupported.

- [ ] **Step 3: Replace multihop generator implementation**

Replace the full contents of `benchmark/generator/multihop.py` with:

```python
from benchmark.generator.common import doc_id, evidence_id, make_rng, pad_neutral_documents
from benchmark.schemas.task import Difficulty, EvidenceDocument, TaskMetadata, VeriLongTask


_ENTITIES = ["Aurora", "Borealis", "Cygnus", "Draco", "Equinox", "Fornax"]
_PROTOCOLS = ["Protocol Blue", "Protocol Green", "Protocol Silver", "Protocol Amber"]
_CONDITIONS = ["low humidity", "night operations", "sealed transit", "manual inspection"]
_APPROVALS = ["Team Delta", "Team Kappa", "Team Meridian", "Team Sol"]
_DESTINATIONS = ["Route 14", "Vault K", "Channel 9", "Bay 27"]


def generate_multihop_task(
    task_id: str,
    seed: int,
    hop_count: int = 3,
    target_context_tokens: int = 8000,
    difficulty: Difficulty = "medium",
    irrelevant_rule_count: int = 1,
    conflicting_rule_count: int = 0,
) -> VeriLongTask:
    if hop_count < 2 or hop_count > 5:
        raise ValueError("hop_count must be between 2 and 5")
    if irrelevant_rule_count < 0:
        raise ValueError("irrelevant_rule_count must be non-negative")
    if conflicting_rule_count < 0:
        raise ValueError("conflicting_rule_count must be non-negative")

    rng = make_rng(seed)
    entity = rng.choice(_ENTITIES)
    protocol = rng.choice(_PROTOCOLS)
    condition = rng.choice(_CONDITIONS)
    approval = rng.choice(_APPROVALS)
    answer = rng.choice(_DESTINATIONS)

    hop_texts = _chain_texts(entity=entity, protocol=protocol, condition=condition, approval=approval, answer=answer)
    documents = [
        EvidenceDocument(
            doc_id=doc_id(index),
            evidence_id=evidence_id(index),
            text=text,
            role="gold",
        )
        for index, text in enumerate(hop_texts[:hop_count], start=1)
    ]

    next_index = len(documents) + 1
    for offset in range(irrelevant_rule_count):
        documents.append(
            EvidenceDocument(
                doc_id=doc_id(next_index),
                evidence_id=evidence_id(next_index),
                text=(
                    f"Facility {entity} Annex follows reference rule {offset + 1}, "
                    f"which routes unrelated audit materials to Holding Bay {offset + 3}."
                ),
                role="distractor",
            )
        )
        next_index += 1

    for offset in range(conflicting_rule_count):
        documents.append(
            EvidenceDocument(
                doc_id=doc_id(next_index),
                evidence_id=evidence_id(next_index),
                text=(
                    f"A retired mapping for {protocol} mentions {rng.choice(_DESTINATIONS)}, "
                    "but the note applies only to deprecated facilities and not the active chain."
                ),
                role="distractor",
            )
        )
        next_index += 1

    documents = pad_neutral_documents(documents, target_context_tokens, rng)

    return VeriLongTask(
        id=task_id,
        task_family="multi_hop_reasoning",
        difficulty=difficulty,
        question=f"Following the active chain for Facility {entity}, what is the required destination?",
        documents=documents,
        gold_answer=answer,
        gold_evidence_ids=[evidence_id(index) for index in range(1, hop_count + 1)],
        distractor_evidence_ids=[evidence_id(index) for index in range(hop_count + 1, next_index)],
        stale_evidence_ids=[],
        expected_steps=_expected_steps(entity=entity, hop_count=hop_count),
        metadata=TaskMetadata(
            target_context_tokens=target_context_tokens,
            evidence_position="distributed",
            hop_count=hop_count,
            extra={
                "irrelevant_rule_count": irrelevant_rule_count,
                "conflicting_rule_count": conflicting_rule_count,
            },
        ),
    )


def _chain_texts(entity: str, protocol: str, condition: str, approval: str, answer: str) -> list[str]:
    return [
        f"Facility {entity} is assigned to {protocol} for the active pilot audit.",
        f"{protocol} applies when the operating condition is {condition}.",
        f"For {condition}, the approval owner is {approval}.",
        f"{approval} uses routing table R7 for this audit family.",
        f"Routing table R7 sends the final package to {answer}.",
    ]


def _expected_steps(entity: str, hop_count: int) -> list[str]:
    steps = [
        f"Find Facility {entity}'s active protocol in {evidence_id(1)}.",
        f"Follow the chain through {evidence_id(2)} to narrow the applicable condition.",
    ]
    if hop_count >= 3:
        steps.append(f"Use {evidence_id(3)} to identify the next owner or rule.")
    if hop_count >= 5:
        steps.append(f"Use {evidence_id(5)} to reach the final destination.")
    return steps[:4]
```

- [ ] **Step 4: Run multihop tests**

Run:

```bash
python -m pytest tests/benchmark/test_generators.py::test_multihop_generator_has_multiple_gold_evidence tests/benchmark/test_generators.py::test_multihop_generator_supports_two_to_five_hops tests/benchmark/test_generators.py::test_multihop_hard_records_irrelevant_and_conflicting_rules -q
```

Expected: PASS.

- [ ] **Step 5: Authorized commit command**

Only after the user explicitly authorizes commits, run:

```bash
git add benchmark/generator/multihop.py tests/benchmark/test_generators.py
git commit -m "feat: support hard multihop chains"
```

---

### Task 4: Extend temporal generator for latest-only hard update chains

**Files:**
- Modify: `benchmark/generator/temporal.py`
- Modify: `tests/benchmark/test_generators.py`

- [ ] **Step 1: Write failing temporal hard test**

Append this test to `tests/benchmark/test_generators.py`:

```python

def test_temporal_hard_has_latest_only_gold_and_many_stale_records():
    task = generate_temporal_task(
        task_id="vlr_pilot_000040",
        seed=40,
        update_count=4,
        stale_count=8,
        target_context_tokens=16000,
        difficulty="hard",
        evidence_position="mixed",
    )

    assert task.difficulty == "hard"
    assert task.metadata.update_count == 4
    assert task.metadata.evidence_position == "mixed"
    assert task.metadata.extra["stale_count"] == 8
    assert task.metadata.extra["latest_update_index"] == 4
    assert len(task.gold_evidence_ids) == 1
    assert len(task.stale_evidence_ids) >= 8
    assert set(task.gold_evidence_ids).isdisjoint(task.stale_evidence_ids)
    gold_document = task.documents_by_evidence_id()[task.gold_evidence_ids[0]]
    assert task.gold_answer in gold_document.text
```

- [ ] **Step 2: Run temporal hard test to verify it fails**

Run:

```bash
python -m pytest tests/benchmark/test_generators.py::test_temporal_hard_has_latest_only_gold_and_many_stale_records -q
```

Expected: FAIL with `TypeError: generate_temporal_task() got an unexpected keyword argument 'stale_count'`.

- [ ] **Step 3: Replace temporal generator implementation**

Replace the full contents of `benchmark/generator/temporal.py` with:

```python
from benchmark.generator.common import doc_id, evidence_id, make_rng, pad_neutral_documents
from benchmark.schemas.task import Difficulty, EvidenceDocument, TaskMetadata, VeriLongTask


_ENTITIES = ["Station Aster", "Station Brindle", "Station Cobalt", "Station Dune"]
_STATUSES = ["pending", "manual review", "standby", "restricted", "approved", "expedited", "cleared", "active"]
_MONTHS = ["January", "February", "March", "April", "May", "June"]


def generate_temporal_task(
    task_id: str,
    seed: int,
    update_count: int = 1,
    target_context_tokens: int = 8000,
    difficulty: Difficulty = "medium",
    stale_count: int = 2,
    evidence_position: str = "mixed",
) -> VeriLongTask:
    if update_count < 1:
        raise ValueError("update_count must be at least 1")
    if stale_count < 1:
        raise ValueError("stale_count must be at least 1")
    if update_count > len(_MONTHS) - 1:
        raise ValueError(f"update_count must be at most {len(_MONTHS) - 1}")

    rng = make_rng(seed)
    entity = rng.choice(_ENTITIES)
    status_sequence = rng.sample(_STATUSES, k=update_count + 1)
    initial_status = status_sequence[0]
    latest_status = status_sequence[-1]

    documents: list[EvidenceDocument] = [
        EvidenceDocument(
            doc_id=doc_id(1),
            evidence_id=evidence_id(1),
            text=f"{_MONTHS[0]} checklist: {entity} status is {initial_status} for the initial review cycle.",
            role="stale",
        )
    ]

    for update_index in range(1, update_count + 1):
        status = status_sequence[update_index]
        role = "gold" if update_index == update_count else "stale"
        documents.append(
            EvidenceDocument(
                doc_id=doc_id(len(documents) + 1),
                evidence_id=evidence_id(len(documents) + 1),
                text=(
                    f"{_MONTHS[update_index]} update {update_index}: {entity} status is now {status}; "
                    "this update supersedes earlier records."
                ),
                role=role,
            )
        )

    while len([document for document in documents if document.role == "stale"]) < stale_count:
        stale_index = len([document for document in documents if document.role == "stale"]) + 1
        stale_status = status_sequence[stale_index % len(status_sequence[:-1])]
        documents.append(
            EvidenceDocument(
                doc_id=doc_id(len(documents) + 1),
                evidence_id=evidence_id(len(documents) + 1),
                text=(
                    f"Legacy checklist copy {stale_index}: {entity} status is listed as {stale_status}, "
                    "but the copy predates the latest update."
                ),
                role="stale",
            )
        )

    documents.append(
        EvidenceDocument(
            doc_id=doc_id(len(documents) + 1),
            evidence_id=evidence_id(len(documents) + 1),
            text=f"A similarly named site, {entity} Annex, uses an unrelated status label for inventory only.",
            role="distractor",
        )
    )

    if evidence_position in {"mixed", "random"}:
        rng.shuffle(documents)
        documents = _renumber_documents(documents)

    gold_ids = [document.evidence_id for document in documents if document.role == "gold"]
    stale_ids = [document.evidence_id for document in documents if document.role == "stale"]
    distractor_ids = [document.evidence_id for document in documents if document.role == "distractor"]
    documents = pad_neutral_documents(documents, target_context_tokens, rng)

    return VeriLongTask(
        id=task_id,
        task_family="temporal_update",
        difficulty=difficulty,
        question=f"What is the current status of {entity}?",
        documents=documents,
        gold_answer=latest_status,
        gold_evidence_ids=gold_ids,
        distractor_evidence_ids=distractor_ids,
        stale_evidence_ids=stale_ids,
        expected_steps=[
            "Identify earlier status records as superseded evidence.",
            f"Use the latest update evidence in {gold_ids[0]} for the current status.",
            "Ignore stale checklist evidence that predates the latest update.",
        ],
        metadata=TaskMetadata(
            target_context_tokens=target_context_tokens,
            evidence_position=evidence_position,
            update_count=update_count,
            extra={"stale_count": stale_count, "latest_update_index": update_count},
        ),
    )


def _renumber_documents(documents: list[EvidenceDocument]) -> list[EvidenceDocument]:
    return [
        EvidenceDocument(
            doc_id=doc_id(index),
            evidence_id=evidence_id(index),
            text=document.text,
            role=document.role,
        )
        for index, document in enumerate(documents, start=1)
    ]
```

- [ ] **Step 4: Run temporal tests**

Run:

```bash
python -m pytest tests/benchmark/test_generators.py::test_temporal_generator_marks_stale_evidence tests/benchmark/test_generators.py::test_temporal_hard_has_latest_only_gold_and_many_stale_records -q
```

Expected: PASS.

- [ ] **Step 5: Authorized commit command**

Only after the user explicitly authorizes commits, run:

```bash
git add benchmark/generator/temporal.py tests/benchmark/test_generators.py
git commit -m "feat: generate hard temporal update chains"
```

---

### Task 5: Use difficulty distribution and profiles in pilot builder

**Files:**
- Modify: `benchmark/generator/build_pilot.py`
- Modify: `tests/benchmark/test_build_pilot.py`

- [ ] **Step 1: Add failing difficulty-distribution tests**

Append these tests to `tests/benchmark/test_build_pilot.py`:

```python

def test_generate_tasks_uses_difficulty_distribution():
    config = _small_config()
    config["difficulty"] = {"easy": 0.2, "medium": 0.3, "hard": 0.5}

    tasks = _generate_tasks(config)
    counts = collections.Counter(task.difficulty for task in tasks)

    assert counts["easy"] == 18
    assert counts["medium"] == 27
    assert counts["hard"] == 45


def test_generate_tasks_applies_hard_generation_overrides():
    config = {
        "seed": 7,
        "size": 9,
        "task_mix": {
            "anti_distractor_retrieval": 3,
            "multi_hop_reasoning": 3,
            "temporal_update": 3,
        },
        "target_context_tokens": [8000],
        "difficulty": {"hard": 1.0},
        "splits": {"train": 0.7, "dev": 0.1, "test": 0.2},
        "extra_splits": {"judge_subset_size": 0, "live_demo_subset_size": 0},
        "generation": {
            "hard": {
                "anti_distractor_retrieval": {"distractor_count": 9},
                "multi_hop_reasoning": {"hop_count": 4, "irrelevant_rule_count": 6, "conflicting_rule_count": 1},
                "temporal_update": {"update_count": 3, "stale_count": 7},
            }
        },
    }

    tasks = _generate_tasks(config)
    by_family = {task.task_family: task for task in tasks[:3]}

    assert {task.difficulty for task in tasks} == {"hard"}
    assert by_family["anti_distractor_retrieval"].metadata.extra["distractor_count"] == 9
    assert by_family["multi_hop_reasoning"].metadata.hop_count == 4
    assert by_family["multi_hop_reasoning"].metadata.extra["irrelevant_rule_count"] == 6
    assert by_family["temporal_update"].metadata.update_count == 3
    assert by_family["temporal_update"].metadata.extra["stale_count"] == 7
```

- [ ] **Step 2: Run builder difficulty tests to verify they fail**

Run:

```bash
python -m pytest tests/benchmark/test_build_pilot.py::test_generate_tasks_uses_difficulty_distribution tests/benchmark/test_build_pilot.py::test_generate_tasks_applies_hard_generation_overrides -q
```

Expected: FAIL because `_generate_tasks` currently ignores `difficulty` and `generation` config.

- [ ] **Step 3: Modify imports in `build_pilot.py`**

In `benchmark/generator/build_pilot.py`, add this import below the existing generator imports:

```python
from benchmark.generator.profiles import resolve_generation_kwargs
```

- [ ] **Step 4: Replace `_generate_tasks` with difficulty-aware implementation**

Replace the existing `_generate_tasks` function in `benchmark/generator/build_pilot.py` with:

```python
def _generate_tasks(config: dict[str, Any]) -> list[VeriLongTask]:
    task_mix = config["task_mix"]
    expected_size = config.get("size")
    total_size = sum(int(count) for count in task_mix.values())
    if expected_size is not None and int(expected_size) != total_size:
        raise ValueError(f"size {expected_size} does not match task_mix total {total_size}")

    tasks: list[VeriLongTask] = []
    task_index = 1
    seed = int(config["seed"])
    target_context_tokens = list(config["target_context_tokens"])
    difficulties = _difficulty_sequence(config.get("difficulty", {"medium": 1.0}), total_size)
    generation_config = config.get("generation", {})

    for task_family, count in task_mix.items():
        if task_family not in _GENERATORS:
            raise ValueError(f"unsupported_task_family:{task_family}")
        for family_index in range(int(count)):
            task_id = f"vlr_pilot_{task_index:06d}"
            target_tokens = int(target_context_tokens[(task_index - 1) % len(target_context_tokens)])
            task_seed = seed + task_index
            difficulty = difficulties[task_index - 1]
            generator = _GENERATORS[task_family]
            generation_kwargs = resolve_generation_kwargs(
                task_family=task_family,
                difficulty=difficulty,
                generation_config=generation_config,
            )
            task = generator(
                task_id=task_id,
                seed=task_seed,
                target_context_tokens=target_tokens,
                **generation_kwargs,
            )
            tasks.append(task)
            task_index += 1

    _assign_splits(tasks, config["splits"])
    _assign_extra_splits(tasks, config.get("extra_splits", {}))
    return tasks
```

- [ ] **Step 5: Add `_difficulty_sequence` helper**

Add this helper below `_generate_tasks` in `benchmark/generator/build_pilot.py`:

```python
def _difficulty_sequence(difficulty_mix: dict[str, float], total_size: int) -> list[str]:
    if total_size <= 0:
        return []
    if not difficulty_mix:
        raise ValueError("difficulty mix must not be empty")

    items = list(difficulty_mix.items())
    counts: list[tuple[str, int]] = []
    assigned = 0
    for index, (difficulty, proportion) in enumerate(items):
        if difficulty not in {"easy", "medium", "hard"}:
            raise ValueError(f"unsupported_difficulty:{difficulty}")
        if index == len(items) - 1:
            count = total_size - assigned
        else:
            count = int(total_size * float(proportion))
        counts.append((difficulty, count))
        assigned += count

    sequence: list[str] = []
    for difficulty, count in counts:
        sequence.extend([difficulty] * count)
    if len(sequence) != total_size:
        raise ValueError(f"difficulty sequence length {len(sequence)} does not match total {total_size}")
    return sequence
```

- [ ] **Step 6: Run builder tests**

Run:

```bash
python -m pytest tests/benchmark/test_build_pilot.py -q
```

Expected: PASS.

- [ ] **Step 7: Authorized commit command**

Only after the user explicitly authorizes commits, run:

```bash
git add benchmark/generator/build_pilot.py tests/benchmark/test_build_pilot.py
git commit -m "feat: apply difficulty profiles in pilot builder"
```

---

### Task 6: Add hard pilot config

**Files:**
- Create: `configs/hard.yaml`
- Modify: `tests/benchmark/test_build_pilot.py`

- [ ] **Step 1: Write failing hard config load test**

Append this test to `tests/benchmark/test_build_pilot.py`:

```python
from pathlib import Path

from benchmark.generator.build_pilot import load_config


def test_hard_config_is_all_hard_and_balanced():
    config = load_config(Path("configs/hard.yaml"))

    assert config["size"] == 180
    assert config["task_mix"] == {
        "anti_distractor_retrieval": 60,
        "multi_hop_reasoning": 60,
        "temporal_update": 60,
    }
    assert config["target_context_tokens"] == [16000, 32000]
    assert config["difficulty"] == {"hard": 1.0}
    assert config["generation"]["hard"]["anti_distractor_retrieval"]["distractor_count"] == 12
    assert config["generation"]["hard"]["multi_hop_reasoning"]["hop_count"] == 5
    assert config["generation"]["hard"]["temporal_update"]["update_count"] == 4
```

- [ ] **Step 2: Run hard config test to verify it fails**

Run:

```bash
python -m pytest tests/benchmark/test_build_pilot.py::test_hard_config_is_all_hard_and_balanced -q
```

Expected: FAIL with `FileNotFoundError` for `configs/hard.yaml`.

- [ ] **Step 3: Create `configs/hard.yaml`**

Create `configs/hard.yaml` with this content:

```yaml
seed: 20260627
output_path: data/pilot/hard_tasks.jsonl
size: 180
task_mix:
  anti_distractor_retrieval: 60
  multi_hop_reasoning: 60
  temporal_update: 60
target_context_tokens:
  - 16000
  - 32000
difficulty:
  hard: 1.0
splits:
  train: 0.70
  dev: 0.10
  test: 0.20
extra_splits:
  judge_subset_size: 30
  live_demo_subset_size: 15
generation:
  hard:
    anti_distractor_retrieval:
      distractor_count: 12
      distractor_strength: adversarial
      evidence_position: random
    multi_hop_reasoning:
      hop_count: 5
      irrelevant_rule_count: 8
      conflicting_rule_count: 2
    temporal_update:
      update_count: 4
      stale_count: 8
      evidence_position: mixed
```

- [ ] **Step 4: Run hard config test**

Run:

```bash
python -m pytest tests/benchmark/test_build_pilot.py::test_hard_config_is_all_hard_and_balanced -q
```

Expected: PASS.

- [ ] **Step 5: Run full builder tests**

Run:

```bash
python -m pytest tests/benchmark/test_build_pilot.py -q
```

Expected: PASS.

- [ ] **Step 6: Authorized commit command**

Only after the user explicitly authorizes commits, run:

```bash
git add configs/hard.yaml tests/benchmark/test_build_pilot.py
git commit -m "feat: add hard pilot generation config"
```

---

### Task 7: Generate and validate pilot plus hard data locally

**Files:**
- Generated/modified by commands: `data/pilot/tasks.jsonl`
- Generated by commands: `data/pilot/hard_tasks.jsonl`

- [ ] **Step 1: Run all benchmark tests**

Run:

```bash
python -m pytest -q
```

Expected: all tests pass. Current baseline before this plan was 39 passing; after this plan the count should be higher because new tests were added.

- [ ] **Step 2: Regenerate standard pilot**

Run:

```bash
python scripts/generate_pilot.py --config configs/pilot.yaml
```

Expected:

```text
generated=1200 valid=1200 output=data/pilot/tasks.jsonl
```

- [ ] **Step 3: Validate standard pilot**

Run:

```bash
python scripts/validate_pilot.py data/pilot/tasks.jsonl
```

Expected:

```text
validated=1200 valid=1200 invalid=0
```

- [ ] **Step 4: Generate hard pilot slice**

Run:

```bash
python scripts/generate_pilot.py --config configs/hard.yaml
```

Expected:

```text
generated=180 valid=180 output=data/pilot/hard_tasks.jsonl
```

- [ ] **Step 5: Validate hard pilot slice**

Run:

```bash
python scripts/validate_pilot.py data/pilot/hard_tasks.jsonl
```

Expected:

```text
validated=180 valid=180 invalid=0
```

- [ ] **Step 6: Inspect generated data distribution with Python**

Run:

```bash
python - <<'PY'
import collections, json
from pathlib import Path

path = Path('data/pilot/hard_tasks.jsonl')
rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
print('count', len(rows))
print('families', dict(collections.Counter(row['task_family'] for row in rows)))
print('difficulty', dict(collections.Counter(row['difficulty'] for row in rows)))
print('splits', dict(collections.Counter(row['metadata']['split'] for row in rows)))
print('contexts', dict(collections.Counter(row['metadata']['target_context_tokens'] for row in rows)))
print('retrieval_distractors', sorted({row['metadata']['extra'].get('distractor_count') for row in rows if row['task_family'] == 'anti_distractor_retrieval'}))
print('multihop_hops', sorted({row['metadata'].get('hop_count') for row in rows if row['task_family'] == 'multi_hop_reasoning'}))
print('temporal_updates', sorted({row['metadata'].get('update_count') for row in rows if row['task_family'] == 'temporal_update'}))
PY
```

Expected output includes:

```text
count 180
families {'anti_distractor_retrieval': 60, 'multi_hop_reasoning': 60, 'temporal_update': 60}
difficulty {'hard': 180}
retrieval_distractors [12]
multihop_hops [5]
temporal_updates [4]
```

The order of dictionary keys may differ.

- [ ] **Step 7: Authorized commit command**

Only after the user explicitly authorizes commits, run:

```bash
git add data/pilot/tasks.jsonl data/pilot/hard_tasks.jsonl
git commit -m "data: regenerate pilot and add hard slice"
```

If generated data files are intentionally untracked in this repository state, do not force-add them; report the generated paths and validation output instead.

---

### Task 8: Optional API hard-dev evaluation gate

**Files:**
- Generated only after explicit user approval: `results/raw/api/gemini_flash_lite_hard_dev30.jsonl`
- Generated only after explicit user approval: `results/raw/api/gemini_pro_hard_dev30.jsonl`

This task is a manual gate. Do not execute it unless the user explicitly approves spending API calls.

- [ ] **Step 1: Confirm API run approval**

Ask the user:

```text
Hard data generation is complete. Do you approve running Gemini flash-lite/pro on hard dev30 using the existing OpenAI-compatible API runner?
```

Expected: explicit user approval before proceeding.

- [ ] **Step 2: Run flash-lite hard dev30**

After approval, run:

```bash
python experiments/eval_api/run_api_eval.py --tasks data/pilot/hard_tasks.jsonl --split dev --limit 30 --stratify --provider openai-compatible --model a/gemini-3.1-flash-lite --out results/raw/api/gemini_flash_lite_hard_dev30.jsonl
```

Expected: command completes and writes `results/raw/api/gemini_flash_lite_hard_dev30.jsonl`. If the runner prints per-task failures, preserve the output and report exact failures.

- [ ] **Step 3: Run pro hard dev30**

After approval, run:

```bash
python experiments/eval_api/run_api_eval.py --tasks data/pilot/hard_tasks.jsonl --split dev --limit 30 --stratify --provider openai-compatible --model a/gemini-3.1-pro --out results/raw/api/gemini_pro_hard_dev30.jsonl
```

Expected: command completes and writes `results/raw/api/gemini_pro_hard_dev30.jsonl`. If the runner prints per-task failures, preserve the output and report exact failures.

- [ ] **Step 4: Score hard outputs**

Run the existing scoring command used by the repository for API outputs. If the repository's scoring entrypoint is `scripts/score_outputs.py`, use:

```bash
python scripts/score_outputs.py --tasks data/pilot/hard_tasks.jsonl --outputs results/raw/api/gemini_flash_lite_hard_dev30.jsonl --out-dir results/raw/api/gemini_flash_lite_hard_dev30_scored
python scripts/score_outputs.py --tasks data/pilot/hard_tasks.jsonl --outputs results/raw/api/gemini_pro_hard_dev30.jsonl --out-dir results/raw/api/gemini_pro_hard_dev30_scored
```

Expected: scored JSONL and summary JSON files are written under the two output directories. If the local scoring CLI uses different flags, stop and report the CLI help output before changing commands.

- [ ] **Step 5: Report real metrics only**

Report only values read from the generated summary JSON files. The report must include:

```text
model
sample_count
overall reward_total_mean
per-family reward_total_mean
citation_precision_mean
stale_citation_rate_mean
format_valid_rate
```

Do not fill missing values from memory or expectation.

---

## 3. Final verification checklist

After Tasks 1-7 are implemented, run:

```bash
python -m pytest -q
python scripts/generate_pilot.py --config configs/pilot.yaml
python scripts/validate_pilot.py data/pilot/tasks.jsonl
python scripts/generate_pilot.py --config configs/hard.yaml
python scripts/validate_pilot.py data/pilot/hard_tasks.jsonl
```

Expected evidence:

```text
All tests pass.
generated=1200 valid=1200 output=data/pilot/tasks.jsonl
validated=1200 valid=1200 invalid=0
generated=180 valid=180 output=data/pilot/hard_tasks.jsonl
validated=180 valid=180 invalid=0
```

Do not claim the hard system is complete unless these commands have passed or their exact failure output has been reported.

---

## 4. Self-review

### Spec coverage

- Config-driven difficulty profiles are implemented by Task 1 and Task 5.
- Hard retrieval distractor count and adversarial similarity are implemented by Task 2.
- Multi-hop 4/5-hop chains are implemented by Task 3.
- Temporal update chains with latest-only gold and stale penalties are implemented by Task 4.
- `configs/hard.yaml` with 180 all-hard tasks and 16K/32K contexts is implemented by Task 6.
- Stratified split preservation is covered by existing split tests plus Task 5 builder tests.
- Local generation and validation are covered by Task 7.
- API evaluation remains an explicit user-approved gate in Task 8.

### Placeholder scan

This plan contains no unresolved implementation placeholders. The optional API gate names exact commands and requires stopping if the scoring CLI differs rather than inventing metrics.

### Type consistency

- `Difficulty` values remain `easy`, `medium`, and `hard`.
- `TaskFamily` values remain `anti_distractor_retrieval`, `multi_hop_reasoning`, and `temporal_update`.
- Generator kwargs returned by `resolve_generation_kwargs()` match the signatures introduced in Tasks 2-4.
- `metadata.extra` keys are consistently named `distractor_count`, `irrelevant_rule_count`, `conflicting_rule_count`, `stale_count`, and `latest_update_index`.
- `target_context_tokens`, `evidence_position`, `distractor_strength`, `hop_count`, and `update_count` use the existing `TaskMetadata` fields.
