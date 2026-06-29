import pytest

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

    gold_doc = task.documents_by_evidence_id()[task.gold_evidence_ids[0]]
    assert task.gold_answer in gold_doc.text
    assert task.gold_evidence_ids[0] in task.expected_steps[0]

    for did in task.distractor_evidence_ids:
        assert "current verification window" not in task.documents_by_evidence_id()[did].text


def test_multihop_generator_has_multiple_gold_evidence():
    task = generate_multihop_task(task_id="vlr_pilot_000002", seed=2, hop_count=3, target_context_tokens=8000)
    assert task.task_family == "multi_hop_reasoning"
    assert_common_task_invariants(task)
    assert len(task.gold_evidence_ids) == 3


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

    # The answer must be derivable from gold evidence: the final hop names it.
    final_gold_doc = task.documents_by_evidence_id()[task.gold_evidence_ids[-1]]
    assert task.gold_answer in final_gold_doc.text
    # No earlier gold hop should already give away the destination.
    for evidence_id in task.gold_evidence_ids[:-1]:
        assert task.gold_answer not in task.documents_by_evidence_id()[evidence_id].text


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

    # Conflicting distractors must not name the true destination.
    for evidence_id in task.distractor_evidence_ids:
        assert task.gold_answer not in task.documents_by_evidence_id()[evidence_id].text


def test_temporal_generator_marks_stale_evidence():
    task = generate_temporal_task(task_id="vlr_pilot_000003", seed=3, update_count=1, target_context_tokens=8000)
    assert task.task_family == "temporal_update"
    assert_common_task_invariants(task)
    assert len(task.stale_evidence_ids) >= 1


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
    # The latest answer must not leak into any stale record.
    for evidence_id in task.stale_evidence_ids:
        assert task.gold_answer not in task.documents_by_evidence_id()[evidence_id].text
