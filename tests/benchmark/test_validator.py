from benchmark.generator.retrieval import generate_retrieval_task
from benchmark.generator.temporal import generate_temporal_task
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


def test_distractor_evidence_must_have_distractor_or_stale_role():
    task = generate_retrieval_task(task_id="vlr_pilot_000001", seed=11, target_context_tokens=8000)
    task.distractor_evidence_ids = [task.gold_evidence_ids[0]]
    report = validate_task(task)
    assert report.valid is False
    assert "invalid_distractor_evidence_role:E01:gold" in report.errors


def test_stale_evidence_must_have_stale_role():
    task = generate_temporal_task(task_id="vlr_pilot_000002", seed=12, update_count=1, target_context_tokens=8000)
    task.stale_evidence_ids = [task.gold_evidence_ids[0]]
    report = validate_task(task)
    assert report.valid is False
    assert "invalid_stale_evidence_role:E02:gold" in report.errors


def test_duplicate_document_id_fails_validation():
    task = generate_retrieval_task(task_id="vlr_pilot_000001", seed=11, target_context_tokens=8000)
    task.documents[1].doc_id = task.documents[0].doc_id
    report = validate_task(task)
    assert report.valid is False
    assert "duplicate_doc_id" in report.errors
