from dataclasses import dataclass, field

from benchmark.schemas.task import VeriLongTask


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    errors: list[str] = field(default_factory=list)


def validate_task(task: VeriLongTask) -> ValidationReport:
    errors: list[str] = []

    if not task.id:
        errors.append("empty_id")

    documents_by_evidence_id = task.documents_by_evidence_id()
    if len(documents_by_evidence_id) != len(task.documents):
        errors.append("duplicate_evidence_id")

    doc_ids = {document.doc_id for document in task.documents}
    if len(doc_ids) != len(task.documents):
        errors.append("duplicate_doc_id")

    evidence_ids = set(documents_by_evidence_id)
    gold_ids = set(task.gold_evidence_ids)
    distractor_ids = set(task.distractor_evidence_ids)
    stale_ids = set(task.stale_evidence_ids)

    for evidence_id in task.gold_evidence_ids:
        document = documents_by_evidence_id.get(evidence_id)
        if document is None or document.role != "gold":
            errors.append(f"missing_gold_evidence_id:{evidence_id}")

    for evidence_id in task.distractor_evidence_ids:
        document = documents_by_evidence_id.get(evidence_id)
        if document is None:
            errors.append(f"missing_distractor_evidence_id:{evidence_id}")
        elif document.role not in {"distractor", "stale"}:
            errors.append(f"invalid_distractor_evidence_role:{evidence_id}:{document.role}")

    for evidence_id in task.stale_evidence_ids:
        document = documents_by_evidence_id.get(evidence_id)
        if document is None:
            errors.append(f"missing_stale_evidence_id:{evidence_id}")
        elif document.role != "stale":
            errors.append(f"invalid_stale_evidence_role:{evidence_id}:{document.role}")

    for evidence_id in sorted(gold_ids & distractor_ids):
        errors.append(f"distractor_evidence_overlaps_gold:{evidence_id}")

    for evidence_id in sorted(gold_ids & stale_ids):
        errors.append(f"stale_evidence_overlaps_gold:{evidence_id}")

    if not task.gold_answer:
        errors.append("empty_gold_answer")

    if not 1 <= len(task.expected_steps) <= 4:
        errors.append("invalid_expected_steps_length")

    if task.task_family == "anti_distractor_retrieval":
        if task.metadata.evidence_position is None:
            errors.append("missing_metadata:evidence_position")
        if task.metadata.distractor_strength is None:
            errors.append("missing_metadata:distractor_strength")
    elif task.task_family == "multi_hop_reasoning":
        if task.metadata.hop_count is None:
            errors.append("missing_metadata:hop_count")
    elif task.task_family == "temporal_update":
        if task.metadata.update_count is None:
            errors.append("missing_metadata:update_count")

    return ValidationReport(valid=not errors, errors=errors)

