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
    if evidence_position not in {"front", "mixed", "random"}:
        raise ValueError(f"unsupported_evidence_position:{evidence_position}")

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
        copy_index = len([document for document in documents if document.role == "stale"])
        stale_status = status_sequence[copy_index % len(status_sequence[:-1])]
        documents.append(
            EvidenceDocument(
                doc_id=doc_id(len(documents) + 1),
                evidence_id=evidence_id(len(documents) + 1),
                text=(
                    f"Legacy checklist copy {copy_index}: {entity} status is listed as {stale_status}, "
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
        # Both "mixed" and "random" shuffle so the latest update is not pinned by
        # position; only "front" keeps the initial record first.
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
