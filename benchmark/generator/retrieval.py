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
    if distractor_strength not in {"lexical", "semantic", "adversarial"}:
        raise ValueError(f"unsupported_distractor_strength:{distractor_strength}")
    if evidence_position not in {"front", "mixed", "random"}:
        raise ValueError(f"unsupported_evidence_position:{evidence_position}")

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
        # Never reuse index 0 ("current verification window") — that is the gold
        # phrase echoed in the question, so distractors must avoid it.
        window = _WINDOWS[1 + (offset % (len(_WINDOWS) - 1))]
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
        # Both "mixed" and "random" randomize gold position; only "front" pins
        # gold first. Position difficulty comes from distractor_count/strength,
        # not a separate placement algorithm.
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
    forms = lexical_forms if strength == "lexical" else adversarial_forms
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
