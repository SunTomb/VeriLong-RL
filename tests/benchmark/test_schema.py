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
