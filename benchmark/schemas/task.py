from typing import Any, Literal

from pydantic import BaseModel, Field


TaskFamily = Literal["anti_distractor_retrieval", "multi_hop_reasoning", "temporal_update"]
EvidenceRole = Literal["gold", "distractor", "stale", "neutral"]
Difficulty = Literal["easy", "medium", "hard"]


class EvidenceDocument(BaseModel):
    doc_id: str
    evidence_id: str
    text: str
    role: EvidenceRole


class TaskMetadata(BaseModel):
    target_context_tokens: int
    evidence_position: str | None = None
    distractor_strength: str | None = None
    hop_count: int | None = None
    update_count: int | None = None
    split: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class VeriLongTask(BaseModel):
    id: str
    task_family: TaskFamily
    difficulty: Difficulty
    question: str
    documents: list[EvidenceDocument]
    gold_answer: str
    gold_evidence_ids: list[str]
    distractor_evidence_ids: list[str] = Field(default_factory=list)
    stale_evidence_ids: list[str] = Field(default_factory=list)
    expected_steps: list[str]
    metadata: TaskMetadata

    def evidence_ids(self) -> set[str]:
        return {document.evidence_id for document in self.documents}

    def documents_by_evidence_id(self) -> dict[str, EvidenceDocument]:
        return {document.evidence_id: document for document in self.documents}

    def gold_documents(self) -> list[EvidenceDocument]:
        return [document for document in self.documents if document.role == "gold"]

    def step_bounds(self) -> tuple[int, int]:
        """Allowed reasoning-step count for this task, derived from structure.

        A global 2-4 bound mis-judges single-hop retrieval, where one grounded
        step is a complete rationale. Bounds are derived from task structure so
        they stay correct as the dataset scales:

        - anti_distractor_retrieval: 1-2 (single-hop; one step is valid).
        - multi_hop_reasoning: 2 .. hop_count + 2 (must combine, but a model may
          legitimately express several hops in one sentence; whether all hops
          are actually used is judged by citation recall, not step count).
        - temporal_update: 2 .. update_count + 2 (recognize + apply updates).
        """

        if self.task_family == "anti_distractor_retrieval":
            return (1, 2)
        if self.task_family == "multi_hop_reasoning":
            hops = self.metadata.hop_count or len(self.gold_evidence_ids) or 2
            hops = max(hops, 2)
            return (2, hops + 2)
        if self.task_family == "temporal_update":
            updates = self.metadata.update_count or 1
            return (2, updates + 2)
        # Fallback for any future family: keep the original conservative band.
        return (2, 4)

