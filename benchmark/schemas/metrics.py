from pydantic import BaseModel, Field


class CitationScores(BaseModel):
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    all_gold_evidence_recall: float = 0.0
    distractor_citation_rate: float = 0.0
    stale_citation_rate: float = 0.0
    invalid_citation_rate: float = 0.0
    overcitation_rate: float = 0.0


class SampleScores(BaseModel):
    task_id: str
    model: str
    answer_exact_match: float
    answer_normalized_match: float
    format_valid: bool
    citation: CitationScores = Field(default_factory=CitationScores)
    reward_total: float | None = None
    error_type: str | None = None
