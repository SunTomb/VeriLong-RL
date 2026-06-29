from pydantic import BaseModel, Field


class ParsedPrediction(BaseModel):
    pred_answer: str | None = None
    pred_evidence_ids: list[str] = Field(default_factory=list)
    pred_steps: list[str] = Field(default_factory=list)
    format_valid: bool = False
    unparsed_text: str = ""
    error_flags: list[str] = Field(default_factory=list)
