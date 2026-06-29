from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SmokeSummary(BaseModel):
    label: str
    baseline: str
    count: int
    reward_total_mean: float
    answer_exact_match_mean: float
    citation_f1_mean: float
    overcitation_rate_mean: float
    note: str = "Smoke baseline only; not a live model leaderboard."


class TaskFamilySummary(BaseModel):
    id: str
    label: str
    description: str
    signal: str


class ProjectStatus(BaseModel):
    phase1_pilot: Literal["completed"] = "completed"
    hard_difficulty: Literal["completed"] = "completed"
    sft_warmup: Literal["completed"] = "completed"
    rlvr_pipeline: Literal["validated"] = "validated"
    rlvr_full_run: Literal["deferred"] = "deferred"
    phase2: Literal["design_only"] = "design_only"


class SummaryResponse(BaseModel):
    project: str
    tagline: str
    status: ProjectStatus
    output_format: list[str]
    task_families: list[TaskFamilySummary]
    smoke_summaries: list[SmokeSummary]


class DemoDocument(BaseModel):
    doc_id: str
    evidence_id: str
    text: str
    role: str


class DemoCaseSummary(BaseModel):
    task_id: str
    task_family: str
    difficulty: str
    question: str
    model: str
    reward_total: float | None = None
    error_type: str | None = None


class DemoCase(BaseModel):
    task_id: str
    task_family: str
    difficulty: str
    question: str
    documents: list[DemoDocument]
    gold_answer: str
    gold_evidence_ids: list[str]
    distractor_evidence_ids: list[str] = Field(default_factory=list)
    stale_evidence_ids: list[str] = Field(default_factory=list)
    model: str
    model_output: str
    parsed_output: dict[str, Any]
    metric_breakdown: dict[str, Any]
    error_type: str | None = None
    prompt_preview: str


class DryRunRequest(BaseModel):
    task_id: str


class DemoRunResponse(BaseModel):
    task_id: str
    model: str
    source: Literal["dry_run"]
    output_text: str
    parsed_output: dict[str, Any]
    metric_breakdown: dict[str, Any]
    error_type: str | None = None
    prompt_preview: str
