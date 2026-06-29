from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from benchmark.schemas.task import EvidenceDocument, TaskMetadata, VeriLongTask
from experiments.eval_api.run_api_eval import build_user_prompt
from web.backend.schemas.demo import (
    DemoCase,
    DemoCaseSummary,
    DemoDocument,
    ProjectStatus,
    SmokeSummary,
    SummaryResponse,
    TaskFamilySummary,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
ORACLE_DIR = REPO_ROOT / "results" / "pilot" / "oracle_smoke"
CORRUPTED_DIR = REPO_ROOT / "results" / "pilot" / "corrupted_smoke"

TASK_FAMILIES = [
    TaskFamilySummary(
        id="anti_distractor_retrieval",
        label="Anti-distractor retrieval",
        description="Find the one supporting record among lexically similar distractors and neutral padding.",
        signal="Citation precision and distractor-citation penalties expose over-citation.",
    ),
    TaskFamilySummary(
        id="multi_hop_reasoning",
        label="Multi-hop reasoning",
        description="Combine multiple evidence records to derive the final answer.",
        signal="Gold-evidence recall and step validity measure whether all hops are used.",
    ),
    TaskFamilySummary(
        id="temporal_update",
        label="Temporal update",
        description="Apply the latest update while ignoring stale records and legacy copies.",
        signal="Stale-citation penalties separate current evidence from outdated evidence.",
    ),
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_summary() -> SummaryResponse:
    return SummaryResponse(
        project="VeriLong-RL",
        tagline="A verifiable long-context benchmark for evidence-grounded reasoning and RLVR.",
        status=ProjectStatus(),
        output_format=["Evidence: E01, E02", "Steps:", "1. Grounded reasoning step.", "Answer: final answer only"],
        task_families=TASK_FAMILIES,
        smoke_summaries=[
            _load_smoke_summary("Oracle smoke", ORACLE_DIR / "summary.json"),
            _load_smoke_summary("Corrupted smoke", CORRUPTED_DIR / "summary.json"),
        ],
    )


def _load_smoke_summary(label: str, path: Path) -> SmokeSummary:
    raw = _read_json(path)
    return SmokeSummary(
        label=label,
        baseline=str(raw["baseline"]),
        count=int(raw["count"]),
        reward_total_mean=float(raw["reward_total_mean"]),
        answer_exact_match_mean=float(raw["answer_exact_match_mean"]),
        citation_f1_mean=float(raw["citation_f1_mean"]),
        overcitation_rate_mean=float(raw["overcitation_rate_mean"]),
    )


@lru_cache(maxsize=1)
def load_cases() -> list[DemoCase]:
    cases = [_case_from_raw(raw) for raw in _read_json(ORACLE_DIR / "cases_for_demo.json")]
    return cases[:24]


def list_case_summaries() -> list[DemoCaseSummary]:
    summaries: list[DemoCaseSummary] = []
    for case in load_cases():
        summaries.append(
            DemoCaseSummary(
                task_id=case.task_id,
                task_family=case.task_family,
                difficulty=case.difficulty,
                question=case.question,
                model=case.model,
                reward_total=_metric_as_float(case.metric_breakdown.get("reward_total")),
                error_type=case.error_type,
            )
        )
    return summaries


def get_case(task_id: str) -> DemoCase | None:
    for case in load_cases():
        if case.task_id == task_id:
            return case
    return None


def case_to_task(case: DemoCase) -> VeriLongTask:
    return VeriLongTask(
        id=case.task_id,
        task_family=case.task_family,
        difficulty=case.difficulty,
        question=case.question,
        documents=[EvidenceDocument(**document.model_dump()) for document in case.documents],
        gold_answer=case.gold_answer,
        gold_evidence_ids=case.gold_evidence_ids,
        distractor_evidence_ids=case.distractor_evidence_ids,
        stale_evidence_ids=case.stale_evidence_ids,
        expected_steps=_expected_steps_from_case(case),
        metadata=TaskMetadata(
            target_context_tokens=_estimate_target_context_tokens(case),
            split="demo",
            extra={"source": "oracle_smoke_cases_for_demo"},
        ),
    )


def _case_from_raw(raw: dict[str, Any]) -> DemoCase:
    task = VeriLongTask(
        id=str(raw["task_id"]),
        task_family=raw["task_family"],
        difficulty=raw["difficulty"],
        question=str(raw["question"]),
        documents=[EvidenceDocument(**document) for document in raw["documents"]],
        gold_answer=str(raw["gold_answer"]),
        gold_evidence_ids=list(raw["gold_evidence_ids"]),
        distractor_evidence_ids=list(raw.get("distractor_evidence_ids", [])),
        stale_evidence_ids=list(raw.get("stale_evidence_ids", [])),
        expected_steps=_expected_steps_from_raw(raw),
        metadata=TaskMetadata(
            target_context_tokens=_estimate_target_context_tokens_raw(raw),
            split="demo",
            extra={"source": "oracle_smoke_cases_for_demo"},
        ),
    )
    return DemoCase(
        task_id=task.id,
        task_family=task.task_family,
        difficulty=task.difficulty,
        question=task.question,
        documents=[DemoDocument(**document.model_dump()) for document in task.documents],
        gold_answer=task.gold_answer,
        gold_evidence_ids=task.gold_evidence_ids,
        distractor_evidence_ids=task.distractor_evidence_ids,
        stale_evidence_ids=task.stale_evidence_ids,
        model=str(raw["model"]),
        model_output=str(raw["model_output"]),
        parsed_output=dict(raw["parsed_output"]),
        metric_breakdown=dict(raw["metric_breakdown"]),
        error_type=raw.get("error_type"),
        prompt_preview=build_user_prompt(task),
    )


def _expected_steps_from_raw(raw: dict[str, Any]) -> list[str]:
    parsed = raw.get("parsed_output", {})
    steps = parsed.get("pred_steps") if isinstance(parsed, dict) else None
    if isinstance(steps, list) and steps:
        return [str(step) for step in steps]
    return [f"Use {', '.join(raw['gold_evidence_ids'])} to answer {raw['gold_answer']}."]


def _expected_steps_from_case(case: DemoCase) -> list[str]:
    steps = case.parsed_output.get("pred_steps")
    if isinstance(steps, list) and steps:
        return [str(step) for step in steps]
    return [f"Use {', '.join(case.gold_evidence_ids)} to answer {case.gold_answer}."]


def _estimate_target_context_tokens(case: DemoCase) -> int:
    return max(1000, sum(len(document.text.split()) for document in case.documents))


def _estimate_target_context_tokens_raw(raw: dict[str, Any]) -> int:
    return max(1000, sum(len(str(document["text"]).split()) for document in raw["documents"]))


def _metric_as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
