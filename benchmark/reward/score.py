from typing import Any

from benchmark.metrics.answer import exact_match, normalized_match
from benchmark.metrics.citation import citation_scores
from benchmark.metrics.format import format_scores
from benchmark.parser.output_parser import parse_model_output
from benchmark.reward.programmatic import compute_reward
from benchmark.schemas.task import VeriLongTask


def score_output_record(task: VeriLongTask, output_record: dict[str, Any]) -> dict[str, Any]:
    output_text = str(output_record.get("output_text", output_record.get("output", "")))
    parsed = parse_model_output(output_text, valid_evidence_ids=task.evidence_ids())
    citation = citation_scores(
        pred_evidence_ids=parsed.pred_evidence_ids,
        gold_evidence_ids=task.gold_evidence_ids,
        distractor_evidence_ids=task.distractor_evidence_ids,
        stale_evidence_ids=task.stale_evidence_ids,
        valid_evidence_ids=task.evidence_ids(),
    )
    format_breakdown = format_scores(parsed, task)
    answer_score = normalized_match(parsed.pred_answer, task.gold_answer)
    reward = compute_reward(
        answer_score=answer_score,
        citation_f1=citation.f1,
        reasoning_score=format_breakdown.step_count_valid,
        format_score=format_breakdown.format_valid,
        distractor_rate=citation.distractor_citation_rate,
        stale_rate=citation.stale_citation_rate,
        invalid_rate=citation.invalid_citation_rate,
    )
    return {
        "task_id": task.id,
        "model": output_record.get("model", "unknown"),
        "task_family": task.task_family,
        "difficulty": task.difficulty,
        "answer_exact_match": exact_match(parsed.pred_answer, task.gold_answer),
        "answer_normalized_match": answer_score,
        "format_valid": format_breakdown.format_valid,
        "step_count_valid": format_breakdown.step_count_valid,
        "citation_precision": citation.precision,
        "citation_recall": citation.recall,
        "citation_f1": citation.f1,
        "all_gold_evidence_recall": citation.all_gold_evidence_recall,
        "distractor_citation_rate": citation.distractor_citation_rate,
        "stale_citation_rate": citation.stale_citation_rate,
        "invalid_citation_rate": citation.invalid_citation_rate,
        "overcitation_rate": citation.overcitation_rate,
        "reward_total": reward.total,
        "reward_components": reward.components,
        "parsed": parsed.model_dump(mode="json"),
        "error_type": _error_type(
            parsed.error_flags,
            format_breakdown.step_count_valid,
            citation.distractor_citation_rate,
            citation.stale_citation_rate,
        ),
    }


def _error_type(
    error_flags: list[str],
    step_count_valid: float,
    distractor_rate: float,
    stale_rate: float,
) -> str | None:
    if error_flags:
        return "format_or_invalid_citation"
    if step_count_valid < 1.0:
        return "invalid_step_count"
    if distractor_rate > 0:
        return "distractor_citation"
    if stale_rate > 0:
        return "stale_citation"
    return None
