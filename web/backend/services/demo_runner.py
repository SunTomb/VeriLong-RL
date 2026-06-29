from __future__ import annotations

from experiments.eval_api.run_api_eval import dry_run_response
from benchmark.reward.score import score_output_record
from web.backend.schemas.demo import DemoRunResponse
from web.backend.services.results_loader import case_to_task, get_case


def run_dry_demo(task_id: str) -> DemoRunResponse | None:
    case = get_case(task_id)
    if case is None:
        return None

    task = case_to_task(case)
    response = dry_run_response(task)
    scored = score_output_record(
        task,
        {
            "model": "dry_run_oracle_stub",
            "output_text": response.output_text,
        },
    )
    return DemoRunResponse(
        task_id=task.id,
        model="dry_run_oracle_stub",
        source="dry_run",
        output_text=response.output_text,
        parsed_output=scored["parsed"],
        metric_breakdown={
            "answer_exact_match": scored["answer_exact_match"],
            "answer_normalized_match": scored["answer_normalized_match"],
            "format_valid": scored["format_valid"],
            "step_count_valid": scored["step_count_valid"],
            "citation_precision": scored["citation_precision"],
            "citation_recall": scored["citation_recall"],
            "citation_f1": scored["citation_f1"],
            "distractor_citation_rate": scored["distractor_citation_rate"],
            "stale_citation_rate": scored["stale_citation_rate"],
            "invalid_citation_rate": scored["invalid_citation_rate"],
            "overcitation_rate": scored["overcitation_rate"],
            "reward_total": scored["reward_total"],
            "reward_components": scored["reward_components"],
        },
        error_type=scored["error_type"],
        prompt_preview=case.prompt_preview,
    )
