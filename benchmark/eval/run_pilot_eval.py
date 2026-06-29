import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.metrics.aggregate import aggregate_numeric
from benchmark.schemas.task import VeriLongTask
from scripts.score_outputs import SUMMARY_FIELDS, score_output_record


SUPPORTED_BASELINES = {"oracle_format_baseline", "corrupted_distractor_baseline"}


def run_pilot_eval(tasks_path: str | Path, baseline: str, out_dir: str | Path) -> dict[str, float | int]:
    if baseline not in SUPPORTED_BASELINES:
        raise ValueError(f"unsupported_baseline:{baseline}")

    tasks = _load_tasks(Path(tasks_path))
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = [_make_output_record(task, baseline) for task in tasks]
    scored = [score_output_record(task, output) for task, output in zip(tasks, outputs, strict=True)]
    summary: dict[str, float | int] = aggregate_numeric(scored, SUMMARY_FIELDS)
    summary["count"] = len(scored)
    summary["baseline"] = baseline

    _write_jsonl(output_dir / "outputs.jsonl", outputs)
    _write_jsonl(output_dir / "scored.jsonl", scored)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "cases_for_demo.json").write_text(
        json.dumps(_make_demo_cases(tasks, outputs, scored), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_tasks(path: Path) -> list[VeriLongTask]:
    tasks: list[VeriLongTask] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                tasks.append(VeriLongTask.model_validate(json.loads(line)))
    return tasks


def _make_output_record(task: VeriLongTask, baseline: str) -> dict[str, Any]:
    evidence_ids = task.gold_evidence_ids
    if baseline == "corrupted_distractor_baseline" and task.distractor_evidence_ids:
        evidence_ids = [task.distractor_evidence_ids[0]] + task.gold_evidence_ids[1:]

    return {
        "task_id": task.id,
        "model": baseline,
        "output_text": _format_output(evidence_ids=evidence_ids, answer=task.gold_answer, task=task),
        "metadata": {"source": "synthetic_smoke_test"},
    }


def _format_output(evidence_ids: list[str], answer: str, task: VeriLongTask) -> str:
    """Build a format-correct oracle output whose step count fits task bounds.

    The oracle baseline is the "perfect format" reference, so its step count
    must satisfy the task-derived bounds (e.g. multi-hop needs >= hop_count
    steps). We emit one grounded step per gold evidence id, then pad/clamp into
    the allowed [min, max] band.
    """

    min_steps, max_steps = task.step_bounds()
    evidence_text = ", ".join(evidence_ids)
    first_evidence = evidence_ids[0] if evidence_ids else "the cited evidence"

    step_sentences = [f"{eid} provides relevant evidence." for eid in evidence_ids[:max_steps]]
    if not step_sentences:
        step_sentences = [f"{first_evidence} states the relevant fact."]
    # Final step states the conclusion; pad up to the minimum if needed.
    while len(step_sentences) < max(min_steps, 1):
        step_sentences.append(f"Combining the cited evidence yields {answer}.")
    step_sentences = step_sentences[:max_steps]
    step_sentences[-1] = f"Therefore the answer is {answer}."

    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(step_sentences, start=1))
    return f"Evidence: {evidence_text}\nSteps:\n{numbered}\nAnswer: {answer}"


def _make_demo_cases(
    tasks: list[VeriLongTask],
    outputs: list[dict[str, Any]],
    scored: list[dict[str, Any]],
    limit: int = 24,
) -> list[dict[str, Any]]:
    # Stratify by task family so every family present in the dataset is
    # represented, instead of taking the head of a family-grouped list.
    by_family: dict[str, list[int]] = {}
    for index, task in enumerate(tasks):
        by_family.setdefault(task.task_family, []).append(index)

    ordered_indices: list[int] = []
    family_queues = list(by_family.values())
    position = 0
    while len(ordered_indices) < min(limit, len(tasks)):
        progressed = False
        for queue in family_queues:
            if position < len(queue):
                ordered_indices.append(queue[position])
                progressed = True
                if len(ordered_indices) >= min(limit, len(tasks)):
                    break
        if not progressed:
            break
        position += 1

    cases = []
    for index in ordered_indices:
        task = tasks[index]
        output = outputs[index]
        score = scored[index]
        cases.append(
            {
                "task_id": task.id,
                "task_family": task.task_family,
                "difficulty": task.difficulty,
                "question": task.question,
                "documents": [document.model_dump(mode="json") for document in task.documents],
                "gold_answer": task.gold_answer,
                "gold_evidence_ids": task.gold_evidence_ids,
                "distractor_evidence_ids": task.distractor_evidence_ids,
                "stale_evidence_ids": task.stale_evidence_ids,
                "model": output["model"],
                "model_output": output["output_text"],
                "parsed_output": score["parsed"],
                "metric_breakdown": {
                    key: value
                    for key, value in score.items()
                    if key.endswith("_match")
                    or key.endswith("_valid")
                    or key.startswith("citation_")
                    or key.endswith("_rate")
                    or key == "reward_total"
                },
                "error_type": score["error_type"],
            }
        )
    return cases


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VeriLong-RL pilot evaluation smoke baselines.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--baseline", required=True, choices=sorted(SUPPORTED_BASELINES))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    summary = run_pilot_eval(tasks_path=args.tasks, baseline=args.baseline, out_dir=args.out_dir)
    print(
        f"baseline={args.baseline} count={summary['count']} "
        f"format_valid_mean={summary['format_valid_mean']:.3f} "
        f"citation_f1_mean={summary['citation_f1_mean']:.3f} "
        f"distractor_citation_rate_mean={summary['distractor_citation_rate_mean']:.3f}"
    )


if __name__ == "__main__":
    main()
