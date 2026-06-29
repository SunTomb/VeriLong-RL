import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.metrics.aggregate import aggregate_numeric
from benchmark.reward.score import score_output_record
from benchmark.schemas.task import VeriLongTask


SUMMARY_FIELDS = [
    "answer_exact_match",
    "answer_normalized_match",
    "format_valid",
    "step_count_valid",
    "citation_precision",
    "citation_recall",
    "citation_f1",
    "all_gold_evidence_recall",
    "distractor_citation_rate",
    "stale_citation_rate",
    "invalid_citation_rate",
    "overcitation_rate",
    "reward_total",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score VeriLong-RL model output JSONL against pilot tasks.")
    parser.add_argument("--tasks", required=True, help="Path to VeriLong-RL task JSONL.")
    parser.add_argument("--outputs", required=True, help="Path to model output JSONL with task_id/model/output_text fields.")
    parser.add_argument("--scored", required=True, help="Path to write scored JSONL.")
    parser.add_argument("--summary", required=True, help="Path to write aggregate summary JSON.")
    args = parser.parse_args()

    tasks = _load_tasks(Path(args.tasks))
    scored_records = []
    with Path(args.outputs).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            output_record = json.loads(line)
            task_id = output_record["task_id"]
            scored_records.append(score_output_record(tasks[task_id], output_record))

    scored_path = Path(args.scored)
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    with scored_path.open("w", encoding="utf-8") as handle:
        for record in scored_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = aggregate_numeric(scored_records, SUMMARY_FIELDS)
    summary["count"] = len(scored_records)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"scored={len(scored_records)} scored_path={scored_path} summary={summary_path}")


def _load_tasks(path: Path) -> dict[str, VeriLongTask]:
    tasks = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            task = VeriLongTask.model_validate(json.loads(line))
            tasks[task.id] = task
    return tasks


if __name__ == "__main__":
    main()
