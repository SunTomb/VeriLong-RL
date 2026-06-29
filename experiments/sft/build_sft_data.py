"""Convert the VeriLong-RL pilot train split into SFT chat examples.

The assistant target is assembled from programmatic gold so it is, by
construction, a perfect reference under the existing parser/metrics:

- Evidence: the task's gold_evidence_ids (never distractor/stale).
- Steps: the task's expected_steps, numbered.
- Answer: the task's gold_answer.

This pure-gold target teaches the format and citation discipline without
risking the over-citation habit a model-distilled target could introduce.
The system/user messages reuse the exact eval prompt so training matches
evaluation. API-distilled Steps are a later increment (Step B), not here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.schemas.task import VeriLongTask
from experiments.eval_api.run_api_eval import SYSTEM_PROMPT, build_user_prompt


def build_assistant_text(task: VeriLongTask, steps: list[str] | None = None) -> str:
    evidence_line = "Evidence: " + ", ".join(task.gold_evidence_ids)
    steps = steps if steps is not None else task.expected_steps
    step_lines = [f"{i}. {step}" for i, step in enumerate(steps, start=1)]
    return "\n".join([evidence_line, "Steps:", *step_lines, f"Answer: {task.gold_answer}"])


def build_sft_example(task: VeriLongTask, distilled_steps: dict[str, list[str]] | None = None) -> dict[str, Any]:
    steps = None
    distilled = False
    if distilled_steps and task.id in distilled_steps:
        steps = distilled_steps[task.id]
        distilled = True
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(task)},
            {"role": "assistant", "content": build_assistant_text(task, steps)},
        ],
        "metadata": {
            "task_id": task.id,
            "task_family": task.task_family,
            "difficulty": task.difficulty,
            "split": task.metadata.split,
            "target_context_tokens": task.metadata.target_context_tokens,
            "distilled_steps": distilled,
        },
    }


def build_sft_dataset(
    tasks_path: Path,
    out_path: Path,
    split: str | None,
    distilled_steps_path: Path | None = None,
) -> dict[str, Any]:
    distilled_steps: dict[str, list[str]] | None = None
    if distilled_steps_path is not None:
        distilled_steps = json.loads(distilled_steps_path.read_text(encoding="utf-8"))

    written = 0
    distilled_used = 0
    by_family: dict[str, int] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tasks_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            task = VeriLongTask.model_validate(json.loads(line))
            if split is not None and task.metadata.split != split:
                continue
            example = build_sft_example(task, distilled_steps)
            dst.write(json.dumps(example, ensure_ascii=False) + "\n")
            written += 1
            if example["metadata"]["distilled_steps"]:
                distilled_used += 1
            by_family[task.task_family] = by_family.get(task.task_family, 0) + 1
    return {"written": written, "distilled_used": distilled_used, "by_family": by_family, "out": str(out_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VeriLong-RL SFT chat data from pilot tasks.")
    parser.add_argument("--tasks", default="data/pilot/tasks.jsonl")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSONL path. Defaults to data/sft/{split}.jsonl so a non-train split is not written to the train path.",
    )
    parser.add_argument(
        "--distilled-steps",
        default=None,
        help="Optional JSON map {task_id: [steps]} from distill_steps.py. When given, those tasks use the distilled Steps text (Evidence/Answer stay gold).",
    )
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path(f"data/sft/{args.split}.jsonl")
    distilled_path = Path(args.distilled_steps) if args.distilled_steps else None
    summary = build_sft_dataset(Path(args.tasks), out_path, split=args.split, distilled_steps_path=distilled_path)
    print(
        f"written={summary['written']} distilled_used={summary['distilled_used']} "
        f"by_family={summary['by_family']} out={summary['out']}"
    )


if __name__ == "__main__":
    main()
