"""Distill reasoning Steps from a strong API model for SFT augmentation.

Step B of the SFT plan. The safety boundary (locked in design): the API only
contributes the natural-language *Steps* phrasing, and only for tasks where its
output is citation-perfect. Evidence IDs and the Answer always remain
programmatic gold in the final SFT target, so a model's over-citation habit can
never leak into training data.

For each train task we call the model, parse its output, and ACCEPT its Steps
text only when ALL hold:
  - the cited evidence set equals the gold set exactly (precision=recall=1.0),
  - no distractor or stale citation,
  - the normalized answer matches gold,
  - the step count is within the task's structural bounds.
Otherwise the task is left out of the map (build_sft_data falls back to the
programmatic gold Steps for it).

Output: a JSON map {task_id: [step, step, ...]} plus an acceptance-rate report.
Runs on the cluster (needs the API gateway); --tasks defaults to the SFT-used
8K-context train subset via --max-context-tokens.

Usage:
    export OPENAI_BASE_URL=... OPENAI_API_KEY=...
    python experiments/sft/distill_steps.py \
        --tasks data/pilot/tasks.jsonl --split train \
        --model claude-opus-4-8 --max-context-tokens 8000 \
        --out data/sft/distilled_steps_claude.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.metrics.answer import normalized_match
from benchmark.metrics.citation import citation_scores
from benchmark.metrics.format import format_scores
from benchmark.parser.output_parser import parse_model_output
from benchmark.schemas.task import VeriLongTask
from experiments.eval_api.run_api_eval import build_user_prompt
from experiments.eval_api.openai_client import OpenAICompatibleClient


# A stricter system prompt used ONLY for distillation. The shared eval
# SYSTEM_PROMPT is intentionally lenient (it must not coach models during
# evaluation), but for harvesting clean Steps phrasing we demand the exact
# numbered format the parser expects and explicit citation discipline. This
# does not weaken the safety filter — every harvested output still must pass
# steps_are_gold_perfect; the stricter prompt only raises the acceptance rate
# of usable phrasings.
DISTILL_SYSTEM_PROMPT = (
    "You are solving a VeriLong-RL evidence-grounded long-context task.\n"
    "Return EXACTLY this format and nothing else:\n"
    "Evidence: E01, E02\n"
    "Steps:\n"
    "1. One short sentence grounded in a cited evidence ID.\n"
    "2. One short sentence grounded in a cited evidence ID.\n"
    "Answer: final answer only\n\n"
    "Rules:\n"
    "- The 'Steps:' header must be on its own line, followed by numbered lines (1., 2., ...).\n"
    "- Cite ONLY evidence IDs that directly support the answer. Do not cite distractor,\n"
    "  outdated, or merely-related evidence, even to dismiss it.\n"
    "- Use 1-4 numbered steps. Each step references the evidence it relies on.\n"
)


def steps_are_gold_perfect(task: VeriLongTask, output_text: str) -> list[str] | None:
    """Return the parsed Steps if the output is citation-perfect, else None."""

    parsed = parse_model_output(output_text, valid_evidence_ids=task.evidence_ids())
    if not parsed.pred_steps:
        return None
    fmt = format_scores(parsed, task)
    if fmt.format_valid != 1.0:
        return None
    if normalized_match(parsed.pred_answer or "", task.gold_answer) != 1.0:
        return None
    cit = citation_scores(
        pred_evidence_ids=parsed.pred_evidence_ids,
        gold_evidence_ids=task.gold_evidence_ids,
        distractor_evidence_ids=task.distractor_evidence_ids,
        stale_evidence_ids=task.stale_evidence_ids,
        valid_evidence_ids=task.evidence_ids(),
    )
    if cit.precision != 1.0 or cit.recall != 1.0:
        return None
    if cit.distractor_citation_rate != 0.0 or cit.stale_citation_rate != 0.0:
        return None
    return parsed.pred_steps


def load_train_tasks(tasks_path: Path, split: str, max_context_tokens: int | None) -> list[VeriLongTask]:
    tasks: list[VeriLongTask] = []
    with tasks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            task = VeriLongTask.model_validate(json.loads(line))
            if task.metadata.split != split:
                continue
            if max_context_tokens is not None and task.metadata.target_context_tokens > max_context_tokens:
                continue
            tasks.append(task)
    return tasks


def run(
    tasks_path: Path,
    out_path: Path,
    model: str,
    split: str,
    max_context_tokens: int | None,
    base_url: str | None,
    api_key: str | None,
    max_tokens: int,
    limit: int | None,
) -> dict[str, Any]:
    tasks = load_train_tasks(tasks_path, split, max_context_tokens)
    if limit is not None:
        tasks = tasks[:limit]
    if not tasks:
        raise SystemExit(f"no tasks matched split={split!r} max_context_tokens={max_context_tokens}")

    client = OpenAICompatibleClient(model=model, max_tokens=max_tokens, base_url=base_url, api_key=api_key, timeout=180.0)

    accepted: dict[str, list[str]] = {}
    by_family_total: dict[str, int] = {}
    by_family_accept: dict[str, int] = {}
    errors = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        by_family_total[task.task_family] = by_family_total.get(task.task_family, 0) + 1
        try:
            response = client.complete(DISTILL_SYSTEM_PROMPT, build_user_prompt(task))
        except Exception as exc:  # noqa: BLE001 - record and continue
            errors += 1
            print(f"error task={task.id}: {str(exc)[:120]}", file=sys.stderr)
            continue
        steps = steps_are_gold_perfect(task, response.output_text)
        if steps is not None:
            accepted[task.id] = steps
            by_family_accept[task.task_family] = by_family_accept.get(task.task_family, 0) + 1
        # Stream the map to disk so a mid-run gateway failure preserves progress.
        out_path.write_text(json.dumps(accepted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = {
        "model": model,
        "tasks": len(tasks),
        "accepted": len(accepted),
        "acceptance_rate": round(len(accepted) / len(tasks), 3) if tasks else 0.0,
        "errors": errors,
        "by_family_total": by_family_total,
        "by_family_accepted": by_family_accept,
        "out": str(out_path),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill citation-perfect Steps from an API model for SFT augmentation.")
    parser.add_argument("--tasks", default="data/pilot/tasks.jsonl")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="claude-opus-4-8")
    parser.add_argument("--max-context-tokens", type=int, default=8000, help="Only distill tasks at/below this context target (match SFT training subset).")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="data/sft/distilled_steps.json")
    args = parser.parse_args()

    report = run(
        tasks_path=Path(args.tasks),
        out_path=Path(args.out),
        model=args.model,
        split=args.split,
        max_context_tokens=args.max_context_tokens,
        base_url=args.base_url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
