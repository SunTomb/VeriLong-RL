"""Run Claude API evaluation over VeriLong-RL pilot tasks.

This runner builds evidence-grounded prompts, calls the Claude client (or a
deterministic stub under ``--dry-run``), caches each raw response on disk, and
writes an output JSONL that the existing scorer
(``scripts/score_outputs.py`` / ``benchmark.eval``) can process unchanged.

Entry gate (see ``experiments/eval_api/README.md``): only run live API calls
after Tasks 1-6 pass locally, and start with a small dev subset, e.g.::

    python experiments/eval_api/run_api_eval.py \
        --tasks data/pilot/tasks.jsonl \
        --split dev \
        --limit 30 \
        --model claude-opus-4-8 \
        --out results/raw/api/claude_opus_4_8_pilot_dev30.jsonl

Use ``--dry-run`` to validate prompt construction, caching, and the output
contract without spending API budget. Dry-run output is clearly tagged so it is
never mistaken for real model metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.schemas.task import VeriLongTask
from experiments.eval_api.claude_client import ClaudeClient, ClaudeResponse
from experiments.eval_api.openai_client import OpenAICompatibleClient

PROMPT_VERSION = "phase1-v1"

SYSTEM_PROMPT = (
    "You are solving a VeriLong-RL evidence-grounded long-context task.\n"
    "Return exactly this format:\n"
    "Evidence: E01, E02\n"
    "Steps:\n"
    "1. One short sentence grounded in cited evidence.\n"
    "2. One short sentence grounded in cited evidence.\n"
    "Answer: final answer only"
)


def build_user_prompt(task: VeriLongTask) -> str:
    """Construct the per-task prompt.

    Only neutral task-facing fields are exposed: id, question, and the full
    document list (doc_id, evidence_id, text). Gold answer, gold evidence IDs,
    distractor IDs, stale IDs, and any metric labels are never included.
    """

    lines: list[str] = [
        f"Task ID: {task.id}",
        f"Question: {task.question}",
        "",
        "Documents:",
    ]
    for document in task.documents:
        lines.append(f"[{document.evidence_id}] (doc {document.doc_id}) {document.text}")
    lines.extend(
        [
            "",
            "Cite only evidence IDs that directly support your answer.",
            "Respond using exactly the required format: Evidence, Steps, Answer.",
        ]
    )
    return "\n".join(lines)


def compute_task_hash(task: VeriLongTask, user_prompt: str) -> str:
    """Stable hash over the prompt input that the model actually receives."""

    digest = hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def cache_path(cache_root: Path, model: str, task_id: str) -> Path:
    return cache_root / model / f"{task_id}.json"


def load_cache(path: Path, model: str, task_hash: str) -> dict[str, Any] | None:
    """Return a cached record only if it matches the current model + task hash."""

    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if record.get("model") != model:
        return None
    if record.get("task_hash") != task_hash:
        return None
    if record.get("prompt_version") != PROMPT_VERSION:
        return None
    return record


def write_cache(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_tasks(
    path: Path,
    split: str | None,
    limit: int | None,
    stratify: bool = False,
) -> list[VeriLongTask]:
    matched: list[VeriLongTask] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            task = VeriLongTask.model_validate(json.loads(line))
            if split is not None and task.metadata.split != split:
                continue
            matched.append(task)

    if stratify and limit is not None:
        return _stratified_sample(matched, limit)
    if limit is not None:
        return matched[:limit]
    return matched


def _stratified_sample(tasks: list[VeriLongTask], limit: int) -> list[VeriLongTask]:
    """Round-robin across task families so a small limit still spans families."""

    by_family: dict[str, list[VeriLongTask]] = {}
    for task in tasks:
        by_family.setdefault(task.task_family, []).append(task)

    queues = list(by_family.values())
    sampled: list[VeriLongTask] = []
    position = 0
    while len(sampled) < min(limit, len(tasks)):
        progressed = False
        for queue in queues:
            if position < len(queue):
                sampled.append(queue[position])
                progressed = True
                if len(sampled) >= min(limit, len(tasks)):
                    break
        if not progressed:
            break
        position += 1
    return sampled


def dry_run_response(task: VeriLongTask) -> ClaudeResponse:
    """Deterministic offline stub.

    This is NOT a model prediction. It echoes gold fields purely to exercise the
    output contract and scorer wiring; rows are tagged ``source=dry_run`` so they
    can never be mistaken for real API metrics.
    """

    evidence_ids = task.gold_evidence_ids or [next(iter(task.evidence_ids()), "E01")]
    first = evidence_ids[0]
    text = (
        f"Evidence: {', '.join(evidence_ids)}\n"
        "Steps:\n"
        f"1. {first} states the relevant fact.\n"
        f"2. Therefore the answer is {task.gold_answer}.\n"
        f"Answer: {task.gold_answer}"
    )
    return ClaudeResponse(output_text=text, raw_metadata={"stub": True})


def run(
    tasks_path: Path,
    out_path: Path,
    model: str,
    split: str | None,
    limit: int | None,
    cache_root: Path,
    dry_run: bool,
    max_tokens: int,
    provider: str = "claude",
    base_url: str | None = None,
    api_key: str | None = None,
    stratify: bool = False,
) -> dict[str, Any]:
    tasks = load_tasks(tasks_path, split=split, limit=limit, stratify=stratify)
    if not tasks:
        raise SystemExit(f"no tasks matched split={split!r} in {tasks_path}")

    client: ClaudeClient | OpenAICompatibleClient | None = None
    if not dry_run:
        if provider == "claude":
            client = ClaudeClient(model=model, max_tokens=max_tokens)
        elif provider == "openai":
            client = OpenAICompatibleClient(
                model=model, max_tokens=max_tokens, base_url=base_url, api_key=api_key
            )
        else:
            raise SystemExit(f"unknown provider: {provider!r} (use 'claude' or 'openai')")

    source = "dry_run" if dry_run else "api"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache_hits = 0
    api_calls = 0
    errors = 0
    output_rows: list[dict[str, Any]] = []

    # Stream rows to disk as we go so a mid-run failure (e.g. a flaky gateway)
    # preserves completed work instead of discarding the whole batch.
    with out_path.open("w", encoding="utf-8") as out_handle:
        for task in tasks:
            user_prompt = build_user_prompt(task)
            task_hash = compute_task_hash(task, user_prompt)
            cpath = cache_path(cache_root, model, task.id)
            error_flag: str | None = None

            cached = load_cache(cpath, model=model, task_hash=task_hash) if not dry_run else None
            if cached is not None:
                output_text = cached["output_text"]
                cache_hits += 1
            elif dry_run:
                output_text = dry_run_response(task).output_text
            else:
                assert client is not None
                try:
                    response = client.complete(SYSTEM_PROMPT, user_prompt)
                    api_calls += 1
                    output_text = response.output_text
                    write_cache(
                        cpath,
                        {
                            "task_id": task.id,
                            "model": model,
                            "prompt_version": PROMPT_VERSION,
                            "task_hash": task_hash,
                            "output_text": output_text,
                            "raw_response_metadata": response.raw_metadata,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                except Exception as exc:  # noqa: BLE001 - record and continue
                    errors += 1
                    output_text = ""
                    error_flag = str(exc)[:300]

            metadata: dict[str, Any] = {
                "source": source,
                "provider": provider,
                "prompt_version": PROMPT_VERSION,
            }
            if error_flag is not None:
                metadata["error"] = error_flag

            row = {
                "task_id": task.id,
                "model": model,
                "output_text": output_text,
                "metadata": metadata,
            }
            output_rows.append(row)
            out_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_handle.flush()

    return {
        "tasks": len(tasks),
        "split": split,
        "model": model,
        "source": source,
        "cache_hits": cache_hits,
        "errors": errors,
        "api_calls": api_calls,
        "out": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Claude API eval over VeriLong-RL pilot tasks.")
    parser.add_argument("--tasks", required=True, help="Path to VeriLong-RL task JSONL.")
    parser.add_argument("--out", required=True, help="Path to write output JSONL (scorer contract).")
    parser.add_argument("--model", default="claude-opus-4-8")
    parser.add_argument(
        "--provider",
        default="claude",
        choices=["claude", "openai"],
        help="API provider. 'openai' targets any OpenAI-compatible /v1/chat/completions gateway.",
    )
    parser.add_argument("--base-url", default=None, help="Base URL for openai provider (or OPENAI_BASE_URL).")
    parser.add_argument("--api-key", default=None, help="API key for openai provider (or OPENAI_API_KEY).")
    parser.add_argument("--split", default=None, help="Filter by metadata.split (e.g. dev/test/train).")
    parser.add_argument("--limit", type=int, default=None, help="Max number of tasks to evaluate.")
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="With --limit, sample evenly across task families (round-robin).",
    )
    parser.add_argument(
        "--cache-root",
        default="results/raw/api_cache",
        help="Root directory for per-task response cache.",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exercise the pipeline offline without API calls. Output is tagged source=dry_run.",
    )
    args = parser.parse_args()

    summary = run(
        tasks_path=Path(args.tasks),
        out_path=Path(args.out),
        model=args.model,
        split=args.split,
        limit=args.limit,
        cache_root=Path(args.cache_root),
        dry_run=args.dry_run,
        max_tokens=args.max_tokens,
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
        stratify=args.stratify,
    )
    print(
        f"source={summary['source']} model={summary['model']} split={summary['split']} "
        f"tasks={summary['tasks']} cache_hits={summary['cache_hits']} "
        f"api_calls={summary['api_calls']} errors={summary['errors']} out={summary['out']}"
    )


if __name__ == "__main__":
    main()
