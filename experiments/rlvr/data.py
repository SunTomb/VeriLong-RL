"""Build the GRPO prompt dataset from VeriLong-RL pilot tasks.

Each row carries the chat-style `prompt` (system + user, the exact eval/SFT
prompt) and `task_json` (the task serialized) so the reward callback can score
each rollout against programmatic gold. Filtering keeps a single family and a
single context length to scope the first GRPO run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from benchmark.schemas.task import VeriLongTask
from experiments.eval_api.run_api_eval import SYSTEM_PROMPT, build_user_prompt


def build_prompt_messages(task: VeriLongTask) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(task)},
    ]


def iter_grpo_rows(
    tasks_path: Path,
    family: str,
    max_context_tokens: int,
    split: str,
) -> Iterator[dict[str, Any]]:
    with Path(tasks_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            task = VeriLongTask.model_validate(json.loads(line))
            if task.task_family != family:
                continue
            if task.metadata.split != split:
                continue
            # exact match: scope to a single context length, not a ceiling
            if task.metadata.target_context_tokens != max_context_tokens:
                continue
            yield {"prompt": build_prompt_messages(task), "task_json": task.model_dump_json()}


def load_grpo_dataset(
    tasks_path: Path,
    family: str,
    max_context_tokens: int,
    split: str,
):
    """Return a HF Dataset. Imported lazily so unit tests need no `datasets`."""
    from datasets import Dataset  # noqa: PLC0415

    rows = list(iter_grpo_rows(tasks_path, family, max_context_tokens, split))
    if not rows:
        raise SystemExit(
            f"no tasks matched family={family!r} split={split!r} ctx={max_context_tokens}"
        )
    return Dataset.from_list(rows)
