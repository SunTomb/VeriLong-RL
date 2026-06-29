"""Programmatic reward callback for GRPO, reusing the eval scoring core.

The reward a completion receives during RL is *identical* to the reward_total
the benchmark scorer would assign, so the RL objective and the reported metric
are the same ruler. We do not re-implement any scoring here.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from benchmark.reward.score import score_output_record
from benchmark.schemas.task import VeriLongTask


def _completion_text(completion: Any) -> str:
    """trl passes a plain string (text mode) or a chat list (conversational)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion)


def make_reward_fn() -> Callable[..., list[float]]:
    def reward_fn(prompts=None, completions=None, task_json=None, **_kwargs) -> list[float]:
        rewards: list[float] = []
        for i, completion in enumerate(completions):
            task = VeriLongTask.model_validate(json.loads(task_json[i]))
            text = _completion_text(completion)
            scored = score_output_record(task, {"output_text": text})
            rewards.append(float(scored["reward_total"]))
        return rewards

    reward_fn.__name__ = "programmatic_citation_reward"
    return reward_fn
