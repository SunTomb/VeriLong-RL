from copy import deepcopy
from typing import Any

from benchmark.schemas.task import Difficulty, TaskFamily


_GENERATOR_KWARGS = {
    "anti_distractor_retrieval": {
        "difficulty",
        "distractor_strength",
        "distractor_count",
        "evidence_position",
    },
    "multi_hop_reasoning": {
        "difficulty",
        "hop_count",
        "irrelevant_rule_count",
        "conflicting_rule_count",
    },
    "temporal_update": {
        "difficulty",
        "update_count",
        "stale_count",
        "evidence_position",
    },
}


_DEFAULT_PROFILES: dict[Difficulty, dict[TaskFamily, dict[str, Any]]] = {
    "easy": {
        "anti_distractor_retrieval": {
            "difficulty": "easy",
            "distractor_strength": "lexical",
            "distractor_count": 1,
            "evidence_position": "front",
        },
        "multi_hop_reasoning": {
            "difficulty": "easy",
            "hop_count": 2,
            "irrelevant_rule_count": 1,
            "conflicting_rule_count": 0,
        },
        "temporal_update": {
            "difficulty": "easy",
            "update_count": 1,
            "stale_count": 2,
            "evidence_position": "front",
        },
    },
    "medium": {
        "anti_distractor_retrieval": {
            "difficulty": "medium",
            "distractor_strength": "semantic",
            "distractor_count": 4,
            "evidence_position": "mixed",
        },
        "multi_hop_reasoning": {
            "difficulty": "medium",
            "hop_count": 3,
            "irrelevant_rule_count": 3,
            "conflicting_rule_count": 1,
        },
        "temporal_update": {
            "difficulty": "medium",
            "update_count": 2,
            "stale_count": 4,
            "evidence_position": "mixed",
        },
    },
    "hard": {
        "anti_distractor_retrieval": {
            "difficulty": "hard",
            "distractor_strength": "adversarial",
            "distractor_count": 12,
            "evidence_position": "random",
        },
        "multi_hop_reasoning": {
            "difficulty": "hard",
            "hop_count": 5,
            "irrelevant_rule_count": 8,
            "conflicting_rule_count": 2,
        },
        "temporal_update": {
            "difficulty": "hard",
            "update_count": 4,
            "stale_count": 8,
            "evidence_position": "mixed",
        },
    },
}


def resolve_generation_kwargs(
    task_family: TaskFamily,
    difficulty: Difficulty,
    generation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if difficulty not in _DEFAULT_PROFILES:
        raise ValueError(f"unsupported_difficulty:{difficulty}")
    if task_family not in _DEFAULT_PROFILES[difficulty]:
        raise ValueError(f"unsupported_task_family:{task_family}")

    kwargs = deepcopy(_DEFAULT_PROFILES[difficulty][task_family])
    difficulty_overrides = (generation_config or {}).get(difficulty, {})
    if not isinstance(difficulty_overrides, dict):
        raise ValueError(f"generation override must be a mapping for {difficulty}")

    overrides = difficulty_overrides.get(task_family, {})
    if not isinstance(overrides, dict):
        raise ValueError(f"generation override must be a mapping for {difficulty}/{task_family}")
    if "difficulty" in overrides:
        raise ValueError(f"cannot override canonical difficulty for {difficulty}/{task_family}")
    kwargs.update(overrides)

    allowed = _GENERATOR_KWARGS[task_family]
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise ValueError(f"unsupported_generation_kwargs:{task_family}:{','.join(unknown)}")

    return kwargs
