from dataclasses import dataclass


DEFAULT_REWARD_WEIGHTS = {
    "answer": 0.40,
    "citation": 0.25,
    "reasoning": 0.20,
    "format": 0.10,
    "distractor": 0.15,
    "stale": 0.15,
    "invalid": 0.10,
}


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    components: dict[str, float]


def compute_reward(
    answer_score: float,
    citation_f1: float,
    reasoning_score: float,
    format_score: float,
    distractor_rate: float,
    stale_rate: float,
    invalid_rate: float,
    weights: dict[str, float] | None = None,
) -> RewardBreakdown:
    active_weights = weights or DEFAULT_REWARD_WEIGHTS
    components = {
        "answer": active_weights["answer"] * answer_score,
        "citation": active_weights["citation"] * citation_f1,
        "reasoning": active_weights["reasoning"] * reasoning_score,
        "format": active_weights["format"] * format_score,
        "distractor_penalty": -active_weights["distractor"] * distractor_rate,
        "stale_penalty": -active_weights["stale"] * stale_rate,
        "invalid_penalty": -active_weights["invalid"] * invalid_rate,
    }
    return RewardBreakdown(total=sum(components.values()), components=components)
