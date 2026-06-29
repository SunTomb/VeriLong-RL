import collections
from pathlib import Path

import pytest

from benchmark.generator.build_pilot import _generate_tasks, load_config
from benchmark.generator.profiles import resolve_generation_kwargs


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _small_config():
    # 30 tasks per family so 0.7/0.1/0.2 splits divide cleanly.
    return {
        "seed": 1,
        "size": 90,
        "task_mix": {
            "anti_distractor_retrieval": 30,
            "multi_hop_reasoning": 30,
            "temporal_update": 30,
        },
        "target_context_tokens": [8000],
        "splits": {"train": 0.7, "dev": 0.1, "test": 0.2},
        "extra_splits": {"judge_subset_size": 9, "live_demo_subset_size": 6},
    }


def test_splits_are_stratified_across_families():
    tasks = _generate_tasks(_small_config())

    by_split = collections.defaultdict(collections.Counter)
    for task in tasks:
        by_split[task.metadata.split][task.task_family] += 1

    # Every split must contain all three families, not just temporal_update.
    for split_name in ("train", "dev", "test"):
        families = set(by_split[split_name])
        assert families == {
            "anti_distractor_retrieval",
            "multi_hop_reasoning",
            "temporal_update",
        }, f"{split_name} missing families: {families}"

    # Proportions hold within each family (30 -> 21 train / 3 dev / 6 test).
    for family in ("anti_distractor_retrieval", "multi_hop_reasoning", "temporal_update"):
        assert by_split["train"][family] == 21
        assert by_split["dev"][family] == 3
        assert by_split["test"][family] == 6


def test_extra_subsets_span_multiple_families():
    tasks = _generate_tasks(_small_config())

    judge_families = collections.Counter()
    demo_families = collections.Counter()
    for task in tasks:
        if task.metadata.extra.get("judge_subset"):
            judge_families[task.task_family] += 1
        if task.metadata.extra.get("live_demo_subset"):
            demo_families[task.task_family] += 1

    assert sum(judge_families.values()) == 9
    assert sum(demo_families.values()) == 6
    # Stratified round-robin must touch more than one family.
    assert len(judge_families) == 3
    assert len(demo_families) >= 2


def test_resolve_generation_kwargs_uses_hard_defaults():
    kwargs = resolve_generation_kwargs(
        task_family="anti_distractor_retrieval",
        difficulty="hard",
        generation_config={},
    )

    assert kwargs["difficulty"] == "hard"
    assert kwargs["distractor_count"] >= 8
    assert kwargs["distractor_strength"] == "adversarial"
    assert kwargs["evidence_position"] == "random"


def test_resolve_generation_kwargs_applies_yaml_override():
    kwargs = resolve_generation_kwargs(
        task_family="multi_hop_reasoning",
        difficulty="hard",
        generation_config={
            "hard": {
                "multi_hop_reasoning": {
                    "hop_count": 5,
                    "irrelevant_rule_count": 8,
                    "conflicting_rule_count": 2,
                }
            }
        },
    )

    assert kwargs == {
        "difficulty": "hard",
        "hop_count": 5,
        "irrelevant_rule_count": 8,
        "conflicting_rule_count": 2,
    }


def test_resolve_generation_kwargs_rejects_non_mapping_difficulty_override():
    with pytest.raises(ValueError, match="generation override must be a mapping for hard"):
        resolve_generation_kwargs(
            task_family="anti_distractor_retrieval",
            difficulty="hard",
            generation_config={"hard": ["not", "a", "mapping"]},
        )


def test_resolve_generation_kwargs_rejects_difficulty_override():
    with pytest.raises(ValueError, match="cannot override canonical difficulty for hard/temporal_update"):
        resolve_generation_kwargs(
            task_family="temporal_update",
            difficulty="hard",
            generation_config={
                "hard": {
                    "temporal_update": {
                        "difficulty": "easy",
                        "update_count": 1,
                    }
                }
            },
        )


def test_resolve_generation_kwargs_rejects_unsupported_difficulty():
    with pytest.raises(ValueError) as exc_info:
        resolve_generation_kwargs(
            task_family="anti_distractor_retrieval",
            difficulty="extreme",
            generation_config={},
        )

    assert "unsupported_difficulty" in str(exc_info.value)


def test_resolve_generation_kwargs_rejects_unsupported_task_family():
    with pytest.raises(ValueError) as exc_info:
        resolve_generation_kwargs(
            task_family="unsupported_family",
            difficulty="hard",
            generation_config={},
        )

    assert "unsupported_task_family" in str(exc_info.value)


def test_resolve_generation_kwargs_rejects_non_mapping_family_override():
    with pytest.raises(ValueError) as exc_info:
        resolve_generation_kwargs(
            task_family="anti_distractor_retrieval",
            difficulty="hard",
            generation_config={"hard": {"anti_distractor_retrieval": 1}},
        )

    assert "generation override must be a mapping" in str(exc_info.value)


def test_resolve_generation_kwargs_rejects_unknown_kwargs():
    with pytest.raises(ValueError) as exc_info:
        resolve_generation_kwargs(
            task_family="anti_distractor_retrieval",
            difficulty="hard",
            generation_config={"hard": {"anti_distractor_retrieval": {"bogus": 1}}},
        )

    assert "unsupported_generation_kwargs" in str(exc_info.value)


def test_generate_tasks_uses_difficulty_distribution():
    config = _small_config()
    config["difficulty"] = {"easy": 0.2, "medium": 0.3, "hard": 0.5}

    tasks = _generate_tasks(config)
    counts = collections.Counter(task.difficulty for task in tasks)

    assert counts["easy"] == 18
    assert counts["medium"] == 27
    assert counts["hard"] == 45


def test_difficulty_is_distributed_within_each_family():
    config = _small_config()
    config["difficulty"] = {"easy": 0.2, "medium": 0.3, "hard": 0.5}

    tasks = _generate_tasks(config)
    by_family = collections.defaultdict(collections.Counter)
    for task in tasks:
        by_family[task.task_family][task.difficulty] += 1

    # Each family (30 tasks) must hold the same 6/9/15 difficulty mix, so a
    # difficulty level cannot concentrate in one family.
    for family in ("anti_distractor_retrieval", "multi_hop_reasoning", "temporal_update"):
        assert by_family[family]["easy"] == 6
        assert by_family[family]["medium"] == 9
        assert by_family[family]["hard"] == 15


def test_splits_are_stratified_across_difficulty():
    # 100 tasks per family so each (family, difficulty) group is large enough
    # that a 10% dev slice does not round down to zero.
    config = {
        "seed": 5,
        "size": 300,
        "task_mix": {
            "anti_distractor_retrieval": 100,
            "multi_hop_reasoning": 100,
            "temporal_update": 100,
        },
        "target_context_tokens": [8000],
        "difficulty": {"easy": 0.2, "medium": 0.3, "hard": 0.5},
        "splits": {"train": 0.7, "dev": 0.1, "test": 0.2},
        "extra_splits": {"judge_subset_size": 0, "live_demo_subset_size": 0},
    }

    tasks = _generate_tasks(config)
    by_split = collections.defaultdict(collections.Counter)
    for task in tasks:
        by_split[task.metadata.split][task.difficulty] += 1

    # Every split must contain all three difficulty levels, not concentrate one.
    for split_name in ("train", "dev", "test"):
        assert set(by_split[split_name]) == {"easy", "medium", "hard"}, (
            f"{split_name} difficulty coverage: {dict(by_split[split_name])}"
        )


def test_difficulty_proportions_must_sum_to_one():
    config = _small_config()
    config["difficulty"] = {"easy": 0.6, "hard": 0.6}

    with pytest.raises(ValueError, match="difficulty proportions must sum to 1.0"):
        _generate_tasks(config)


def test_generate_tasks_applies_hard_generation_overrides():
    config = {
        "seed": 7,
        "size": 9,
        "task_mix": {
            "anti_distractor_retrieval": 3,
            "multi_hop_reasoning": 3,
            "temporal_update": 3,
        },
        "target_context_tokens": [8000],
        "difficulty": {"hard": 1.0},
        "splits": {"train": 0.7, "dev": 0.1, "test": 0.2},
        "extra_splits": {"judge_subset_size": 0, "live_demo_subset_size": 0},
        "generation": {
            "hard": {
                "anti_distractor_retrieval": {"distractor_count": 9},
                "multi_hop_reasoning": {"hop_count": 4, "irrelevant_rule_count": 6, "conflicting_rule_count": 1},
                "temporal_update": {"update_count": 3, "stale_count": 7},
            }
        },
    }

    tasks = _generate_tasks(config)
    by_family = {}
    for task in tasks:
        by_family.setdefault(task.task_family, task)

    assert {task.difficulty for task in tasks} == {"hard"}
    assert by_family["anti_distractor_retrieval"].metadata.extra["distractor_count"] == 9
    assert by_family["multi_hop_reasoning"].metadata.hop_count == 4
    assert by_family["multi_hop_reasoning"].metadata.extra["irrelevant_rule_count"] == 6
    assert by_family["temporal_update"].metadata.update_count == 3
    assert by_family["temporal_update"].metadata.extra["stale_count"] == 7


def test_hard_config_is_all_hard_and_balanced():
    config = load_config(_REPO_ROOT / "configs" / "hard.yaml")

    assert config["size"] == 180
    assert config["task_mix"] == {
        "anti_distractor_retrieval": 60,
        "multi_hop_reasoning": 60,
        "temporal_update": 60,
    }
    assert config["target_context_tokens"] == [16000, 32000]
    assert config["difficulty"] == {"hard": 1.0}
    assert config["generation"]["hard"]["anti_distractor_retrieval"]["distractor_count"] == 12
    assert config["generation"]["hard"]["multi_hop_reasoning"]["hop_count"] == 5
    assert config["generation"]["hard"]["temporal_update"]["update_count"] == 4
