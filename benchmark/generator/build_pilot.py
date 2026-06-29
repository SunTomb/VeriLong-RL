import json
from pathlib import Path
from typing import Any

import yaml

from benchmark.generator.multihop import generate_multihop_task
from benchmark.generator.profiles import resolve_generation_kwargs
from benchmark.generator.retrieval import generate_retrieval_task
from benchmark.generator.temporal import generate_temporal_task
from benchmark.schemas.task import VeriLongTask
from benchmark.validators.task_validator import ValidationReport, validate_task


_GENERATORS = {
    "anti_distractor_retrieval": generate_retrieval_task,
    "multi_hop_reasoning": generate_multihop_task,
    "temporal_update": generate_temporal_task,
}


class PilotValidationError(ValueError):
    def __init__(self, reports: list[tuple[str, ValidationReport]]):
        self.reports = reports
        details = "; ".join(f"{task_id}:{','.join(report.errors)}" for task_id, report in reports[:5])
        super().__init__(f"pilot validation failed for {len(reports)} task(s): {details}")


def load_config(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, dict):
        return config
    with Path(config).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("pilot config must be a mapping")
    return loaded


def build_pilot(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    config_dict = load_config(config)
    tasks = _generate_tasks(config_dict)
    invalid_reports = [(task.id, report) for task in tasks if not (report := validate_task(task)).valid]
    if invalid_reports:
        raise PilotValidationError(invalid_reports)

    output_value = str(config_dict["output_path"])
    output_path = Path(output_value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.model_dump(mode="json"), ensure_ascii=False) + "\n")

    return {"generated": len(tasks), "valid": len(tasks), "output": output_value}


def _generate_tasks(config: dict[str, Any]) -> list[VeriLongTask]:
    task_mix = config["task_mix"]
    expected_size = config.get("size")
    total_size = sum(int(count) for count in task_mix.values())
    if expected_size is not None and int(expected_size) != total_size:
        raise ValueError(f"size {expected_size} does not match task_mix total {total_size}")

    tasks: list[VeriLongTask] = []
    task_index = 1
    seed = int(config["seed"])
    target_context_tokens = list(config["target_context_tokens"])
    difficulty_mix = config.get("difficulty", {"medium": 1.0})
    generation_config = config.get("generation", {})

    for task_family, count in task_mix.items():
        if task_family not in _GENERATORS:
            raise ValueError(f"unsupported_task_family:{task_family}")
        # Each family draws its own difficulty sequence so a difficulty level
        # cannot concentrate in a single family. Combined with the per-(family,
        # difficulty) split stratification below, this keeps every split
        # representative on both the family and difficulty axes.
        family_difficulties = _difficulty_sequence(difficulty_mix, int(count))
        for family_index in range(int(count)):
            task_id = f"vlr_pilot_{task_index:06d}"
            target_tokens = int(target_context_tokens[(task_index - 1) % len(target_context_tokens)])
            task_seed = seed + task_index
            difficulty = family_difficulties[family_index]
            generator = _GENERATORS[task_family]
            generation_kwargs = resolve_generation_kwargs(
                task_family=task_family,
                difficulty=difficulty,
                generation_config=generation_config,
            )
            task = generator(
                task_id=task_id,
                seed=task_seed,
                target_context_tokens=target_tokens,
                **generation_kwargs,
            )
            tasks.append(task)
            task_index += 1

    _assign_splits(tasks, config["splits"])
    _assign_extra_splits(tasks, config.get("extra_splits", {}))
    return tasks


def _difficulty_sequence(difficulty_mix: dict[str, float], total_size: int) -> list[str]:
    if total_size <= 0:
        return []
    if not difficulty_mix:
        raise ValueError("difficulty mix must not be empty")

    items = list(difficulty_mix.items())
    for difficulty, _ in items:
        if difficulty not in {"easy", "medium", "hard"}:
            raise ValueError(f"unsupported_difficulty:{difficulty}")
    proportion_sum = sum(float(proportion) for _, proportion in items)
    if abs(proportion_sum - 1.0) > 0.01:
        raise ValueError(f"difficulty proportions must sum to 1.0, got {proportion_sum}")

    counts: list[tuple[str, int]] = []
    assigned = 0
    for index, (difficulty, proportion) in enumerate(items):
        if index == len(items) - 1:
            count = total_size - assigned
        else:
            count = int(total_size * float(proportion))
        counts.append((difficulty, count))
        assigned += count

    sequence: list[str] = []
    for difficulty, count in counts:
        sequence.extend([difficulty] * count)
    if len(sequence) != total_size:
        raise ValueError(f"difficulty sequence length {len(sequence)} does not match total {total_size}")
    return sequence


def _assign_splits(tasks: list[VeriLongTask], splits: dict[str, float]) -> None:
    """Assign train/dev/test stratified within each (family, difficulty) group.

    Tasks are generated grouped by family, so a naive contiguous slice would
    push whole families into a single split (e.g. dev/test ending up entirely
    temporal_update). Stratifying within each family keeps every split
    representative of the task mix; stratifying also by difficulty keeps every
    split representative of the difficulty mix, so difficulty cannot become a
    per-family/per-split confound.
    """

    split_items = list(splits.items())
    for group_indices in _group_indices_by_family_difficulty(tasks).values():
        group_total = len(group_indices)
        assigned = 0
        for split_index, (split_name, proportion) in enumerate(split_items):
            if split_index == len(split_items) - 1:
                count = group_total - assigned
            else:
                count = int(group_total * float(proportion))
            for task_index in group_indices[assigned : assigned + count]:
                tasks[task_index].metadata.split = split_name
            assigned += count


def _assign_extra_splits(tasks: list[VeriLongTask], extra_splits: dict[str, Any]) -> None:
    """Tag judge / live-demo subsets, stratified across task families.

    Taking the head of the list would put every subset task in the first
    family. Instead we round-robin across families so each subset stays mixed.
    """

    judge_subset_size = int(extra_splits.get("judge_subset_size", 0))
    live_demo_subset_size = int(extra_splits.get("live_demo_subset_size", 0))

    stratified_order = _stratified_order(tasks)
    for task_index in stratified_order[:judge_subset_size]:
        tasks[task_index].metadata.extra["judge_subset"] = True
    for task_index in stratified_order[judge_subset_size : judge_subset_size + live_demo_subset_size]:
        tasks[task_index].metadata.extra["live_demo_subset"] = True


def _group_indices_by_family(tasks: list[VeriLongTask]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, task in enumerate(tasks):
        groups.setdefault(task.task_family, []).append(index)
    return groups


def _group_indices_by_family_difficulty(tasks: list[VeriLongTask]) -> dict[tuple[str, str], list[int]]:
    groups: dict[tuple[str, str], list[int]] = {}
    for index, task in enumerate(tasks):
        groups.setdefault((task.task_family, task.difficulty), []).append(index)
    return groups


def _stratified_order(tasks: list[VeriLongTask]) -> list[int]:
    """Round-robin interleave task indices across families."""

    family_queues = list(_group_indices_by_family(tasks).values())
    ordered: list[int] = []
    position = 0
    while len(ordered) < len(tasks):
        progressed = False
        for queue in family_queues:
            if position < len(queue):
                ordered.append(queue[position])
                progressed = True
        if not progressed:
            break
        position += 1
    return ordered

