from collections.abc import Iterable
from typing import Any


def mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


def aggregate_numeric(records: list[dict[str, Any]], field_names: list[str]) -> dict[str, float]:
    return {f"{field_name}_mean": mean(float(record.get(field_name, 0.0)) for record in records) for field_name in field_names}
