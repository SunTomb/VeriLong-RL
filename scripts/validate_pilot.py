import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError

from benchmark.schemas.task import VeriLongTask
from benchmark.validators.task_validator import validate_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a VeriLong-RL pilot JSONL file.")
    parser.add_argument("jsonl_path")
    args = parser.parse_args()

    validated = 0
    valid = 0
    invalid = 0

    with open(args.jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            validated += 1
            try:
                task = VeriLongTask.model_validate(json.loads(line))
                report = validate_task(task)
            except (json.JSONDecodeError, ValidationError) as error:
                invalid += 1
                print(f"invalid_line={validated} errors={error}", file=sys.stderr)
                continue
            if report.valid:
                valid += 1
            else:
                invalid += 1
                print(f"invalid_task={task.id} errors={','.join(report.errors)}", file=sys.stderr)

    print(f"validated={validated} valid={valid} invalid={invalid}")
    if invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

