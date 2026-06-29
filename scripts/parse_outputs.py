import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.parser.output_parser import parse_model_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse VeriLong-RL model output JSONL.")
    parser.add_argument("input_path", help="JSONL with task_id, output, and valid_evidence_ids fields.")
    parser.add_argument("output_path", help="Path to write parsed prediction JSONL.")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parsed_count = 0
    with input_path.open("r", encoding="utf-8") as input_handle, output_path.open("w", encoding="utf-8") as output_handle:
        for line_number, line in enumerate(input_handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            parsed = parse_model_output(
                str(record.get("output", "")),
                valid_evidence_ids=set(record.get("valid_evidence_ids", [])),
            )
            output_record = {
                "task_id": record.get("task_id"),
                "line_number": line_number,
                "parsed": parsed.model_dump(mode="json"),
            }
            output_handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            parsed_count += 1

    print(f"parsed={parsed_count} output={output_path}")


if __name__ == "__main__":
    main()
