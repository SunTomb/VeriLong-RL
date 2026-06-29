import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.generator.build_pilot import build_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the VeriLong-RL Phase 1 pilot JSONL.")
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()

    result = build_pilot(args.config)
    print(f"generated={result['generated']} valid={result['valid']} output={result['output']}")


if __name__ == "__main__":
    main()

