"""Evaluate an open-source HF model (default Qwen2.5-7B-Instruct) on VeriLong-RL.

Runs locally on a GPU node (e.g. Song-3 A100). It reuses the exact prompt and
output contract from the API eval path so results are scored by the same
parser/metrics pipeline:

    python experiments/eval_open_source/run_hf_eval.py \
        --tasks data/pilot/tasks.jsonl \
        --split dev --limit 30 --stratify \
        --model /NAS/yesh/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct \
        --out results/raw/open_source/qwen2_5_7b_instruct_dev30.jsonl

Design choices:

- Uses transformers (already present in the cluster env); no vLLM dependency.
- Greedy decoding (do_sample=False) for repeatable grounded answers.
- Chat template applied via the tokenizer so the system/user contract matches
  the API path.
- ``--max-tasks-smoke`` / ``--fake`` allow validating wiring without a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.eval_api.run_api_eval import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_user_prompt,
    load_tasks,
)


def build_chat_messages(user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _fake_output(task: Any) -> str:
    """Offline stub for wiring tests (no GPU). Echoes gold; tagged fake."""

    evidence = task.gold_evidence_ids or ["E01"]
    lines = [f"Evidence: {', '.join(evidence)}", "Steps:"]
    for i, eid in enumerate(evidence, start=1):
        lines.append(f"{i}. {eid} supports the answer.")
    if len(evidence) < 2:
        lines.append("2. Therefore the answer follows.")
    lines.append(f"Answer: {task.gold_answer}")
    return "\n".join(lines)


def run(
    tasks_path: Path,
    out_path: Path,
    model_path: str,
    split: str | None,
    limit: int | None,
    stratify: bool,
    max_new_tokens: int,
    fake: bool,
    adapter_path: str | None = None,
) -> dict[str, Any]:
    tasks = load_tasks(tasks_path, split=split, limit=limit, stratify=stratify)
    if not tasks:
        raise SystemExit(f"no tasks matched split={split!r}")

    model_name = Path(model_path).name if not fake else "fake-open-source"
    if not fake and adapter_path:
        model_name = f"{model_name}+{Path(adapter_path).name}"
    generate = None if fake else _load_generator(model_path, max_new_tokens, adapter_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            user_prompt = build_user_prompt(task)
            if fake:
                output_text = _fake_output(task)
            else:
                output_text = generate(build_chat_messages(user_prompt))
            row = {
                "task_id": task.id,
                "model": model_name,
                "output_text": output_text,
                "metadata": {
                    "source": "fake" if fake else "open_source",
                    "provider": "transformers",
                    "prompt_version": PROMPT_VERSION,
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            n += 1

    return {"tasks": n, "model": model_name, "out": str(out_path), "fake": fake}


def _load_generator(model_path: str, max_new_tokens: int, adapter_path: str | None = None):
    """Load the HF model once and return a closure that generates text."""

    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if adapter_path:
        from peft import PeftModel  # noqa: PLC0415

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    def generate(messages: list[dict[str, str]]) -> str:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    return generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an open-source HF model on VeriLong-RL.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF repo id (resolved from HF_HOME cache when TRANSFORMERS_OFFLINE=1) or a local path.",
    )
    parser.add_argument("--adapter", default=None, help="Optional peft LoRA adapter path to load on top of --model.")
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stratify", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--fake", action="store_true", help="Offline wiring test; no GPU/model load.")
    args = parser.parse_args()

    summary = run(
        tasks_path=Path(args.tasks),
        out_path=Path(args.out),
        model_path=args.model,
        split=args.split,
        limit=args.limit,
        stratify=args.stratify,
        max_new_tokens=args.max_new_tokens,
        fake=args.fake,
        adapter_path=args.adapter,
    )
    print(f"model={summary['model']} tasks={summary['tasks']} fake={summary['fake']} out={summary['out']}")


if __name__ == "__main__":
    main()
