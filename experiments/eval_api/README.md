# API Evaluation Adapter Gate

This directory documents the API evaluation contract for VeriLong-RL Phase 1. Do not add API-calling code until Tasks 1–6 pass locally and the first run is explicitly scoped to a small development subset.

## Entry gate

API evaluation may start only after:

1. `python -m pytest tests/benchmark -q` passes.
2. `python scripts/generate_pilot.py --config configs/pilot.yaml` produces validated pilot data.
3. `python scripts/validate_pilot.py data/pilot/tasks.jsonl` reports `validated=1200 valid=1200 invalid=0`.
4. `oracle_format_baseline` and `corrupted_distractor_baseline` smoke runs produce real `summary.json` files under `results/pilot/`.

First API run must be a small dev subset:

```bash
python experiments/eval_api/run_api_eval.py \
  --tasks data/pilot/tasks.jsonl \
  --split dev \
  --limit 30 \
  --model claude-opus-4-8 \
  --out results/raw/api/claude_opus_4_8_pilot_dev30.jsonl
```

The output file is then scored by the normal parser/metrics path. Do not fabricate metrics if the API run fails or is skipped.

## Claude API constraints

Default implementation constraints:

- Default strong Claude model: `claude-opus-4-8`.
- Adaptive thinking: `thinking={"type": "adaptive"}`.
- Do not set `budget_tokens` on Opus 4.8/4.7/Fable 5.
- Do not set `temperature`, `top_p`, or `top_k` on Opus 4.8/4.7/Fable 5.
- Use streaming for long context input or large `max_tokens`.
- Use `output_config.format` for structured outputs when structured output is needed; do not use deprecated `output_format`.
- Do not use assistant prefill to force JSON or output shape.
- Offline batch eval should use queue/retry/cache.
- Live demo should stay small and provide cached fallback.

## Prompt contract

The model prompt must request evidence-grounded short rationales, not hidden chain-of-thought:

```text
You are solving a VeriLong-RL evidence-grounded long-context task.
Return exactly this format:
Evidence: E01, E02
Steps:
1. One short sentence grounded in cited evidence.
2. One short sentence grounded in cited evidence.
Answer: final answer only
```

Required task prompt fields:

- task id
- question
- full document list with `doc_id`, `evidence_id`, and text
- required output format

The prompt must not expose gold answer, gold evidence IDs, distractor IDs, stale IDs, or metric labels.

## Cache design

Cache key fields:

```text
task_id
model
prompt_version
task_hash
```

Cache file path:

```text
results/raw/api_cache/{model}/{task_id}.json
```

Each cache record should include:

```json
{
  "task_id": "vlr_pilot_000001",
  "model": "claude-opus-4-8",
  "prompt_version": "phase1-v1",
  "task_hash": "sha256-of-task-prompt-input",
  "output_text": "Evidence: E01\nSteps:\n1. ...\n2. ...\nAnswer: ...",
  "raw_response_metadata": {},
  "created_at": "filled-by-runner"
}
```

## Output JSONL contract

API output rows must match the same scoring contract used by local baselines:

```json
{
  "task_id": "vlr_pilot_000001",
  "model": "claude-opus-4-8",
  "output_text": "Evidence: E01\nSteps:\n1. E01 states the relevant fact.\n2. Therefore the answer is A17.\nAnswer: A17",
  "metadata": {"source": "api", "prompt_version": "phase1-v1"}
}
```

The scorer remains responsible for parser errors, invalid evidence IDs, answer matching, citation metrics, and reward calculation.
