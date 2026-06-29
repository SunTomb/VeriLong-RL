# Open-Source Model Evaluation

Evaluates an open-source HF model (default `Qwen2.5-7B-Instruct`) on the
VeriLong-RL pilot benchmark, using the **same prompt and output contract** as
the API eval path (`experiments/eval_api/`), so outputs are scored by the same
`scripts/score_outputs.py` parser/metrics pipeline.

## Entry gate

Only run open-source eval after pilot parser/metrics are stable (Tasks 1-6 +
step-bound fix). First model and scope per plan Task 8:

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Context: 8K/16K pilot dev/test subset first; 32K/64K are stretch.

## Cluster facts (observed)

- Node: any free GPU node (e.g. **Tang-2-Wu**, 8× A40 46GB; **Song-3-Wu**, 8×
  A100-80GB). Pick a free GPU from `nvidia-smi`.
- Environment: this project reuses the shared **`gmsra`** conda env
  (`/NAS/yesh/miniconda3`), which has torch 2.12 + transformers 4.46.3 (no
  vLLM — this runner uses transformers). Following the MemUpdateBench
  convention, project setup lives in `/NAS/yesh/VeriLong-RL/activate.sh`
  (activates gmsra, sets `HF_HUB_CACHE`/`HF_HUB_OFFLINE`/`PYTHONPATH`). Do not
  create a duplicate env; `pydantic`/`pyyaml` were added into gmsra once.
- Model cache: `/NAS/yesh/hf_cache/hub` (repo id `Qwen/Qwen2.5-7B-Instruct`
  resolves offline from here).
- NAS / repo sync target: `/NAS/yesh/VeriLong-RL`.

## Run

Wiring test (no GPU, offline):

```bash
python experiments/eval_open_source/run_hf_eval.py \
  --tasks data/pilot/tasks.jsonl --split dev --limit 6 --stratify --fake \
  --out results/raw/open_source/_wiring.jsonl
```

Real run on a GPU node (single GPU):

```bash
source /NAS/yesh/VeriLong-RL/activate.sh
CUDA_VISIBLE_DEVICES=0 bash experiments/eval_open_source/run_qwen_eval.sh dev 30
```

The launcher sources `activate.sh`, runs stratified eval across all three task
families, and scores the output.

## Contract

Output rows match the shared scoring contract:

```json
{"task_id": "...", "model": "...", "output_text": "Evidence: ...\nSteps:\n...\nAnswer: ...",
 "metadata": {"source": "open_source", "provider": "transformers", "prompt_version": "phase1-v1"}}
```

Decoding is greedy (`do_sample=False`) for repeatable, grounded answers. Do not
fabricate metrics if a run fails; record the failure instead.
