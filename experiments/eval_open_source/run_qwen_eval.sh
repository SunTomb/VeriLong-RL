#!/usr/bin/env bash
# Launch VeriLong-RL open-source eval on a single GPU.
#
# Usage (on a GPU node, repo at /NAS/yesh/VeriLong-RL):
#   CUDA_VISIBLE_DEVICES=0 bash experiments/eval_open_source/run_qwen_eval.sh dev 30
#
# Args: SPLIT (default dev), LIMIT (default 30).
# Pick a free GPU index from nvidia-smi and pin it via CUDA_VISIBLE_DEVICES.
# Environment (conda gmsra + HF cache/offline + PYTHONPATH) comes from the
# project activate.sh, matching the MemUpdateBench convention.
set -euo pipefail

SPLIT="${1:-dev}"
LIMIT="${2:-30}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
OUT="results/raw/open_source/qwen2_5_7b_instruct_${SPLIT}${LIMIT}.jsonl"

# shellcheck disable=SC1091
source /NAS/yesh/VeriLong-RL/activate.sh

echo "[run_qwen_eval] gpu=$CUDA_VISIBLE_DEVICES model=$MODEL split=$SPLIT limit=$LIMIT"
python experiments/eval_open_source/run_hf_eval.py \
  --tasks data/pilot/tasks.jsonl \
  --split "$SPLIT" --limit "$LIMIT" --stratify \
  --model "$MODEL" \
  --out "$OUT"

echo "[run_qwen_eval] scoring"
python scripts/score_outputs.py \
  --tasks data/pilot/tasks.jsonl \
  --outputs "$OUT" \
  --scored "${OUT%.jsonl}.scored.jsonl" \
  --summary "${OUT%.jsonl}.summary.json"

echo "[run_qwen_eval] done -> ${OUT%.jsonl}.summary.json"
