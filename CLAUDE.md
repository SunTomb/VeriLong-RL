# VeriLong-RL — Agent Guide

**A verifiable long-context benchmark for evidence-grounded reasoning and RLVR.**
Not a survey: the deliverable is a benchmark/dataset + decomposed evaluation +
SFT/RLVR experiments + an interactive demo, produced in a fixed time budget.

## Current state (end of Phase 1 pilot)

Phase 1 pilot loop is **complete and verified**: deterministic generation →
validation → parsing → metrics/reward → evaluation, plus API and open-source
model evals. `python -m pytest -q` → **39 passing**.

Done:
- 3 task families generate + validate (1200-task pilot in `data/pilot/tasks.jsonl`).
- Parser / metrics / programmatic reward; oracle & corrupted smoke baselines.
- **API eval** (`experiments/eval_api/`): Claude adapter + OpenAI-compatible
  client (used a Gemini proxy for real runs), caching, `--stratify`, fault tolerance.
- **Open-source eval** (`experiments/eval_open_source/`): Qwen2.5-7B-Instruct via
  transformers on the cluster.
- **Key result** (dev, stratified 30): reward flash-lite 0.945 / Gemini-pro 0.939
  / Qwen2.5-7B 0.751. Benchmark now discriminates. Qwen's weaknesses are
  over-citation (retrieval) and temporal reasoning — both programmatic reward
  signals, i.e. RLVR targets.

## Where things are

- Code (benchmark package, experiments, tests): repo root, tracked on `master`.
- **Design spec**: `VeriLong-RL/docs/superpowers/specs/2026-06-26-verilong-rl-design.md` (untracked).
- **Implementation plan** (Tasks 1-10, gates): `VeriLong-RL/docs/superpowers/plans/2026-06-26-phase1-pilot-implementation.md` (untracked). **Read this first.**
- Working memory: `memory/` (tracked) — split bug, eval findings, cluster notes.
- Older project memory: `VeriLong-RL/memory/` (untracked) — decisions, scope, cluster resources.

## Hard constraints (from the plan/spec — do not violate)

- **Phase gating**: Phase 1 (synthetic multi-doc evidence) is the only
  non-negotiable mainline. Phase 2 (real papers) / Phase 3 (repo-level) are
  **blocked until the user explicitly approves**.
- Data scale: Pilot 1K-2K → Core 10K-20K → Full 50K+. Never jump to Full.
- Fixed output format: `Evidence:` / `Steps:` / `Answer:` (see spec).
- Task families: `anti_distractor_retrieval`, `multi_hop_reasoning`, `temporal_update`.
- Reward is programmatic; LLM judge is for data filtering / calibration / analysis, NOT online RL reward.
- Never fabricate metrics. Experimental numbers come from real runs only.
- Splits must be stratified per family (a bug where dev/test were all
  temporal_update was fixed — see `memory/pilot-split-not-stratified-bug.md`).

## Environment & cluster

- Local dev: Python 3.11+, pydantic v2, pytest. `python -m pytest -q`.
- Cluster: SSH alias `Tang-2-Wu` / `Song-3-Wu` (IP 210.45.70.34, user wujcan).
  **GPUs are shared — `nvidia-smi` first, pick the lowest-memory card, pin
  `CUDA_VISIBLE_DEVICES`.** Details + the torch/driver fix in
  `memory/cluster-gpu-usage-tang-song.md`.
- Env: shared conda `gmsra` (`/NAS/yesh/miniconda3`), torch downgraded to
  cu121 for the CUDA-12.2 driver. Project entry: `/NAS/yesh/VeriLong-RL/activate.sh`.
  Do not create duplicate envs.
- Models cached at `/NAS/yesh/hf_cache/hub` (use repo ids, offline).
- Code sync to cluster: `git archive HEAD | scp` to `/tmp`, extract to
  `/NAS/yesh/VeriLong-RL` (no rsync on the Windows client).

## Conventions

- Match existing code style. Tests live in `tests/benchmark/`.
- Raw API outputs / caches go under `results/raw/` (gitignored).
- Commit only when asked. Branch off, don't commit to master directly unless told.
