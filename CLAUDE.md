# VeriLong-RL — Agent Guide

**A verifiable long-context benchmark for evidence-grounded reasoning and RLVR.**
Deliverable: benchmark/dataset + decomposed evaluation + SFT/RLVR experiments + interactive demo.

## Current state (post Phase 1 + web demo)

All Phase 1 work is **complete and on `main`**. `python -m pytest -q` → **79 passing**.

### Done
- **3 task families** generate + validate: `anti_distractor_retrieval`, `multi_hop_reasoning`, `temporal_update`.
  - Pilot: 1200-task `data/pilot/tasks.jsonl` (LFS-tracked); hard slice `data/pilot/hard_tasks.jsonl`.
- **Hard difficulty system** (`configs/hard.yaml`, `benchmark/generator/profiles.py`): configurable distractor count, hop count, stale count, (family, difficulty) stratified split.
- **Parser / metrics / programmatic reward**; oracle & corrupted smoke baselines.
- **API eval** (`experiments/eval_api/`): Claude adapter + OpenAI-compatible client, caching, `--stratify`, fault tolerance.
- **Open-source eval** (`experiments/eval_open_source/`): Qwen2.5-7B-Instruct via transformers + optional LoRA adapter.
- **SFT warmup** (`experiments/sft/`): LoRA SFT v1 (pure gold) + v2 (Claude-distilled) + v2-gemini (Gemini-distilled). Key result: dev30 saturates; hard slice discriminates. gemini-v2 hard reward 0.950 matches top API models.
- **RLVR GRPO pipeline** (`experiments/rlvr/`): trl 0.14 GRPOTrainer, smoke-validated end-to-end. Full 14K-token training **deferred** — needs 80GB A100 (46GB A40 OOMs on logits tensor). Config pinned in `experiments/rlvr/configs/grpo_v1.yaml`.
- **Web demo** (`web/`): FastAPI backend (offline-first, reuses benchmark scorer) + React/Vite frontend. Chinese UI, academic visual style. Local: `uvicorn web.backend.app:app` + `npm run dev`.
- **GitHub**: https://github.com/SunTomb/VeriLong-RL (LFS for JSONL data, main branch).
- **Phase 2 design spec** at `docs/superpowers/specs/2026-06-28-phase2-real-paper-design.md`.

### Key empirical results (real runs, not fabricated)
- **API leaderboard** (dev30): Gemini 3.1 Pro 0.939 > gpt-5.4-mini 0.929 > Gemini 3.5 Flash 0.895 > claude-opus-4-6 0.891 > gpt-5.5 0.860 > claude-opus-4-8 0.844
- **API hard54** (16K+32K context): Gemini Pro/Flash 0.950, gpt-5.5 0.938, gpt-5.4-mini 0.909, claude-opus-4-6/4-8 0.885
- **SFT comparison** (hard slice): base 0.685 → v1-gold 0.891 → v2-claude 0.906 → v2-gemini 0.950

## Where things are (all paths relative to repo root `D:\USTC\2026Summer\面试`)

- Code + tests: repo root, `main` branch, pushed to GitHub.
- Specs: `docs/superpowers/specs/` (tracked).
- Plans: `docs/superpowers/plans/` (tracked).
- Working memory: `memory/` (tracked) — see MEMORY.md index.
- Raw outputs/caches: `results/raw/` (gitignored).

## Phase gating

- **Phase 1** (synthetic multi-doc evidence): ✅ complete.
- **Phase 2** (real papers): ✅ **USER HAS EXPLICITLY APPROVED.** Design spec at `docs/superpowers/specs/2026-06-28-phase2-real-paper-design.md`. Next step: implementation plan → pilot ingestion.
- **Phase 3** (repo-level): ❌ blocked until explicit user approval.
- Data scale: Pilot 1K-2K → Core 10K-20K → Full 50K+. Never jump to Full.

## Hard constraints (do not violate)

- Fixed output format: `Evidence:` / `Steps:` / `Answer:` (parser must pass unchanged).
- Reward is programmatic; LLM judge is for filtering/calibration only — NOT online RL reward.
- Never fabricate metrics. All numbers come from real runs.
- Splits must be stratified per (family, difficulty) — see `memory/pilot-split-not-stratified-bug.md`.
- API keys via environment variables only, never written to files or returned to browser.
- Commit only when asked. Branch off `main`, never commit to main directly unless told.

## Environment & cluster

- **Local dev**: Python 3.11+, pydantic v2, pytest. `python -m pytest -q`. Web: `fastapi`, `uvicorn`, Node 25 + npm.
- **Cluster**: SSH alias `Tang-2-Wu` / `Song-3-Wu` (IP 210.45.70.34, user wujcan).
  GPUs are shared — `nvidia-smi` first, pin `CUDA_VISIBLE_DEVICES`. Details in `memory/cluster-gpu-usage-tang-song.md`.
- **Env**: shared conda `gmsra` (`/NAS/yesh/miniconda3`), torch 2.5.1+cu121. RLVR uses separate `verilong_rl` env (gmsra clone + trl 0.14 + vllm 0.6.6.post1).
- **Project entry on cluster**: `/NAS/yesh/VeriLong-RL/activate.sh`. Do not create duplicate envs.
- **Models**: cached at `/NAS/yesh/hf_cache/hub`. Use repo IDs offline (`TRANSFORMERS_OFFLINE=1`).
- **Code sync**: `git archive HEAD | ssh wujcan@210.45.70.34 tar -xf - -C /NAS/yesh/VeriLong-RL` (no rsync on Windows).

## Conventions

- Tests live in `tests/benchmark/` (Python) and `web/frontend/src/__tests__/` (TypeScript).
- Raw API outputs / caches → `results/raw/` (gitignored).
- Web raw outputs → `results/raw/` (not `web/`).
- No fabricated numbers; no `VeriLong-RL/` subfolder changes (untracked docs live there as scratch space).
