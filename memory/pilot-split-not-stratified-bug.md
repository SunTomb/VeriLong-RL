---
name: pilot-split-not-stratified-bug
description: build_pilot 的 _assign_splits 未按任务族分层，导致 dev/test split 全是 temporal_update
metadata:
  type: project
---

`benchmark/generator/build_pilot.py` 的 `_assign_splits`（约 90-101 行）按 tasks 列表的**连续顺序**切分 train/dev/test。但任务是按族分块生成的（先全部 anti_distractor_retrieval，再 multi_hop_reasoning，最后 temporal_update），所以连续切分让：

- train = 前 840 条（retrieval 400 + multihop 400 + temporal 40，偏斜）
- dev = 120 条**全是 temporal_update**
- test = 240 条**全是 temporal_update**

后果：dev/test 评测 split 完全没有 retrieval / multihop 任务，跨族强弱模型对比无法在 dev/test 上进行；任何只看 dev/test 的指标都只反映 temporal_update 单族。

**Why:** 这是 Phase 1 pilot benchmark 的设计缺陷，发现于在 worktree `worktree-phase1-pilot`（分支 HEAD 含 Task 7 提交 `91d6edc`）上用 Gemini 代理跑真实 API 评测时。

**How to apply:** 修复方案是让 split 在**每个任务族内部**按比例分层切分（stratified split），使每个 split 都保持三族混合比例。修复后必须重新生成 `data/pilot/tasks.jsonl`、重跑 validator 与 pilot smoke、并重跑任何已有的 API 评测（dev split 内容会变）。`_assign_extra_splits` 的 judge/live_demo 子集同理也是取列表头部，可能需要一并审视。

Related: [[project-decisions]]
