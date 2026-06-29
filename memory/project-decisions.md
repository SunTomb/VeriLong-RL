---
name: project-decisions
description: VeriLong-RL 已确认的 benchmark、数据、训练、评测与 demo 设计决策
metadata:
  type: project
  originSessionId: 382b2af7-64ab-42e1-b374-e44bd8f47bc5
  migratedAt: 2026-06-26
---

VeriLong-RL 的正式定位是：**A verifiable long-context benchmark for evidence-grounded reasoning and RLVR**。已确认采用“方案 B：Benchmark + Interactive Demo 双核心”，即同时交付 benchmark/dataset、系统化评测、SFT+RLVR 训练实验、dashboard 与 live API demo。

已确认关键决策：

1. 项目独立于用户原有研究项目；原项目只提供集群与服务器调用经验。
2. 总方向为 Long Context RL-style verifiable task design，Evidence use / grounding / anti-distractor 为副方向。
3. 阶段门控采用混合场景：Phase 1 必须先完成 synthetic multi-document evidence tasks；Phase 2/3 的真实论文/技术报告与代码仓库级任务只能在 Phase 1 完成后由用户再次决定。
4. 输出格式采用 `Answer + 极简结构化推理 + cited evidence IDs`，用于 parser、metrics、dashboard、SFT target 与 RLVR reward。
5. Phase 1 核心任务族为 anti-distractor retrieval、multi-hop evidence reasoning、conflict / temporal update reasoning。
6. 数据规模目标为 50K+ tasks，但必须 Pilot 1K–2K → Core 10K–20K → Full 50K+ 分阶段推进。
7. 数据生成采用规则生成结构与 gold labels + LLM 改写自然语言/干扰项 + 程序校验 + judge 抽样审计的混合 pipeline。
8. 评测必须拆分 answer correctness、citation correctness、reasoning consistency、distractor/stale evidence use、format validity 等指标，不能只看 final answer。
9. 训练策略为 Qwen2.5-7B / Qwen2.5-7B-Instruct 的 LoRA/QLoRA SFT warmup，再尝试 GRPO/RLVR；SFT 是主线，RLVR 是进攻项。
10. Reward 采用程序化 reward 为主；LLM judge/verifier 用于数据过滤、reward 校准、preference pair 构造和最终分析，不作为在线 RL 主 reward。
11. 离线评测采用 API frontier models 与开源 7B line 对称对比；live demo 默认只调用 API 模型，开源训练结果以 cached/offline results 展示。
12. Web 形态是 VeriLong-RL Benchmark Portal，包含 overview、dataset/task generator、leaderboard、breakdown analysis、case study viewer、training results 和 live demo。
13. 当前阶段只应创建文档与记忆；完整工程脚手架应在 implementation plan 经确认后再生成。

**Why:** 这些决策保证项目既符合 Long Context data strategy / survey 考核题目，又能体现个人实验设计、训练验证与工程展示能力。

**How to apply:** 新 Agent 接手时应先阅读 `docs/superpowers/specs/2026-06-26-verilong-rl-design.md`，再写 implementation plan；不要跳过 Pilot 闭环，不要提前扩展 Phase 2/3，不要直接创建完整代码工程。

Related: [[long-context-experiment-scope]], [[cluster-resources]]
