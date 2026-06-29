# 新对话启动提示词

下面这段可以直接复制到新的 Claude Code / Agent 对话中使用。

```text
请你在 `D:\USTC\2026Summer\面试\VeriLong-RL` 中继续 VeriLong-RL 项目。

本项目的定位是：VeriLong-RL: A Verifiable Long-Context Benchmark for Evidence-Grounded Reasoning and RLVR。它是一个 Long Context data strategy / survey 最终考核项目，但目标不是只做综述，而是在固定时间内完成一个 benchmark + experiments + interactive demo 的完整产出。

请先阅读以下文件，不要直接开始写工程代码：

1. `docs/superpowers/specs/2026-06-26-verilong-rl-design.md`：完整设计文档。
2. `memory/MEMORY.md`：本项目本地记忆索引。
3. `memory/long-context-experiment-scope.md`：阶段门控，Phase 1 必须先做合成多文档证据任务。
4. `memory/cluster-resources.md`：Tang/Song GPU 集群、NAS 路径与已缓存模型。
5. `memory/project-decisions.md`：已确认的 benchmark、数据、训练、评测与 demo 决策。
6. `docs/handoff/prior-session-tail-extract.txt`：上一对话未完成任务与迁移说明。

当前状态：

- 设计文档已写入 `docs/superpowers/specs/2026-06-26-verilong-rl-design.md`。
- 本地项目记忆已迁移到 `memory/`。
- 当前只完成了设计与交接材料；还没有生成完整工程脚手架。
- 下一步应该先写 implementation plan，而不是马上大规模写代码。

必须遵守的项目约束：

1. 总路线采用“方案 B：Benchmark + Interactive Demo 双核心”。
2. Phase 1 是唯一不能牺牲的主线：synthetic multi-document evidence benchmark。
3. Phase 2 真实论文/技术报告任务与 Phase 3 代码仓库级任务只能在 Phase 1 完成后，再询问用户是否继续。
4. Phase 1 核心任务族为：anti-distractor retrieval、multi-hop evidence reasoning、conflict / temporal update reasoning。
5. 模型输出格式采用 Answer + 极简结构化推理 + cited evidence IDs。
6. 数据生成采用规则结构/gold labels + LLM 改写 + 程序校验 + judge 抽样审计。
7. 数据规模目标是 Pilot 1K–2K → Core 10K–20K → Full 50K+，不要一开始就跳到 Full。
8. 训练主线默认 Qwen2.5-7B / Qwen2.5-7B-Instruct LoRA/QLoRA SFT，然后尝试 GRPO/RLVR。
9. LLM judge/verifier 用于数据过滤、reward 校准、preference pair 构造和最终分析，不作为在线 RL 主 reward。
10. Live demo 默认只调用 API 模型；开源训练模型结果以 cached/offline results 展示。
11. 不要伪造实验结果；design doc 里的 leaderboard `value` 只是未来实验完成后填充的占位说明。
12. 不要在 implementation plan 批准前创建完整工程脚手架。

请你接下来做三件事：

1. 先简要复述你从设计文档和记忆中理解到的项目目标、阶段门控和当前状态。
2. 然后为 Phase 1 编写一份可执行 implementation plan，覆盖 dataset schema/generator、parser/metrics/reward、pilot eval、SFT/RLVR、backend/frontend demo、presentation artifacts。
3. 在计划中优先拆出 Milestone 1：Pilot 闭环，并列出第一批应该创建的目录、文件和验证命令。

请保持中文沟通，必要的技术名词可用英文。请按现有设计推进，不要重新发散项目方向。
```
