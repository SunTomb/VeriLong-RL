---
name: pilot-eval-findings-gemini
description: VeriLong-RL Phase 1 首次真实 API 评测（Gemini 代理）结果与发现的指标设计问题
metadata:
  type: project
---

在 worktree `worktree-phase1-pilot` 上用临时 Gemini 代理（OpenAI 兼容网关）完成 Phase 1 首次真实 API 评测。模型 `a/gemini-3.1-flash-lite`（弱）与 `a/gemini-3.1-pro`（强），dev split 分层采样 30 条（每族 10）。

**对比结果（overall）：** 两模型 answer_normalized_match 均 1.0；flash-lite reward_total 0.945、format_valid 1.0；pro reward_total 0.889、format_valid 0.833。强模型 reward 反而更低。

**根因发现（重要的指标设计问题）：** parser/format 规则硬编码要求推理步数 2–4（`benchmark/parser/output_parser.py:27`、`benchmark/metrics/format.py:15`），与任务族无关。pro 在 5 个 `anti_distractor_retrieval`（单跳检索）任务上给出**正确答案 + 正确证据 + 单步推理**，被判 `invalid_step_count:1` 而 format 无效、reward 被扣。flash-lite 因机械输出两步（含填充句）反而拿满分。

**结论：** 强制 ≥2 步对单跳检索任务不合理，会惩罚简洁正确的回答。这不是模型缺陷，是 reward/format 规则与任务族不匹配。需要决定：按任务族设置最小步数（retrieval 允许 1 步，multihop/temporal 要求 ≥2），还是放宽全局下限。修改后需重跑评测。

**已修复（提交 `df6d8e2`）：** 用 `VeriLongTask.step_bounds()` 按结构推导步数区间：retrieval 1-2、multihop 2..hop_count+2、temporal 2..update_count+2。parser 改为任务无关（去掉 invalid_step_count flag），步数判定移到 `format_scores(parsed, task)`。注意第二次迭代：最初把 multihop min 设为 hop_count，结果 Gemini 把 3 跳压缩成 2 步（证据全引用、答案对）又被误判——用户确认放宽到 min=2，多跳覆盖度交给 citation recall 把关。

**最终对比（dev 分层 30，每族 10）：** 两模型 answer/format/citation 几乎持平，reward flash-lite 0.945 vs pro 0.939，仅 pro 的 stale_citation_rate 略高（0.033 vs 0.017）。**两模型答对率 100%，说明 pilot 难度对前沿模型偏低、缺乏区分度——下一步应上调难度（干扰项强度、hop 数、context 长度）。**

**三方对比（加入开源 Qwen2.5-7B-Instruct，Tang-2 GPU7 transformers 推理）：** benchmark 终于有区分度。reward：flash-lite 0.945 / pro 0.939 / Qwen 0.751。Qwen 失败模式按族分化：
- multihop reward 0.95、cit_precision 1.0 —— 与前沿模型持平，多跳推理没问题。
- retrieval reward 0.79、cit_precision 0.50 —— 严重过度引用，一半引用是干扰项（distractor_rate 0.167、overcitation 0.25）；recall 1.0 说明"找得到证据但管不住嘴"。
- temporal reward 0.51 —— 最弱，引用过期证据 + 答案常错（拉低整体 answer 到 0.70）。

**对 RLVR 的意义：** 开源小模型瓶颈在"引用纪律"和"时序推理"，而非证据检索；这两者都是程序化可计算的奖励信号（distractor/stale citation rate），正是 RLVR 可直接优化的靶点。结果文件本地在 `results/raw/open_source/qwen2_5_7b_instruct_dev30.*`。

**Why:** 这正体现了 verifiable benchmark 的价值——能暴露"答对但不符合可验证约束"的差异；但也说明 format 规则本身必须与任务语义对齐，否则会产生误导性的强弱排序。

**工程产物（已提交于该分支）：**
- `91d6edc` Task 7 Claude adapter
- `4c42e4e` split 分层修复 + 重生成数据
- `70d6ca6` OpenAI 兼容 client + `--stratify`
- `daa40a5` runner 逐条容错 + 增量写盘
原始结果在 `results/raw/api/`（被 .gitignore 忽略，未入库）。

Related: [[pilot-split-not-stratified-bug]], [[project-decisions]]
