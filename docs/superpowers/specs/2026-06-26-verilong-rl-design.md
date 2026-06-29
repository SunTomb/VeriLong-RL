# VeriLong-RL 设计文档

日期：2026-06-26  
状态：已完成设计稿，待实施计划拆解  
项目路径：`D:\USTC\2026Summer\面试\VeriLong-RL`  
工作名：**VeriLong-RL: A Verifiable Long-Context Benchmark for Evidence-Grounded Reasoning and RLVR**

---

## 1. 背景与研究定位

本项目面向 2026 年夏季最终考核中的 Long Context data strategy / survey 主题。项目不应只是论文综述，也不应绑定用户既有研究项目；它应体现固定时间内从调研、提出问题、构造实验、训练验证到工程展示的独立产出能力。

核心定位是：

> **VeriLong-RL 是一个面向 evidence-grounded reasoning 与 RLVR 的可验证长上下文基准与交互式展示系统。**

它关注的问题不是“哪个模型在长上下文中分数最高”，而是：

1. 模型是否真正使用了正确证据，而不是只给出看似正确的最终答案；
2. 长上下文任务能否被设计成足够可验证、可归因、可用于 RLVR；
3. SFT 与 RLVR/GRPO 分别改善了哪些能力：格式遵循、证据引用、答案正确性、抗干扰、推理一致性；
4. 什么样的 long-context task family 与 reward signal 更适合训练 evidence-grounded reasoning。

项目最终包装为 **benchmark 项目形态**，同时包含可交互 live demo。推荐英文一句话定义：

> A verifiable long-context benchmark for evidence-grounded reasoning and RLVR.

---

## 2. 总体方案选择

前序 brainstorming 已确定采用 **方案 B：Benchmark + Interactive Demo 双核心**。

### 2.1 为什么选方案 B

方案 B 同时保留科研深度、实验完整性和工程展示能力：

- Benchmark / dataset 主体体现独立任务设计与评测严谨性；
- API + 开源 7B 对称评测体现模型分析能力；
- Qwen2.5-7B SFT + RLVR/GRPO 训练体现硬核实验能力；
- Dashboard + live demo 体现工程落地与现场展示冲击力；
- 如果 RLVR 训练效果不显著，benchmark、API 评测、SFT、dashboard 仍能形成完整闭环。

### 2.2 总交付物

最终交付物分为五类：

1. **Survey / Presentation**
   - 15–20 分钟汇报；
   - 前半部分总结 Long Context data strategy 与 RLVR 相关脉络；
   - 后半部分介绍 VeriLong-RL 的任务、实验、训练与 demo；
   - 重点体现“从文献洞察提出可验证任务设计”的个人思考。

2. **Benchmark / Dataset**
   - Phase 1 目标为 50K+ synthetic multi-document tasks；
   - 三个核心任务族：anti-distractor retrieval、multi-hop evidence reasoning、conflict / temporal update；
   - 每个样例带 gold answer、gold evidence IDs、distractor/stale evidence IDs、expected short reasoning steps 和 difficulty metadata。

3. **Evaluation / Metrics**
   - 分解评估 answer correctness、citation correctness、reasoning consistency、distractor contamination、stale evidence use、format validity；
   - 支持 task family、context length、evidence position、distractor strength、training stage 等维度切片。

4. **Training Experiments**
   - 开源线：Qwen2.5-7B / Qwen2.5-7B-Instruct / SFT / SFT+RLVR；
   - 训练采用 LoRA/QLoRA SFT warmup，再尝试 GRPO/RLVR；
   - LLM judge 用于数据过滤、reward 校准、偏好对构造和最终评测分析，不作为主在线 RL reward。

5. **Web Demo / Dashboard**
   - 离线结果 dashboard；
   - case study viewer；
   - online live demo：现场选择任务参数、实时生成一个小任务、调用 API 模型、解析答案和证据引用，并显示与 gold evidence 的对齐结果；
   - live demo 默认只调用 API 模型，开源训练模型结果以 cached/offline results 展示。

---

## 3. Scope 与阶段门控

本项目采用“混合场景”总方向，但必须阶段化推进。

### Phase 1：合成多文档证据任务主线（必做）

Phase 1 是唯一不能被牺牲的主线。必须先完成：

- synthetic multi-document evidence benchmark；
- 三个核心任务族；
- pilot/core/full 数据生成；
- parser、metrics、reward；
- API + Qwen2.5-7B 评测；
- SFT warmup；
- RLVR/GRPO 尝试；
- dashboard + live API demo；
- presentation 可用图表与 case study。

### Phase 2：真实论文/技术报告长文档任务（可选）

Phase 1 完成后，用户根据耗时、结果质量和剩余时间决定是否继续。Phase 2 可加入少量真实论文或技术报告任务，用于外部泛化测试。它不能反向拖慢 Phase 1。

### Phase 3：代码仓库级长上下文任务（可选）

Phase 2 后若仍有时间，再考虑 repo-level long-context tasks，例如跨文件 evidence、API usage、bug localization 或修改建议。该阶段工程复杂度高，只作为进攻扩展。

---

## 4. Benchmark 任务与数据设计

### 4.1 样例基本形态

每个 Phase 1 样例包含：

```text
Question
Long Context = [Document 1, Document 2, ..., Document N]
Gold Answer
Gold Evidence IDs
Distractor Evidence IDs
Stale Evidence IDs, if any
Expected Short Reasoning Steps
Metadata
```

模型输出采用用户已确认的 **Answer + 极简结构化推理 + cited evidence IDs**：

```text
Evidence: E03, E17
Steps:
1. E03 establishes the initial condition.
2. E17 updates or combines the relevant fact.
3. Therefore the answer is ...
Answer: ...
```

该格式的目的不是让模型暴露完整思维链，而是提供短、可解析、可评估的 evidence-grounded rationale。它服务于：

- 自动解析；
- citation scoring；
- reasoning consistency scoring；
- dashboard case visualization；
- SFT target；
- RLVR reward。

### 4.2 Task Family A：Anti-distractor Evidence Retrieval

**目标**：测试模型能否从大量语义相似干扰项中找到正确证据。

样例结构：

- 多个文档提到相似实体、事件或属性；
- 只有 1–2 个 evidence 支持正确答案；
- distractors 在 lexical 或 semantic 层面相似，但缺少关键条件、时间或约束；
- 问题要求引用 evidence IDs 并给出最终答案。

可控变量：

| 变量 | 取值示例 |
|---|---|
| context length | 8K / 16K / 32K / 64K |
| evidence position | front / middle / end / random |
| distractor strength | none / lexical / semantic / adversarial |
| number of distractors | 0 / 4 / 8 / 16 / 32 |
| evidence count | single / multi-evidence |

重点指标：answer accuracy、citation precision/recall/F1、distractor citation rate、position robustness。

### 4.3 Task Family B：Multi-hop Evidence Reasoning

**目标**：测试模型能否组合多个分散证据完成推理。

抽象样例：

```text
E04: Project Orion uses protocol P7.
E19: Protocol P7 requires approval from Team Delta if risk level > 3.
E31: Project Orion has risk level 5.
Question: Which team must approve Project Orion?
Answer: Team Delta
```

可控变量：

| 变量 | 取值示例 |
|---|---|
| hop count | 2-hop / 3-hop / 4-hop |
| evidence distance | near / medium / far |
| rule complexity | direct / conditional / chained |
| irrelevant rule count | low / medium / high |
| conflicting intermediate facts | none / mild / hard |

重点指标：answer accuracy、all-evidence recall、partial-evidence citation、reasoning step coverage、missing-hop error rate。

### 4.4 Task Family C：Conflict / Temporal Update Reasoning

**目标**：测试模型能否处理冲突证据、过期证据和最新证据。

抽象样例：

```text
E02: As of March, Device A uses firmware F1.
E15: In April, Device A was upgraded from F1 to F2.
E27: A legacy checklist still lists Device A as F1.
Question: What firmware does Device A currently use?
Answer: F2
```

可控变量：

| 变量 | 取值示例 |
|---|---|
| update count | 1 / 2 / 4 |
| stale evidence count | 1 / 4 / 8 / 16 |
| conflict position | old-front / old-end / new-front / new-end / mixed |
| timestamp form | explicit timestamp / implicit order / priority rule |
| latest evidence position | front / middle / end / random |

重点指标：answer accuracy、latest-evidence citation recall、stale evidence citation rate、conflict resolution accuracy、temporal robustness。

---

## 5. 数据规模、切分与生成 Pipeline

### 5.1 数据规模

目标是 **50K+ tasks**，但必须分阶段生成，避免一开始就把工程风险放大。

| Stage | Size | 目的 |
|---|---:|---|
| Pilot | 1K–2K | 验证格式、parser、reward、网页样例、少量模型试跑 |
| Core | 10K–20K | 主实验、API 对比、SFT warmup、dashboard 初版 |
| Full | 50K+ | RLVR/GRPO、统计分析、hard split、benchmark 级展示 |

### 5.2 数据切分

默认切分：

```text
train: 70%
dev: 10%
test: 20%
```

额外切分：

- `seen-template test`：同分布模板；
- `held-out-template test`：未见模板；
- `hard split`：更长上下文、更强干扰、更深 hop；
- `judge subset`：用于 LLM judge / verifier；
- `live-demo subset`：小上下文、稳定、适合现场演示。

### 5.3 混合生成策略

采用 **规则生成结构 + LLM 提升自然性 + 程序校验 + judge 抽样审计**。

#### Step 1：程序生成结构

程序负责生成：

- entities；
- document skeletons；
- evidence graph；
- answer；
- gold evidence IDs；
- distractor/stale evidence IDs；
- context metadata；
- task family；
- difficulty tags；
- train/dev/test split tag。

这一步保证变量可控、答案唯一、标签可验证。

#### Step 2：LLM 改写自然语言

LLM 负责：

- 将模板文档改写为自然段落；
- 生成语义相似干扰项；
- 改写问题表述；
- 增加风格、领域和叙述多样性；
- 降低模板味。

约束：LLM 不能改变 gold answer、core relation、evidence IDs、timestamp / priority rule。

#### Step 3：程序校验

校验包括：

- gold evidence 是否仍包含必要信息；
- distractor 是否不支持正确答案；
- answer 是否唯一；
- evidence ID 是否可解析；
- context length 是否达标；
- task variables 是否符合目标分布；
- 输出样例是否能被 parser 和 metrics 正常消费。

#### Step 4：LLM judge 抽样审计

对一部分样本运行 judge：

- question 是否清楚；
- gold evidence 是否足以推出答案；
- distractor 是否足够迷惑但不改变答案；
- 是否存在多解；
- expected steps 是否合理。

低质量样本进入过滤或再生成队列。

### 5.4 JSONL 数据格式

建议样例格式：

```json
{
  "id": "vlr_train_000001",
  "task_family": "temporal_update",
  "difficulty": "hard",
  "question": "...",
  "documents": [
    {
      "doc_id": "D01",
      "evidence_id": "E01",
      "text": "...",
      "role": "gold|distractor|stale|neutral"
    }
  ],
  "gold_answer": "...",
  "gold_evidence_ids": ["E15"],
  "distractor_evidence_ids": ["E02", "E27"],
  "expected_steps": [
    "Identify the latest update for Device A.",
    "Ignore stale checklist evidence.",
    "Use the April update to answer F2."
  ],
  "metadata": {
    "target_context_tokens": 32000,
    "evidence_position": "middle",
    "distractor_strength": "semantic",
    "hop_count": 2,
    "update_count": 1
  }
}
```

---

## 6. 输出解析、指标与 Reward 设计

### 6.1 输出解析

Parser 抽取：

- `pred_answer`；
- `pred_evidence_ids`；
- `pred_steps`；
- `format_valid`；
- `unparsed_text`；
- error flags：missing answer、missing evidence、invalid evidence ID、malformed step。

解析失败也要记录，不直接丢弃。解析失败本身是 format-following failure，可用于 dashboard 和训练 reward。

### 6.2 Answer Metrics

- Exact Match；
- Normalized Match；
- Type-aware match：数字、日期、实体名、yes/no、多选集合；
- answer accuracy by task family；
- answer accuracy by context length；
- answer accuracy by difficulty。

Answer accuracy 不能单独作为核心结论，因为它无法区分“猜对”和“基于正确证据答对”。

### 6.3 Citation / Evidence Metrics

- Citation Precision；
- Citation Recall；
- Citation F1；
- All-gold evidence recall；
- Distractor Citation Rate；
- Stale Evidence Citation Rate；
- Invalid Citation Rate；
- Over-citation Rate。

这些指标是 VeriLong-RL 区别于普通 long-context QA 的关键。

### 6.4 Reasoning Metrics

- Step Count Validity：是否为 2–4 条短步骤；
- Evidence-step Alignment：每个 step 是否对应正确 evidence；
- Missing-hop Rate；
- Contradiction-use Rate；
- Reasoning-Answer Consistency；
- Reasoning-Citation Consistency；
- Judge Faithfulness Score。

Reasoning metrics 采用程序规则 + LLM judge 双层评估。

### 6.5 程序化训练 Reward

训练内 reward 以程序化为主，保证速度、稳定性与可复现。

建议总 reward：

```text
R = w_answer * R_answer
  + w_citation * R_citation
  + w_reasoning * R_reasoning
  + w_format * R_format
  - w_distractor * P_distractor
  - w_stale * P_stale
  - w_invalid * P_invalid
```

初始权重：

```text
w_answer = 0.40
w_citation = 0.25
w_reasoning = 0.20
w_format = 0.10
w_distractor = 0.15
w_stale = 0.15
w_invalid = 0.10
```

权重由 dev set 与 judge subset 校准。训练时优先使用快速程序 reward；judge 不进入主在线 RL loop。

### 6.6 LLM Judge / Verifier 角色

用户已选择 judge 用于 **数据过滤与 reward 校准**。

Judge 使用位置：

1. 数据质量审计；
2. 模型输出评估；
3. 程序 reward 校准；
4. preference pair 构造；
5. final case study 分析。

Judge 输出结构：

```json
{
  "answer_supported": true,
  "citation_correct": true,
  "reasoning_faithful": false,
  "used_distractor": true,
  "used_stale_evidence": false,
  "ambiguity": false,
  "score": 0.62,
  "error_type": "distractor_contamination",
  "short_rationale": "The answer is correct, but step 2 relies on E08, which is a distractor."
}
```

Judge 固定 prompt、固定模型、固定 schema，并缓存结果。Judge 不作为唯一真值，而是用于校准和分析。

### 6.7 Error Taxonomy

| Error Type | 含义 |
|---|---|
| `answer_wrong` | 最终答案错误 |
| `citation_missing` | 答案正确但没有引用关键证据 |
| `distractor_contamination` | 引用或使用干扰证据 |
| `stale_evidence_use` | 使用过期证据 |
| `missing_hop` | 多跳推理漏掉必要中间证据 |
| `position_bias` | 某些位置证据表现显著更差 |
| `format_failure` | 输出格式无法解析 |
| `overcitation` | 引用过多无关证据 |
| `reasoning_answer_mismatch` | 推理与最终答案不一致 |
| `invalid_citation` | 引用不存在 evidence ID |

---

## 7. 模型评测与训练实验设计

### 7.1 实验分层

#### Layer 1：离线大规模诊断评测

目标：验证 VeriLong-RL 能否区分不同模型的 long-context evidence use 能力，并分析 failure modes。

模型层级：

- Claude API：默认使用 `claude-opus-4-8`，可加 `claude-sonnet-4-6` / `claude-haiku-4-5` 做成本与能力对照；若组织权限、预算和数据保留要求允许，可将 `claude-fable-5` 作为最强上界但不作为默认；
- 其他 API 模型：GPT / Gemini 等通过项目配置的 API gateway 或 provider SDK 接入，具体模型 ID 在实施时以当前账号可用列表为准；
- 开源模型：Qwen2.5-7B base / instruct / SFT / SFT+RLVR；
- 可选开源 baseline：Llama-3.1-8B、Llama-2-7B、Self-RAG Llama2-7B；
- 检索 baseline：Contriever / MiniLM。

#### Layer 2：SFT Warmup

目标：让 Qwen2.5-7B 学会输出格式、引用 evidence IDs、生成短结构推理，并提升基础 accuracy 与 citation recall。

默认策略：

- 第一轮用 `Qwen2.5-7B-Instruct`，保证格式学习和训练成功率；
- 如果时间充足，再补 `Qwen2.5-7B` base 对照；
- LoRA / QLoRA；
- sequence length 先 8K/16K，再 32K；
- 64K 主要用于 evaluation，不强求训练覆盖。

#### Layer 3：GRPO / RLVR

目标：在 SFT checkpoint 基础上，用程序 reward 强化 answer correctness、citation correctness、reasoning consistency、抗干扰和 stale evidence avoidance。

默认策略：

- 从 Qwen2.5-7B-SFT checkpoint 开始；
- 先跑 8K/16K retrieval family；
- 稳定后加入 multihop 与 temporal-conflict；
- reward 以程序化为主；
- judge 用于数据过滤、reward 校准、preference pair 与分析，不作为在线 RL 主 reward。

### 7.2 已确认服务器资源

已通过只读 SSH 到 `Tang-3-Wu` 查看 `/NAS/yesh`，确认资源约束：

- `Tang-1-Wu / Tang-2-Wu / Tang-3-Wu`：各 `8×A40`；
- `Song-3-Wu`：`8×A100`；
- NAS 共享路径：`/NAS/yesh`；
- 已有 HF cache 权重包括：
  - `Qwen/Qwen2.5-7B-Instruct`；
  - `Qwen/Qwen2.5-7B`；
  - `meta-llama/Llama-3.1-8B`；
  - `meta-llama/Llama-2-7b-hf`；
  - `NousResearch/Llama-2-7b-hf`；
  - `selfrag/selfrag_llama2_7b`；
  - `facebook/contriever-msmarco`；
  - `sentence-transformers/all-MiniLM-L6-v2`。

因此主训练模型默认采用 Qwen2.5-7B 系列，检索 baseline 可使用 Contriever / MiniLM。

### 7.3 Evaluation Matrix

| 维度 | 取值示例 |
|---|---|
| task family | retrieval / multihop / temporal-conflict |
| context length | 8K / 16K / 32K / 64K |
| evidence position | front / middle / end / random |
| distractor strength | none / lexical / semantic / adversarial |
| evidence count | 1 / 2 / 3 / 4 |
| hop count | 1 / 2 / 3 / 4 |
| conflict/update count | 0 / 1 / 2 / 4 |
| model group | API / open-source base / instruct / SFT / RLVR |

为避免组合爆炸，主实验采用主效应设计：每次只变化 1–2 个变量，并保留 balanced dev/test 与 hard split。

### 7.4 主要研究问题

**RQ1：不同模型在 evidence-grounded long-context tasks 上的差异是什么？**

比较 Claude / GPT / Gemini 与 Qwen 系列，使用 answer accuracy、citation F1、distractor rate、judge faithfulness。

**RQ2：长上下文失败主要来自哪里？**

分析证据没找对、找对但没用、被干扰证据污染、漏掉 multi-hop、使用 stale evidence、输出格式失败等错误。

**RQ3：SFT 与 RLVR 分别改善了什么？**

预期 SFT 主要改善 format validity、citation recall 和 task following；RLVR 可能改善 citation precision、distractor avoidance、stale evidence avoidance 和 reasoning consistency。

**RQ4：哪些 task / reward 更适合 Long Context RL？**

分析 retrieval、multihop、temporal-conflict 三类任务对 reward 设计的要求，以及 judge 校准能否发现程序 reward 盲点。

### 7.5 API 模型调用约束

涉及 Claude API 的实现必须遵守当前 API 设计约束：

- 默认 Claude 强模型 ID 使用 `claude-opus-4-8`；
- Claude Opus 4.8 支持 1M context、128K max output，但大输出应使用 streaming；
- Opus 4.8 / 4.7 使用 adaptive thinking，不能使用 `budget_tokens`；
- Opus 4.8 / 4.7 不接受 `temperature` / `top_p` / `top_k`；
- 结构化输出使用 `output_config.format`，不要使用旧的 `output_format`；
- 不使用 assistant prefill 控制 JSON；
- 如果做高并发离线评测，优先使用 batch / queue / retry / cache 设计，避免实时 demo 直接依赖大规模 API 请求；
- live demo 限制 context length 和 timeout，并提供 cached fallback。

---

## 8. Web Dashboard / Live Demo 与工程架构

### 8.1 网页定位

网页定位为 **VeriLong-RL Benchmark Portal**，不是单纯介绍页。它承担三层作用：

1. 科研展示：任务定义、数据生成 pipeline、metrics、实验结果；
2. 工程展示：后端 API、任务生成、模型调用、结果解析、可视化；
3. 答辩辅助：现场可运行小型任务，即使 live call 失败也能回退 cached examples。

### 8.2 页面设计

#### Page 1：Home / Project Overview

内容：

- 项目标题 `VeriLong-RL`；
- 一句话定义；
- 三个核心问题：
  - Are models answering from the right evidence?
  - Can long-context tasks be made verifiable enough for RLVR?
  - Do SFT and RLVR improve answer correctness, citation correctness, and reasoning faithfulness differently?
- 三个任务族卡片；
- 关键成果数字：50K+ tasks、3 task families、API + open-source evaluation、SFT + RLVR、interactive live demo。

#### Page 2：Dataset / Task Generator

展示数据生成 pipeline：

```text
Programmatic schema generation
→ LLM paraphrase / distractor generation
→ deterministic validation
→ judge-based quality audit
→ train/dev/test split
```

支持选择 task family、context length、distractor strength、evidence position、hop count、conflict/update count，并展示生成样例、gold evidence、distractor/stale evidence、expected steps 和 metadata。

#### Page 3：Leaderboard

展示模型整体表现：

| Model | Overall | Answer Acc | Citation F1 | Reasoning Consistency | Distractor Rate |
|---|---:|---:|---:|---:|---:|
| Claude Opus 4.8 | value | value | value | value | value |
| GPT configured strong model | value | value | value | value | value |
| Gemini configured strong model | value | value | value | value | value |
| Qwen2.5-7B-Instruct | value | value | value | value | value |
| Qwen2.5-7B-SFT | value | value | value | value | value |
| Qwen2.5-7B-RLVR | value | value | value | value | value |

表中的 `value` 在实验完成后由结果文件填充；设计阶段不伪造结果。

#### Page 4：Breakdown Analysis

展示按 task family、context length、evidence position、distractor strength、training stage 切分的图表。

#### Page 5：Case Study Viewer

每个案例展示：

- question；
- documents；
- gold evidence 高亮；
- distractor/stale evidence 高亮；
- model output；
- parsed answer；
- parsed evidence IDs；
- parsed reasoning steps；
- metric breakdown；
- judge verdict；
- error type。

典型案例：答案对但引用错、使用 stale evidence、多跳漏证据、被语义相似 distractor 污染、RLVR 后 citation precision 提升。

#### Page 6：Training Results

展示：

- SFT loss curve；
- dev answer accuracy；
- citation F1；
- format validity；
- RL reward curve；
- RLVR 前后对比；
- checkpoint comparison。

#### Page 7：Live Demo

流程：

1. 用户选择 task family、difficulty、small/medium context length、API model；
2. 后端生成一个小任务；
3. 后端调用 API 模型；
4. 解析输出；
5. 返回 answer、evidence IDs、steps、metric score、gold evidence 对齐和 error type。

稳定性设计：

- live demo 默认只跑 4K/8K/16K；
- 设置 API timeout；
- cached fallback；
- 预置 demo examples；
- 限制请求频率；
- 不暴露 API key；
- 对外展示时只开放有限模型。

### 8.3 后端架构

建议 Python FastAPI：

```text
backend/
  app.py
  api/
    routes_tasks.py
    routes_eval.py
    routes_leaderboard.py
    routes_live_demo.py
  core/
    generator/
    validators/
    parser/
    metrics/
    reward/
    judge/
    model_clients/
  storage/
    datasets/
    results/
    cached_runs/
```

核心 API：

- `GET /api/summary`
- `GET /api/leaderboard`
- `GET /api/results/breakdown`
- `GET /api/cases`
- `POST /api/tasks/generate`
- `POST /api/live/run`
- `GET /api/training/curves`

### 8.4 前端架构

建议 React + Vite + TypeScript：

```text
frontend/
  src/
    pages/
      Home.tsx
      Dataset.tsx
      Leaderboard.tsx
      Analysis.tsx
      CaseViewer.tsx
      Training.tsx
      LiveDemo.tsx
    components/
      MetricCard.tsx
      EvidenceHighlighter.tsx
      ModelSelector.tsx
      ResultTable.tsx
      ChartPanel.tsx
      CaseTimeline.tsx
```

图表库可在实施时选择 Recharts / ECharts / Plotly。默认优先选择易开发、适合 dashboard 的方案。

### 8.5 数据与结果存储

本项目初期不需要复杂数据库。建议：

- dataset：JSONL / Parquet；
- results：JSONL / CSV / Parquet；
- dashboard summary：预聚合 JSON；
- cached live examples：JSON；
- training curves：CSV / JSON。

后端读取这些文件并提供 API。只有当 live demo 需要记录请求历史时，再引入 SQLite。

---

## 9. PPT / Survey 叙事设计

Presentation 叙事不应变成“我读了很多论文”。推荐结构：

### Part 1：Survey Motivation

- Long Context 的能力越来越重要；
- data strategy 从 pretraining 长序列、SFT instruction、long-context alignment 发展到 RLVR；
- 现有评测常只看 final answer，难以判断模型是否真正 grounding 到证据。

### Part 2：Gap

核心 gap：

> Long-context RL 需要可验证任务，但很多长上下文任务不是天然可验证；只看答案会掩盖 evidence use 的失败。

进一步展开：

- 模型可能答案对但引用错；
- 可能被 distractor 污染；
- 可能使用 stale evidence；
- multi-hop 中可能漏掉关键 evidence；
- reward 如果只看 final answer，可能鼓励捷径。

### Part 3：VeriLong-RL Design

介绍：

- 三个任务族；
- answer + evidence + short steps 输出格式；
- 程序控制 gold labels + LLM 提升自然性；
- metrics / reward decomposition；
- judge 用于校准而非在线 reward。

### Part 4：Experiments

展示：

- API 模型 vs Qwen 训练线；
- SFT 与 RLVR 前后对比；
- 按 task family / length / distractor strength 的 breakdown；
- error taxonomy；
- 典型 case。

### Part 5：Demo

现场打开 dashboard：

- leaderboard；
- case viewer；
- live demo 生成一个任务并调用 API；
- 展示 evidence alignment。

### Part 6：Takeaways

建议结论模板：

1. Long-context evaluation should be evidence-aware, not answer-only.
2. Verifiable synthetic tasks can bridge evaluation and RLVR training.
3. SFT mainly teaches format and citation behavior, while RLVR can target evidence-grounded correctness.
4. Distractor and stale-evidence failures expose limitations hidden by aggregate accuracy.

---

## 10. 建议仓库结构

```text
VeriLong-RL/
  Preliminary research/
    deep-research-report.md
    Long Context Data Strategies Survey.md
  docs/
    superpowers/
      specs/
        2026-06-26-verilong-rl-design.md
    handoff/
      new-agent-prompt.md
      prior-session-tail-extract.txt
  memory/
    MEMORY.md
    long-context-experiment-scope.md
    cluster-resources.md
    project-decisions.md
  benchmark/
    schemas/
    generator/
    validators/
    parser/
    metrics/
    reward/
  data/
    pilot/
    core/
    full/
  experiments/
    eval_api/
    eval_open_source/
    sft/
    rlvr/
    judge/
  backend/
  frontend/
  scripts/
  results/
    raw/
    processed/
    figures/
```

本设计文档只创建文档与记忆，不创建完整工程脚手架；工程脚手架应在实施计划确认后再生成。

---

## 11. 里程碑与执行顺序

### Milestone 0：项目初始化与实施计划

- 完成设计文档；
- 完成本地项目记忆迁移；
- 写 implementation plan；
- 明确第一周具体任务。

### Milestone 1：Pilot 闭环

目标：1K–2K pilot tasks。

完成：

- 三个任务族最小生成器；
- JSONL schema；
- parser；
- answer/citation/format metrics；
- 少量 API 模型试跑；
- case viewer 所需样例数据。

### Milestone 2：Core Benchmark

目标：10K–20K core tasks。

完成：

- LLM 改写与程序校验；
- dev/test/hard split；
- API + Qwen instruct 主要评测；
- dashboard leaderboard 和 breakdown 初版。

### Milestone 3：SFT Warmup

完成：

- SFT 数据格式转换；
- Qwen2.5-7B-Instruct LoRA/QLoRA；
- dev eval；
- SFT vs instruct 对比；
- 训练曲线。

### Milestone 4：RLVR / GRPO

完成：

- reward functions；
- GRPO pipeline；
- 小规模 retrieval family smoke test；
- 加入 multihop / temporal；
- RLVR vs SFT 对比；
- 失败时保留 reward analysis 与 SFT 作为完整结果。

### Milestone 5：Dashboard + Live Demo

完成：

- FastAPI backend；
- React dashboard；
- leaderboard / analysis / case viewer / training results；
- live demo API model call；
- cached fallback。

### Milestone 6：Presentation

完成：

- survey 叙事；
- 实验图表；
- demo script；
- 讲稿；
- 风险预案：live demo 失败时使用 cached run。

---

## 12. 风险与控制策略

| 风险 | 控制策略 |
|---|---|
| 50K+ 数据生成耗时过长 | Pilot → Core → Full 分阶段，先保证 1K–2K 闭环 |
| LLM 改写导致标签漂移 | 程序校验 + judge 抽样审计 + 失败样本再生成 |
| API 成本过高 | 先跑 small dev subset；使用缓存、批处理、抽样 judge；live demo 限制长度 |
| RLVR/GRPO 不稳定 | SFT 是必做主线；GRPO 从 8K retrieval 开始；失败时展示 reward analysis |
| 64K 训练显存压力 | 训练先 8K/16K/32K；64K 主要用于 eval |
| Dashboard 工程量过大 | 先静态数据驱动 dashboard，再加 live demo |
| Live demo API 延迟或失败 | timeout + cached fallback + 预置 demo examples |
| 模型调用接口变化 | Claude 调用遵循当前 API 约束；其他 provider 模型 ID 实施时从配置读取 |
| 项目被真实论文/代码仓库扩展拖散 | Phase 1 完成前禁止扩展；Phase 2/3 必须再次询问用户 |

---

## 13. 成功标准

### 最低成功标准

- 完成 Phase 1 pilot/core 数据闭环；
- 至少有 API 模型与 Qwen2.5-7B-Instruct 的评测结果；
- 有 parser、metrics、error taxonomy；
- 有可展示 dashboard 和 case viewer；
- presentation 能讲清 survey → gap → task design → results。

### 目标成功标准

- 50K+ synthetic tasks；
- API + Qwen base/instruct/SFT/RLVR 完整对比；
- SFT 提升 format validity 和 citation recall；
- RLVR 至少在某些 task family 上改善 citation precision、distractor/stale avoidance 或 reasoning consistency；
- live demo 可稳定展示；
- PPT 中有清晰研究贡献与工程产出。

### 进攻成功标准

- Phase 2 加入真实论文/技术报告任务；
- Phase 3 加入代码仓库级任务；
- 形成可公开 project page / benchmark repo / dataset card；
- dashboard 接近正式 benchmark 官网体验。

---

## 14. 已确认设计决策摘要

1. 项目独立于用户原有研究项目；原项目只提供集群与服务器调用经验。
2. 总方向为 Long Context RL-style verifiable task design，Evidence use 为副方向。
3. 采用两层实验：离线诊断系统为主线，SFT + RLVR/GRPO 为进攻项。
4. 任务场景采用混合场景，但先完成合成多文档证据任务。
5. 输出格式为 Answer + 极简结构化推理 + cited evidence IDs。
6. 任务族采用 retrieval、multihop、temporal-conflict 三者分层推进。
7. 训练策略采用 SFT + RLVR/GRPO。
8. 网页展示目标是 online runnable demo，但 live demo 只调用 API 模型。
9. 离线评测采用 API + 开源 7B 对称对比。
10. 数据规模目标为 50K+，采用 Pilot/Core/Full 阶段推进。
11. 数据生成采用规则 + LLM 混合生成。
12. Reward 采用 verifier / judge-enhanced 设计，但 judge 放在数据过滤与 reward 校准，不作为在线 RL 主 reward。
13. 最终包装为 Benchmark 项目形态。

---

## 15. 下一步

设计完成后，下一步应进入 implementation planning，而不是直接大规模写代码。实施计划需要拆出：

1. 数据 schema 与 generator；
2. parser / metrics / reward；
3. pilot 评测；
4. SFT 数据与训练脚本；
5. RLVR/GRPO 训练脚本；
6. FastAPI backend；
7. React dashboard；
8. presentation 图表与讲稿。

在实施计划前，必须保持 Phase 1 优先，不要提前扩展真实论文或代码仓库任务。
