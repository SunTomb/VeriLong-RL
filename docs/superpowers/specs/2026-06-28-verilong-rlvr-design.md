# VeriLong-RL Phase 1 RLVR 设计文档

> 工作名：**VeriLong-RL RLVR — GRPO on programmatic citation reward**
> 日期：2026-06-28
> 范围：Phase 1 合成多文档证据任务主线内的 RLVR 增量（Layer 3）。不触碰 Phase 2/3。

## 1. 目标与研究问题

在 SFT warmup 之上叠加一轮 RLVR（GRPO），用**纯程序化 reward** 进一步强化证据引用纪律，回答设计 spec 的 **RQ3**：

> SFT 与 RLVR 分别改善了什么？预期 SFT 改善 format validity / citation recall / task following；**RLVR 改善 citation precision / distractor avoidance / stale avoidance / reasoning consistency**。

这是 SFT 阶段已建立的主线"答对≠用对证据"的自然延续：dev30 已饱和（base 外三个 SFT 全 0.950），区分度全在 hard slice 的引用纪律。RLVR 要验证的是：**在 SFT 已学会格式与召回之后，RL 能否把引用精度（不引干扰证据、不引过期证据、不过度引用）再推一步。**

## 2. 已敲定决策（brainstorming 2026-06-28）

| # | 决策点 | 选定 | 理由 |
|---|---|---|---|
| 1 | RL 起点 checkpoint | **v1-gold SFT**（`checkpoints/qwen2_5_7b_sft_v1`） | 标准 SFT→RL 流程，符合 spec『从 SFT checkpoint 起』；v1 在 hard 上仍有 overcite/stale 残留（各 0.074），留出提升空间；从已学会格式的 checkpoint 起，RL 冷启动稳。 |
| 2 | reward 函数 | **复用现成 `compute_reward`** | 与评测同一把尺，RL 优化目标 = 报告指标，无 reward-hacking 解释负担、最可复现。GRPO 自带 KL 正则到参考策略（`beta`）防退化。 |
| 3 | 任务范围 | **先 retrieval family、8K-context** | retrieval 是 base overcite 最明显的族（hard overcite 0.432），RL 信号最清晰；符合 spec『先跑 retrieval』；跑通闭环后再扩 multihop/temporal。 |
| 4 | rollout 后端 | **vllm 加速（2 卡）** | vllm 0.6.6.post1 已装于 `verilong_rl` env；长上下文采样比 HF generate 快数倍；trl GRPO 用 vllm 占一张卡做生成、训练在另一张卡，Song-3 大卡可负担。 |
| 5 | 成功判据 | **hard slice 上 v1 vs v1+RL 引用纪律** | 主判据：hard16k retrieval 的 citation precision↑/overcitation↓/distractor↓；次判据：reward_total 不降、answer 不退化；过程证据：训练曲线（reward 及各 component 随 step）。dev30 已饱和，不作主判据。 |
| — | 训练脚本搭法 | **方案 A：trl GRPOTrainer + 复用打分链路** | 最少新代码；reward 复用已验证的 `score_output_record` 核心；与现有 benchmark 无缝集成。 |

## 3. 架构与数据流

```
retrieval 8K train tasks (data/pilot/tasks.jsonl, split=train, family=retrieval, ctx=8000)
        │  build_user_prompt(task)  (复用 eval/SFT 同一 prompt)
        ▼
   prompt dataset  ──►  GRPOTrainer (trl 0.14.0)
        │                   │  每个 prompt 采样 G 个 completion (vllm 生成)
        │                   ▼
        │             reward_funcs(prompts, completions, **task_cols)
        │                   │  对每个 completion:
        │                   │    parse_model_output → citation/format/answer metrics
        │                   │    → compute_reward → 标量 total
        │                   ▼
        │             group-relative advantage → policy gradient + KL(beta) to v1 ref
        ▼
   v1 SFT (LoRA adapter, 冻结 base) ──► 继续训练 LoRA ──► v1+RL adapter
        │
        ▼
   eval: run_hf_eval --adapter v1+RL  on  hard_eval_sft_16k.jsonl (retrieval 子集)
        │  score_outputs → summary
        ▼
   对比表: v1-gold vs v1+RL  (citation_precision / overcitation / distractor / reward_total / answer)
```

核心复用点：**reward 不重写**。新增的 reward function 只是把现有 `parse_model_output → citation_scores/format_scores/normalized_match → compute_reward` 包成 trl 期望的回调签名。这保证 RL 的优化目标与最终评测口径完全一致。

## 4. 组件与文件结构

新增（全部在 `experiments/rlvr/`，与 `experiments/sft/` 平级，保持现有风格）：

| 文件 | 职责 | 依赖 |
|---|---|---|
| `experiments/rlvr/__init__.py` | 包标记 | — |
| `experiments/rlvr/reward.py` | `make_reward_fn()` 返回 trl 兼容的 reward 回调；内部复用抽出的 `benchmark.reward.score.score_output_record`（即评测同一打分核心）。纯函数、可单测。 | benchmark.reward.score |
| `experiments/rlvr/data.py` | `load_grpo_prompts(tasks_path, family, max_context_tokens, split)` → HF Dataset，每行含 `prompt`（= SYSTEM_PROMPT + build_user_prompt）和透传的 `task_json`（该 task 的 JSON 字符串）供 reward 还原打分。 | benchmark.schemas / run_api_eval (SYSTEM_PROMPT, build_user_prompt) |
| `experiments/rlvr/train_grpo.py` | 主训练脚本：读 config、加载 v1 base+adapter、构造 prompt dataset、配置 GRPOConfig（含 use_vllm）、跑 GRPOTrainer、导出 v1+RL adapter + 训练曲线 JSON。 | trl, peft, transformers, 上面三者 |
| `experiments/rlvr/configs/grpo_v1.yaml` | 超参：起点 adapter、family、num_generations(G)、beta(KL)、lr、max_prompt/completion_length、vllm gpu、max_steps、save。 | — |
| `tests/benchmark/test_rlvr_reward.py` | 单测 reward 回调：gold-perfect 输出得高 reward、引干扰/过期得低 reward、格式错得低 reward；与 score_outputs 口径一致性。 | experiments.rlvr.reward |

**边界清晰性**：`reward.py` 不知道 trl/训练存在（纯打分）；`data.py` 不知道 reward（只造 prompt+task 列）；`train_grpo.py` 编排三者。每个可独立理解和测试。

## 5. Reward 回调设计（核心）

trl 0.14 `GRPOTrainer` 的 `reward_funcs` 签名：`fn(prompts, completions, **kwargs) -> list[float]`，其中 `kwargs` 是 dataset 中除 `prompt` 外的列（按列名透传，每个是 batch list）。

```python
# experiments/rlvr/reward.py  （示意；选定方案 (b)，实现见 plan）
def make_reward_fn():
    def reward_fn(prompts, completions, task_json, **_):
        # task_json[i]: 该 prompt 对应 task 的 JSON 字符串（data.py 透传）
        rewards = []
        for i, completion in enumerate(completions):
            text = completion if isinstance(completion, str) else completion[0]["content"]
            task = VeriLongTask.model_validate(json.loads(task_json[i]))
            # 直接复用评测打分核心，保证 reward 与报告同尺
            scored = score_output_record(task, {"output_text": text})
            rewards.append(scored["reward_total"])
        return rewards
    return reward_fn
```

说明：`score_output_record`（现于 `scripts/score_outputs.py`）内部已串好 `parse_model_output → citation_scores/format_scores/normalized_match → compute_reward`。为让 reward.py 能 import 它而不依赖 `scripts/`，plan 第一步把 `score_output_record` 抽到 `benchmark/reward/score.py`，`score_outputs.py` 改为从那里 import（保持现有 CLI 行为不变、现有测试不破）。

一致性测试：同一 (task, output_text)，reward.py 的输出必须等于 `score_outputs.score_output_record` 的 `reward_total`（误差 <1e-6）。

## 6. 训练配置与集群资源

**起点**：`Qwen/Qwen2.5-7B-Instruct` base + LoRA adapter `checkpoints/qwen2_5_7b_sft_v1`（v1-gold）。GRPO 继续训练这个 LoRA（base 冻结），导出 `checkpoints/qwen2_5_7b_sft_v1_grpo`。

**GRPOConfig 初始超参（grpo_v1.yaml）**：
- `num_generations`（G，每 prompt 采样数）：8
- `beta`（KL 到 v1 参考策略）：0.04（温和，防退化不抑制学习）
- `learning_rate`：1e-6（RL 远小于 SFT 的 2e-4）
- `max_prompt_length`：14336（retrieval 8K ≈ 14K token，与 SFT max_seq_len 一致口径）
- `max_completion_length`：256（Evidence/Steps/Answer 输出短）
- `per_device_train_batch_size`：1，`gradient_accumulation_steps`：8
- `max_steps`：先 100（闭环验证），看曲线再延
- `temperature`：0.7（rollout 需多样性，区别于评测的贪心）
- `use_vllm`：true
- `save_strategy`：steps，每 25 步存一次

**集群（2 卡，Song-3 大卡优先）**：
- vllm 生成占 1 卡（trl 的 `vllm_device`，如 `cuda:1`），policy 训练占另 1 卡。
- 长上下文 + G=8 采样显存高，Song-3 80GB 卡是首选；不够则降 G 或 max_prompt_length。
- 严格 `CUDA_VISIBLE_DEVICES` 锁定，`nvidia-smi` 先看空卡，沿用 SFT 的 GPU 约定。
- env：`verilong_rl`（非 gmsra），torch 2.5.1+cu121 / vllm 0.6.6.post1 / trl 0.14.0 已验证。

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| trl 0.14 + vllm 0.6.6 在长上下文 GRPO 的 API/显存细节未实跑验证 | 先 `--smoke`：G=2、max_steps=2、max_prompt 截短，跑通管线再放大。仿 SFT 的 smoke 模式。 |
| vllm 2 卡配置复杂、首次易错 | smoke 阶段先单卡 HF generate 验证 reward 闭环（reward 在上升），再切 vllm 扩规模。回退路径明确。 |
| dev30 已饱和，RL 提升可能仅在 hard 显现，甚至 v1 hard 0.891 也接近上限 | 主判据放在 hard retrieval 的 **citation_precision/overcitation** 这些 v1 仍未满分的细分指标，而非已近满的 reward_total。若无提升也是诚实结论（"SFT 已接近此难度上限，RL 边际收益有限"）。 |
| RL 退化（reward hacking / 输出崩坏） | KL beta 约束 + 每 25 步 checkpoint + 监控训练曲线各 component；answer_normalized 跌破阈值即停。 |
| 长上下文 rollout 极慢 | retrieval 单族 + max_steps 100 起；vllm 加速；先验证再扩。 |
| NAS 100% 满 | adapter 仅 ~160MB；rollout 不落盘大文件；训练前确认剩余空间。 |

## 8. 测试策略

- **单元（本地，无 GPU）**：`test_rlvr_reward.py` —— reward 回调对 gold-perfect/distractor/stale/格式错输出的打分符合预期；且与 `score_output_record` 的 `reward_total` 数值一致（同尺验证）。`data.py` 的 prompt 构造与透传列正确。
- **冒烟（集群）**：`train_grpo.py --smoke`（G=2, max_steps=2, 截短 prompt, 单卡 HF generate）跑通端到端管线、不崩。
- **闭环验证（集群）**：max_steps 100、vllm、retrieval 8K，看训练曲线 reward 是否上升、各 component 走向。
- **终点评测**：v1+RL adapter 在 `hard_eval_sft_16k.jsonl` 的 retrieval 子集上评，与 v1-gold 同口径对比。
- 现有 67 tests 必须保持绿（新增不破坏）。

## 9. 成功标准与交付

**成功（任一即有价值的真实结论）**：
1. 正向：hard retrieval 上 v1+RL 的 citation_precision↑ 或 overcitation↓，且 answer 不退化、reward_total 不降 → RLVR 改善引用纪律，验证 RQ3。
2. 中性但诚实：无显著提升，训练曲线显示 reward 已饱和 → "SFT 已接近此难度 reward 上限，RL 边际收益有限"，同样是 RQ3 的有效答案，与 dev30 饱和现象一致。
3. 反例也记录：若退化，分析 reward hacking 形态（这本身是 RLVR 研究的有价值观察）。

**交付物**：
- `experiments/rlvr/`（reward / data / train_grpo / config）+ 单测，67→更多 tests 绿。
- v1+RL adapter（`checkpoints/qwen2_5_7b_sft_v1_grpo`）。
- 训练曲线 JSON + hard retrieval 上 v1 vs v1+RL 对比表（真实数字）。
- memory 更新：RLVR 结果并入 [[sft-warmup-plan]] 或新建条目。

**不做（YAGNI / 阶段门控）**：multihop/temporal 的 RL（闭环验证后才考虑）、32K 上下文 RL、Phase 2/3 任务、在线 LLM judge reward（spec 明确禁止）、多 RL 算法对比（只 GRPO）。

## 10. Related
- 设计 spec：`2026-06-26-verilong-rl-design.md` Layer 3 / RQ3
- memory：[[sft-warmup-plan]]（v1/v2 结果、四方对比）、[[cluster-gpu-usage-tang-song]]（verilong_rl env、Song-3 大卡）、[[hard-difficulty-system]]（hard slice）、[[project-decisions]]



