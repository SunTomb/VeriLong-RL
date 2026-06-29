# 长上下文数据策略深度综述

## 结论摘要

如果把“长上下文能力”严格限定为**数据与训练配方**问题，而不讨论 RoPE 扩展、稀疏注意力、线性注意力等架构改造，那么过去两年的公开证据几乎指向同一个结论：**真正决定长上下文是否“可用”的，不是单纯把上下文窗拉长，而是把预训练、SFT、RL 三个阶段都改造成“长度感知”的数据系统**。其中，最稳定的主线是：在持续预训练阶段保留大量高质量短数据以防退化；在 SFT 阶段引入少量但高度针对性的长样本或长任务；在 RL 阶段用**密集、可验证、过程式**奖励替代纯最终答案奖励，以解决长链条中的稀疏监督与 credit assignment 问题。citeturn18view1turn19view4turn28view0turn27view2turn40search6turn39view0

另一个非常清晰的趋势是，**工业界公开披露的“长上下文数据配方”远少于学术界和少数开源团队**。Meta 的 Llama 3/3.1、Princeton 的 ProLong、THUDM 的 LongAlign/LongWriter、LongMagpie、Together AI 的层级式合成长上下文 SFT，是目前最有“配方细节”的公开材料；而 Google Gemini 1.5、OpenAI GPT‑4.1、Anthropic Claude、DeepSeek 等前沿闭源或半闭源系统，公开文档普遍会承认模型经过了长上下文训练或对 1M 上下文进行了专门优化，但**很少公开 exact ratio、长度 curriculum、长短样本混比或质量过滤细节**。citeturn18view0turn19view4turn31view0turn32view0turn35view0turn10view0turn6search2turn8search13turn7search3

对你准备 15–20 分钟学术报告而言，一个最有效的中心论点可以是：**行业已经从“只做长预训练”转向“短长混合的持续预训练 + 轻量但强针对性的长 SFT + 面向长链条的稠密 RL 奖励”**。这比单独讲“把训练长度从 8K 拉到 128K/1M”更符合 2024–2026 的公开证据。citeturn18view1turn19view4turn28view0turn27view2turn40search6turn39view0

## 预训练与持续预训练数据策略

### 总体趋势

在公开资料中，最成熟的长上下文持续预训练 recipe 来自 Meta 的 **Llama 3.1** 与 Princeton 的 **ProLong**。Llama 3 的公开报告明确给出了预训练数据混合思路：其最终预训练 mix 约为 **50% general knowledge、25% mathematical/reasoning、17% code、8% multilingual**；同时，405B 版本先在 **8K** 窗口上完成标准预训练，再进行一个把上下文从 **8K 逐步拉到 128K** 的持续预训练阶段，且这一阶段用了大约 **800B tokens**，长度提升分成 **六个阶段** 完成。citeturn18view0turn18view1

ProLong 则把这个问题推进到了“**到底该喂什么长数据、短数据该保留多少、训练长度和评测长度如何错位**”这一层面。该工作在 ACL 2025 接收版中明确指出：**code repositories 与 long books 是最优长数据来源**，但必须和高质量短数据一起训练；如果“只喂长数据”，即使困惑度继续下降，真实长任务性能反而会恶化。citeturn26view0turn28view0

### 长短数据混配

这一维度上，公开证据最强的是 ProLong。作者把长数据定义为 **单文档 64K chunk**，短数据则通过**多文档 packing 到 64K**。系统性的比例实验表明，最佳平均表现出现在 **60% 长数据 + 40% 短数据**；当长数据比例继续上升到极端时，短任务能力单调下降，而长任务在 SFT 之后也会出现退化。citeturn28view0

ProLong 还进一步给出了其短数据配方 **ShortMix**：**25% FineWeb、25% FineWeb‑Edu、10% Wikipedia、10% Tulu‑v2、10% StackExchange、10% ArXiv、10% OpenWebMath**。这很重要，因为它说明“保住短能力”并不是靠原始 web mixture 自动发生，而是靠**后段 deliberately knowledge‑intensive 的短数据再混合**。citeturn28view0

Meta 的报告虽然没有披露“<8K / 32K / 128K”分桶比例，但给出了同样方向的信号：在基础预训练期间会**动态调整 data mix**，例如上调 non‑English 提升多语表现、上采样数学数据提升 reasoning、在后期加入更新的网页数据刷新知识截止日期、同时下采样事后发现的低质量子集。换言之，工业界已经把长上下文训练看成**持续调配 data recipe** 的过程，而不是固定配方的一次性训练。citeturn14view0

### 长度 curriculum

Llama 3.1 的长度 curriculum 非常直接：**8K → 128K，分六阶段推进**，每一阶段的“过关标准”不是 PPL，而是两条：短上下文能力要恢复、needle‑in‑a‑haystack 要在目标长度上“完全解对”。这说明工业界已经把长度上采样当作**能力恢复问题**而不是单纯训练步数问题。citeturn18view1

ProLong 的结论更进一步：**训练长度超过评测长度会带来额外收益**。作者最终让模型总共训练 **40B tokens**，并在 recipe 总结里强调自己进行了 **20B tokens@64K** 与 **20B tokens@512K** 的训练扩展；这正对应其核心观察——为拿到 128K 评测性能，训练时不应只停在 128K，而要故意超训练到更长长度。citeturn26view0turn27view0

Together AI 的 **Scaling Instruction‑Tuned LLMs to Million‑Token Contexts via Hierarchical Synthetic Data Generation** 也延续了这种“分阶段拉长”的思路。该工作报告：为了把 Llama‑3.1‑8B‑Instruct 拉到 **1M context**，训练从 **180K** 开始，依次经过 **350K、650K、1M** 检查点；数据样本数分别为 **2000、1280、600、200**，并通过多文档组合把上下文拉长到任意尺度。虽然该工作同时包含 RoPE stepwise scaling，但其真正有价值的数据信息是：**百万长度不是靠“找到天然 1M 文档”，而是靠层级式、多文档的 synthetic instruction 生成**。citeturn37view3turn37view5

### packing 与跨文档污染控制

这一主题上，公开文献已经出现两条分支。

第一条是 **“显式 document boundary” 派**。Llama 3 报告显示，Meta 在长序列训练时使用了 **document mask**，并明确说明其 context parallel 实现选择 all‑gather 路线的一个原因，就是更容易支持**不同注意力 mask，例如 document mask**。这说明他们明确在工程层面防止跨文档注意力污染。citeturn20view0turn20view1

ProLong 也直接研究了这一点，并把最终 recipe 写成 **“Full attention with cross-document attention masking”**。同时，作者在论文里指出，**禁用跨文档注意力**会同时改善短、长上下文表现，而且还能提高训练吞吐。更具体地，他们为了利用 document masking，把多文档短数据和 SFT 数据都打包成 **64K chunks**；配合 variable‑length attention 与 minibatch reordering，64K 训练吞吐从 **2770 tokens/s/GPU** 提高到 **3095 tokens/s/GPU**。citeturn27view0turn29view0

第二条是 **“高效 packing” 派**。LongAlign 研究的是长 SFT 阶段如何在长度分布极不均匀时提高效率。作者表明，**packing 与 sorted batching 都能把训练时间降到 naive batching 的一半以下**；同时，仅 packing 不够，因为 pack 中不同样本长度差异会让长样本对 loss 贡献更大，于是他们加入 **loss weighting**。在 ChatGLM3‑6B‑64k 上，**packing + loss weighting** 相比纯 packing 使 LongBench‑Chat 得分从 **5.76 提升到 6.21**，约 **+7.8%**。citeturn34view3turn34view4

### 超长文本质量过滤

在“超长文本往往噪声更多”这一现实问题上，Meta 的公开做法最值得引用，因为它给了可执行、可复现的过滤思想。Llama 3 的 web curation 管线包括：**URL/document/line 三层去重**，其中 line‑level 去重会删掉在每个 **30M 文档 bucket** 中出现超过 **6 次** 的行；还会用 **duplicated n‑gram coverage ratio** 去过滤日志/报错式重复文本，用 **dirty word counting** 过滤成人站点，用 **token‑distribution KL divergence** 识别 token 分布异常的文档；在模型式过滤上，则用 FastText/Wikipedia‑reference classifier 与基于 **Llama 2 标注、DistilRoBERTa 打分** 的质量分类器做二次筛选。citeturn18view0

这套做法对“长上下文数据”尤其关键，因为长书籍、长代码仓库、长网页 dump 的主要风险不是长度本身，而是**长而低密度**。ProLong 也从反面证明了这一点：虽然 CommonCrawl 中确实有很多长文，但在他们的 ablation 中，**书籍与代码仓库**作为长数据源显著优于 CommonCrawl、ArXiv 等来源；说明“只要够长就行”是错误假设。citeturn28view0

### 预训练阶段可直接放进汇报的对照表

| 来源 | 机构 | 数据策略关键信息 | 公开到的具体数字 | 对报告最有价值的 takeaway | 来源 |
|---|---|---|---|---|---|
| The Llama 3 Herd of Models | Meta | 先 8K 预训练，再做长上下文持续预训练；动态调整 multilingual/math/recent web/low-quality subsets；使用 document mask | 15T 左右语料；405B 主模型 15.6T tokens；长上下文阶段 800B tokens；8K→128K 分 6 阶段；数据 mix 为 50/25/17/8 | 工业界公开 recipe 中最完整的“长短兼顾式 CPT”范本 | citeturn18view0turn18view1turn20view0 |
| How to Train Long-Context Language Models Effectively | Princeton / ACL 2025 | 长数据=64K 单文档；短数据=packing 到 64K；长短混训；训练长度超过评测长度 | 最优 60% long + 40% short；ShortMix 为 25/25/10/10/10/10/10；总训练 40B tokens；最长 512K | 不要只喂长数据；真正有效的是“高质量短数据护航下的长数据持续训练” | citeturn26view0turn28view0turn27view0 |
| Gemini 1.5 Technical Report | Google DeepMind | 多模态、多语预训练；instruction tuning + human preference；面向 2M+、验证到 10M | 报告确认 2M+ context、正向结果到 10M；但未披露长短 data mix ratio | Google 公开承认长上下文专门训练，但对数据 recipe 细节披露极少 | citeturn9view0turn10view0 |
| DeepSeek‑V3 Technical Report | DeepSeek | 14.8T 预训练后做长上下文 extension；后接 SFT 和 RL | 14.8T tokens；长上下文 extension 为两个各 1000 step 的附加阶段 | DeepSeek 公开了“有额外长上下文训练”，但未公开长数据配比与 SFT/RL 长样本细节 | citeturn7search0turn7search3 |
| GPT‑4.1 in the API | OpenAI | 官方确认专门训练到 1M 上下文并学习“找相关信息、忽略干扰” | 1M context；未公开 data mix 或训练 schedule | OpenAI 公开了 capability 目标，但并未公开 recipe | citeturn6search2 |

### 我的判断

**行业已经不相信“纯长预训练能自动解决长上下文”了。** 从 Llama 3 到 ProLong，公开证据都在说明：长数据只是能力的一半，另一半是如何通过短数据、数据过滤、文档边界、长度 curriculum 去防“过拟合于长形式、退化于真实任务”。如果要用一句话概括这一部分，最合适的是：**长上下文预训练不是单一长度扩展，而是一个 data mixture + curriculum + masking 的联合优化问题**。citeturn18view1turn28view0turn27view0

## SFT 数据配方

### 一个核心反转

如果只看 2023 年前后的经验，很多人会以为“要获得长上下文指令能力，就必须准备海量长 SFT 数据”。但 2024–2026 的公开论文其实给出了一个更细的答案：**长 SFT 数据很有用，但作用方式比直觉更稀疏、更定向；很多情况下，少量高价值长 SFT 或甚至纯短 SFT 都能比“大量低质长 SFT”更好。**citeturn19view4turn27view2turn31view0

ProLong 在这方面最具颠覆性。作者测试了 UltraChat、Tulu‑v2、ShareGPT 以及多种 synthetic long instruction，结果是：**UltraChat 这样的标准短上下文 SFT 数据，已经能在其 setting 下给出最强的长任务表现；加入 synthetic long SFT 不仅没有继续变好，哪怕只加 1% 都会变差。** 论文因此最终选择**仅用 UltraChat** 作为 SFT 数据。citeturn27view2turn27view3

这并不意味着“长 SFT 不重要”，而是意味着：**长 SFT 的边际价值高度依赖数据质量、任务设计与 base model 的持续预训练阶段是否已经打牢。**

### 公开可复述的长 SFT recipe

Meta 的 Llama 3 报告给出了当前工业界最具体的长 SFT 合成方式之一。其 long‑context SFT 主要依赖三类 synthetic data：**长文 QA、长文 summarization、repo 级代码推理**。具体生成方式是：对长文先切成 **8K chunks**，用旧版模型在随机 chunk 上生成 QA；但训练时把**整篇文档作为上下文**喂给模型。对 summarization，则采用**层级摘要**：先用 8K 上下文模型做块摘要，再总结摘要，训练时再要求模型在整篇输入上做全局总结。对代码任务，则在 repo 内找出被至少 **5 个文件依赖**的关键文件，移除后让模型定位依赖文件并补写缺失代码。更重要的是，Meta 还做了序列长度分桶：**16K、32K、64K、128K**。最后通过 ablation 发现，**只要在原短 SFT 中混入 0.1% 的长 synthetic data，就能在短、长基准上取得最佳平衡**。citeturn19view4

这个 **0.1%** 很有启发性。它说明 frontier 模型的长 SFT 不是“把 SFT 规模整体拉成长”，而是把很小比例、但高度针对性的长任务样本作为**能力保持器/能力激活器**插入已有对话配方中。citeturn19view4

### 长 SFT 数据源的组成与比例

LongAlign 给出了另一个公开 recipe。它用 **76k ShareGPT** 作为短 instruction 数据，再把 **LongAlign‑10k** 这样的长 instruction 数据插进去；这些长样本来自 **9 个来源**，通过 Self‑Instruct 生成，长度覆盖 **8K–64K**。作者系统比较 0k、5k、10k、20k 长 instruction 的效果，发现**长 instruction 量越多越有助于长任务，而且不会明显伤害通用能力**；同时，**多样性** 比单纯数量更关键，因为 LongAlpaca‑12k 虽然规模不小，但只基于 papers/books 的 QA，来源和任务类型都更窄，泛化提升不如 LongAlign‑10k。citeturn33view0turn34view0turn34view1turn34view2turn34view5

LongWriter 则把重点放到了“**长输出**”而非长输入。该 ICLR 2025 论文发现，现有长上下文模型之所以普遍卡在 **2000 words 左右** 的输出上限，本质原因并不在预训练，而在 SFT 数据缺少长输出样本。于是作者用 AgentWrite + GPT‑4o 构造了 **LongWriter‑6k**，包含 **6000 条** 输出长度 **2k–32k words** 的 SFT 数据，并将其与 **180k chat SFT data** 混合训练，最终把模型可用输出长度拉到 **10,000+ words**。更有意思的是，其 ablation 还发现：把“写作计划”显式拼到输出前面并不一定更好；而直接用回译改写 instruction 的方法，会显著伤害长写作质量。citeturn32view0

LongMagpie 则代表第三种趋势：**用已经对齐好的长上下文模型自己“长出”长指令数据**。这篇 NeurIPS 2025 论文提出，给 aligned long‑context LLM 一个文档后，只提供特殊 user turn token，模型会自回归地产生与文档相关的问题；再把这个 query‑response 对与原文档、以及采样的其他文档组合起来，就可以自动生成单文档和多文档长 instruction 数据。作者报告了 **190k** 与 **450k** 两个数据规模，并提出 **p‑Mix**：先 prepending 短指令，再以概率 **P_L** 追加长指令，否则再追加短指令，如此循环直到接近最大长度，以防模型过拟合到长上下文模式。citeturn31view0

### 长 SFT 的质量控制

这一维度上，公开实践主要有三种。

第一种是**答案/轨迹验证型过滤**。Llama 3 在 reasoning data 上会用 outcome reward model 与 stepwise reward model 过滤中间推理错误，对更难题目还会用 **MCTS + stepwise RM** 生成有效 reasoning traces；在 code reasoning 数据上，用**代码执行反馈**淘汰错误轨迹。虽然这些做法不是专门为长上下文设计，但它们天然适用于 repo 级代码分析、长文推理等“长输入 + 长链条”的任务。citeturn15view0

第二种是**token‑level loss 校正**。ProLong 在 appendix 中指出，长 SFT 尤其是 synthetic 长指令会让不同 GPU 上“有效训练 token”极不均衡，因此普通 all‑reduce 如果按 sequence 平均而不是按 valid token 平均，会扭曲 domain proportion 与优化方向。于是作者将损失改为**对所有有效训练 token 求平均**。这实际上是长 SFT 里经常被忽视但极重要的一点：**长样本不仅改变内容分布，还改变 loss 统计学。**citeturn29view0

第三种是**LLM‑as‑a‑Judge / model‑based evaluation**。HELMET 基准在长 QA 和长 summarization 上使用 **GPT‑4o** 做模型式评估，而不是依赖 ROUGE 这类弱指标；LongBench‑Chat 也使用 GPT‑4 结合 ground truth 和 few‑shot 评分例子来打分。虽然这不是训练内 filtering，但它实际已经变成长 SFT 数据迭代的“离线 judge 基础设施”。citeturn26view0turn33view0

### SFT 阶段可直接放进汇报的对照表

| 工作 | 机构/会议 | 长 SFT 数据组成 | 明确数字 | 关键发现 | 来源 |
|---|---|---|---|---|---|
| Llama 3 Herd of Models | Meta | 长文 QA、层级摘要、repo 级代码依赖推理；长度桶 16K/32K/64K/128K | 长 SFT 只需混入 **0.1%** synthetic long-context data | 长 SFT 不需要大比例；极小剂量的高价值长样本就能“保住”长能力 | citeturn19view4 |
| ProLong | Princeton / ACL 2025 | 比较 UltraChat、Tulu‑v2、ShareGPT 与 synthetic long SFT | 发现加 **1% synthetic long SFT** 都会伤害其 setting 下结果 | 对强 base + 好 CPT 而言，短 SFT 可能比低质长 SFT 更优 | citeturn27view2turn27view3 |
| LongAlign | Tsinghua + Zhipu / Findings EMNLP 2024 | 76k ShareGPT + 5k/10k/20k LongAlign 长指令；9 sources；8K–64K | **LongAlign‑10k** 是核心公开集；**76k ShareGPT** 作短集 | 需要“多样而不是单一题型”的长 instruction data | citeturn34view0turn34view2turn34view5 |
| LongWriter | Tsinghua + Zhipu / ICLR 2025 | LongWriter‑6k 长输出 SFT + 一般 chat SFT | **6000** 条长输出样本，输出 **2k–32k words**；另混 **180k** 通用 SFT | 输出窗口大小高度受 SFT 数据最大输出长度支配 | citeturn32view0 |
| LongMagpie | IIE CAS + Xiaohongshu + Tsinghua / NeurIPS 2025 | 自回归 query 生成；单文档与多文档指令；p‑Mix 概率混短长指令 | 比较 **190k** 与 **450k** 样本 | 从“人工写长指令”转向“模型自举生成长指令”是 2025 年明显趋势 | citeturn31view0 |
| Hierarchical Synthetic Data Generation | Together AI / ICLR 2025 | 层级 QA、多文档组合、任意长度 synthetic instruction | 180K/350K/650K/1M 训练长度；样本数为 **2000/1280/600/200** | 百万 context 的长 SFT 主要靠多文档分层合成，而非天然百万长文 | citeturn37view3turn37view5 |

### 我的判断

这一部分最值得在报告中强调的“个人洞见”是：**行业不是从“大量长 SFT”里获得长能力，而是从“少量但任务化的长 SFT”里获得长能力。** 换句话说，长 SFT 更像是 patch 而不是 bulk。真正大的 token budget 往往花在持续预训练；SFT 阶段要解决的是**任务格式、全局读写行为、长输出遵循性**，而不是再灌一次海量长文本。citeturn19view4turn27view3turn32view0

## 强化学习任务与奖励设计

### 为什么 RL 在长上下文里突然重要

到 2025–2026，公开研究开始普遍承认：只靠 SFT 仍然不够解决长上下文中的“**找证据—用证据—抗干扰—保持长链条一致性**”问题。长上下文场景下，模型经常能靠参数记忆“蒙对”最后答案，导致 outcome‑only reward 无法真正训练出**grounding** 能力。LongRLVR 的摘要对此下了非常清晰的判断：**仅基于 final answer 的奖励过于稀疏，无法有效引导模型识别相关证据；因此必须加入 dense and verifiable context reward。**citeturn40search6

LongR 则从另一个角度重复了这个判断：长上下文 RL 如果只用稀疏结果奖励，会让模型在 context grounding 上出现 vanishing gradient，因此需要额外的**contextual dense reward**。citeturn39view0

### 哪些 RL 任务真正针对长上下文

目前公开的长上下文 RL 任务大致可以分四类。

第一类是**证据对齐 / grounding 型任务**。LongRLVR 把重点放在“奖励模型是否找到了正确 grounding information”；LongR 的 “Think‑and‑Read” 机制要求模型在推理链中交替生成 reasoning 与 document consultation；EAPO 则明确提出“Evidence‑Augmented Reasoning”，把 evidence extraction 看成长上下文 reasoning 的决定性瓶颈。citeturn40search6turn39view0turn40search5

第二类是**可验证的长链条检索推理任务**。LoongRL 提出的 **KeyChain** 把普通 multi‑hop QA 改造成高难长上下文版本：用 UUID chain 把真实问题藏在大量 distractor documents 中，模型必须一步步 trace chain、找到真问题、检索事实、再推理回答。作者还报告了一个非常强的结果：**在 16K context 上做 RL，学到的“plan‑retrieve‑reason‑recheck”模式可以泛化到 128K**，在 Qwen2.5‑7B/14B 上分别带来 **+23.5% / +21.1%** 的绝对提升。citeturn40search1

第三类是**文档重构 / 无监督 RLVR 任务**。Document Reconstruction Unlocks Scalable Long‑Context RLVR 的核心是：从长文档中移除若干段落并替换成 placeholder，让模型在候选段落集合中把缺失段落**识别并按正确顺序重建**。这个设计的妙处在于，它不需要人工 QA 标签，也不需要教师模型打分，却迫使模型学习全局叙事一致性。作者表明，这种无监督 RL 能明显提升 RULER，并在 LongBench v2 上取得合理增益。citeturn38search0

第四类是**超长输出规划与写作 RL**。LongWriter‑Zero 直接跳过 synthetic SFT，使用 RL 从 base model 出发，训练模型生成超长高质量文本。它的关键不是“找到答案”，而是让模型通过 RL 学会**规划、长度控制、结构一致性、长文本质量维持**。论文报告其从 Qwen2.5‑32B 出发，使用专门 reward models 约束长度、写作质量与格式，最终在 WritingBench 与 Arena‑Write 上优于传统 SFT 方法。citeturn38search3turn38search13

### 长序列 reward design 的主流路线

当前公开工作已经形成了三条比较清楚的 reward 路线。

第一条是**可验证上下文奖励**。LongRLVR 的方法论最典型：在答案 reward 之外，再增加一个**dense、verifiable 的 context reward**，显式奖励模型选择正确的 evidence/chunk。作者在摘要中给出的数字也很有说服力：14B 模型在 **RULER‑QA 73.17 → 88.90**，在 **LongBench v2 39.8 → 46.5**。citeturn40search6

第二条是**信息论式稠密奖励**。LongR 不是直接训练一个黑盒 reward model，而是使用 **Relative Information Gain** 去度量模型引用的 context segment 是否真的提高了答案所需的信息效用。它同时配合**渐进式 curriculum**，把训练上下文从 **16K 拉到 32K**。作者报告在 LongBench v2 上有约 **9%** 的改进，并且能迁移到 RULER、InfiniteBench 与不同 RL 算法。citeturn39view0

第三条是**证据质量 reward model + co‑evolution**。EAPO 认为 outcome reward 之所以不行，是因为它没监督 evidence quality，所以引入 **Group‑Relative Evidence Reward**，并加入 **Adaptive Reward‑Policy Co‑Evolution**，在训练中持续刷新 reward model 的判别能力。这个方向非常接近“长上下文 PRM/RM”的雏形。citeturn40search5turn40search11

### 长上下文 reward modeling 本身也要扩窗

LongRM 把问题再往前推进了一步：不是只问“policy 怎么学”，而是问“**reward model 自己能不能在 128K 场景下维持 context‑aware judgment**”。该论文提出 **Long‑RewardBench**，覆盖 Pairwise Comparison 与 Best‑of‑N 两类任务，并指出当前 SOTA generative RM 在长上下文场景下非常脆弱；随后提出 multi‑stage strategy，把任意模型升级成 robust LongRM。作者甚至声称其 **8B LongRM** 能超过很多 **70B** 级基线，并匹配 **Gemini 2.5 Pro** 的表现。无论数字最终是否还能被后续工作刷新，这篇 paper 的意义在于：**长上下文 RL 不只是 policy 问题，也是 long-context RM/PRM 问题。**citeturn38search2turn38search6

如果你想把 PRM 也纳入这一节，ACL 2025 的 **Dynamic and Generalizable Process Reward Modeling** 可以作为更通用的背景文献。它并非专门针对长上下文，但核心思想是用更通用、文本化、动态过程监督替代静态粗粒度打分；这与长上下文里需要**段落级、证据级、步骤级** reward 的方向是高度一致的。citeturn3search21

### RL 阶段可直接放进汇报的对照表

| 工作 | 类型 | 长上下文 RL 任务 | 奖励设计 | 可引用数字 | 来源 |
|---|---|---|---|---|---|
| LongRLVR | ICLR 2026 在投/公开预印本 | 长上下文 grounding 与 QA | final answer reward + **dense verifiable context reward** | 14B：RULER‑QA **73.17→88.90**；LongBench v2 **39.8→46.5** | citeturn40search6 |
| LongR | 2026 预印本 | Think‑and‑Read 长文推理 | **Relative Information Gain** 稠密奖励 + curriculum | LongBench v2 约 **+9%**；训练 curriculum **16K→32K** | citeturn39view0 |
| EAPO | 2026 预印本 | 证据增强长上下文推理 | **Group‑Relative Evidence Reward** + reward-policy co-evolution | 8 benchmarks 上优于 baseline；强调 evidence extraction 是瓶颈 | citeturn40search5turn40search11 |
| Document Reconstruction Unlocks Scalable Long-Context RLVR | 2026 预印本 | 段落重构、顺序恢复 | 无监督、可验证重构奖励 | 无需人工 QA 或教师模型；提升 RULER 与 LongBench v2 | citeturn38search0 |
| LoongRL | 2025 / ICLR 2026 | KeyChain 高难多跳长上下文 QA | 基于可验证链式任务的 RL | 在 **16K** 训练可泛化到 **128K**；7B/14B 绝对提升 **+23.5/+21.1** | citeturn40search1 |
| LongWriter‑Zero | 2025 预印本 | 超长写作 | 长度控制 + 写作质量 + 结构格式的组合奖励 | 从 Qwen2.5‑32B 起步，无 synthetic SFT 也能达到 SOTA 长写作 | citeturn38search3turn38search13 |
| LongRM | 2025 预印本 | 长上下文 RM/Best‑of‑N | multi‑stage LongRM 训练 | Long‑RewardBench 到 **128K**；8B LongRM 对比 70B 级强基线 | citeturn38search2turn38search6 |

### 我的判断

这条线最值得在报告里明确说出来：**长上下文 RL 的核心问题不是“答案对不对”，而是“模型有没有真的读到并使用正确证据”。** 因而，RL 阶段的真正创新点并非算法名词本身，而是从**结果奖励**转向**证据奖励、信息增益奖励、过程奖励、重构奖励**。这跟短链条数学推理中的 ORM/PRM 演进几乎是平行发展的。citeturn40search6turn39view0turn40search5turn38search2

## 行业公开配方对照与趋势判断

### 哪些公司真正公开了“recipe”

从公开透明度看，**Meta > Google/DeepSeek > OpenAI/Anthropic** 大致是当前格局。Meta 的 Llama 3 报告已经足够让人复述数据 pipeline、去重、过滤、mix、长度 curriculum、长 SFT 合成和部分 DPO 细节。Google 的 Gemini 1.5 报告承认有多模态多语预训练、instruction tuning 与 human preference，并报告了长上下文能力，但未公开长短文 mix、课程式 schedule、packing 细节。DeepSeek‑V3 公开了 14.8T 预训练、长上下文 extension 阶段与 SFT/RL 的存在，但没有给出长上下文数据配比。OpenAI GPT‑4.1 和 Anthropic Claude 公开信息更偏 capability announcement：OpenAI 只明确说过 **GPT‑4.1 被训练为能在完整 1M 上下文中可靠注意相关信息、忽略干扰**；Anthropic 对 Claude 3/4/4.6 公开的是 **200K–1M context、long-context retrieval/reasoning 改善**，但没有公开训练 recipe。citeturn18view0turn10view0turn7search3turn6search2turn8search4turn8search13turn8search1

### 一个适合在汇报中使用的产业结论

如果听众问“OpenAI、Anthropic、Google 到底是不是也在做这些数据配方”，最稳妥的回答是：**极大概率是，但公开文档几乎不披露 exact recipe**。Google 至少在技术报告里承认了多模态/多语预训练与 instruction tuning；OpenAI 明确承认对完整 **1M context** 做了专门训练；Anthropic 明确持续宣传长上下文 retrieval/reasoning 与 1M context 的产品能力；但真正能被学术报告精确引用的“数据数字”，仍主要来自 Meta 和公开论文。citeturn10view0turn6search2turn8search13turn8search1

### 一个更本质的趋势

一个很重要的趋势是，**行业明显在从“长文档建模”走向“长任务建模”**。前者只关心模型能不能在 128K/1M 输入上不崩；后者则关心它能不能完成**多文档问答、跨 repo 代码推理、长文摘要、长写作、长轨迹代理任务**。这也是为什么 2025–2026 的代表 work 都不再满足于 needle-in-a-haystack，而会使用 HELMET、LongBench v2、InfiniteBench、WritingBench、Long‑RewardBench 等更贴近真实使用的 benchmark。citeturn26view0turn31view0turn39view0turn38search2

## 适合十五到二十分钟报告的组织方式

### 推荐叙事主线

最适合 15–20 分钟 academic talk 的叙事，不是按公司讲，而是按**三段数据流水线**讲：

| 环节 | 你要讲的核心句 | 最强证据 |
|---|---|---|
| 持续预训练 | 长上下文不是“只喂长文本”，而是**长短混合 + 长度 curriculum + 文档边界控制** | Llama 3 的 50/25/17/8 与 8K→128K 六阶段；ProLong 的 60% long / 40% short 与 ShortMix | 
| SFT | 长 SFT 不是越多越好，而是**任务化、稀疏但高价值** | Llama 3 只混 **0.1%** 长 synthetic SFT；ProLong 中 **1%** synthetic long 反而伤害；LongWriter/LongMagpie 展示长 SFT 成功边界 |
| RL | 真正困难的是**读对证据并在长链条中维持信用分配** | LongRLVR、LongR、EAPO、LoongRL、LongRM | 

以上三句都可以被公开论文直接支撑。citeturn19view4turn28view0turn27view2turn32view0turn31view0turn40search6turn39view0turn40search5turn40search1turn38search2

### 一种可直接照搬的时间分配

| 幻灯片主题 | 建议时长 | 你应该强调什么 |
|---|---|---|
| 问题定义与边界 | 1.5–2 分钟 | 明确**不讨论架构扩展**，只谈 data recipe；说明为什么 PPL / NIAH 不够 | citeturn26view0 |
| 预训练/CPT | 4–5 分钟 | Llama 3 与 ProLong：long/short mix、books/repos、ShortMix、六阶段 curriculum、document mask | citeturn18view0turn18view1turn28view0turn20view0 |
| SFT recipe | 4–5 分钟 | Llama 3 的 0.1%、ProLong 的“短 SFT 即可”、LongAlign 的 10k 长指令、LongWriter/LongMagpie 的 synthetic pipeline | citeturn19view4turn27view2turn34view0turn32view0turn31view0 |
| RL & reward | 4–5 分钟 | 结果奖励为何失效；verifiable context reward / information gain / evidence reward / LongRM | citeturn40search6turn39view0turn40search5turn38search2 |
| 行业趋势与开放问题 | 2–3 分钟 | 为什么闭源公司几乎不披露 recipe；未来会向“长任务数据系统”继续演化 | citeturn6search2turn8search13turn10view0turn7search3 |

### 适合收尾的一页“个人观点”

如果你想在报告最后给出一个凝练、像 Principal Scientist 的判断，我建议用下面这段：

> **长期看，长上下文的竞争壁垒不会主要来自架构，而会来自数据系统。**  
> 真正有用的 recipe 不是把训练长度不断拉长，而是：  
> **高质量长文源选择 → 短长混配守住 base 能力 → 少量长 SFT 激活任务格式 → 稠密 RL 奖励逼模型真正“读、找、证、推”**。  
> 公开文献已经基本完成这个方向的验证。citeturn28view0turn19view4turn40search6turn39view0

## 开放问题与局限

当前公开文献仍有三个明显空白。第一，**闭源前沿模型在 128K–1M 长度下的 exact data ratio、packing 细节、过滤阈值仍几乎不公开**，所以任何把 GPT‑4.1、Claude 4.6、Gemini 1.5、DeepSeek‑V3 放在同一张 recipe 表里做精确比较的报告，都必须明确标注“未披露”。citeturn6search2turn8search13turn10view0turn7search3

第二，2025–2026 的 RL 长上下文论文增速很快，但其中一部分仍是**预印本或在投会议版本**。它们对方向判断非常有价值，但在学术 presentation 中，最好把它们区分为“emerging evidence”而非完全定型共识。citeturn40search6turn39view0turn40search5turn38search0turn38search2

第三，很多论文把“长上下文能力”与“长输出能力”“长代理轨迹能力”混在一起讨论。严格来说，这三者重叠但不完全等价。LongWriter/LongWriter‑Zero 证明了**长输出上限与 SFT/RL 一致强相关**，但它们解决的更多是“长生成”而不是“长检索推理”；在报告中最好专门说明这一点。citeturn32view0turn38search3