# **长上下文大语言模型高级数据工程策略研究报告**

在当前大型语言模型（LLM）的发展进程中，向超长上下文（如128K至1M Token）的扩展范式已发生根本性转变。过去，学术界与工业界高度依赖底层架构的修改，例如旋转位置编码（RoPE）的外推、线性注意力机制的引入以及动态稀疏索引的开发，以缓解序列建模中二次计算复杂度的瓶颈。然而，来自前沿模型的大量实证研究表明，长上下文处理能力本质上是一个极其复杂的数据工程挑战。模型在扩展视野下进行检索、综合与推理的内在能力，很大程度上是在初始大规模预训练阶段就已经具备的潜在技能。要真正解锁并完善这一技能，无需对底层架构进行推倒重来，而是需要通过极其严密的动态数据课程调度、复杂的序列打包策略以及极高标准的长上下文后训练（Post-training）流水线来实现。  
本报告对包括Meta（Llama 3.1）、DeepSeek（DeepSeek-V3/R1）、阿里巴巴（Qwen2.5-1M）及顶尖学术机构在内的前沿机构所采用的最新长上下文数据工程策略进行了详尽的深度剖析。全文严格围绕三大核心支柱展开：预训练与持续预训练数据策略、监督微调（SFT）数据配方，以及强化学习（RL）任务与奖励设计。

## **第一支柱：预训练与持续预训练数据策略**

将模型的有效上下文窗口从标准预训练的4K至8K Token大幅扩展至128K甚至1M Token，需要经历一个被称为“持续预训练（Continual Pre-training）”的专属阶段。这一阶段在课程学习的节奏控制、数据域的混合比例以及多文档序列的打包方式上，均提出了前所未有的技术挑战。

### **动态课程学习与长度升级调度策略**

如果在扩展初期直接使用最大长度的序列对模型进行暴力训练，不仅计算成本极高，且往往会导致模型对短上下文基础能力的灾难性遗忘。因此，行业领先的开发者无一例外地采用了阶段性课程学习（Curriculum Learning）机制，在逐步增加序列长度的同时，动态调整数据分布。  
在Meta发布的Llama 3.1（405B参数，128K上下文）中，长上下文扩展消耗了约8000亿（800B）Token的计算预算1。该课程被严密地划分为六个增量长度适应阶段，从最初的8K窗口逐步攀升至128K2。向下一阶段推进的条件受到两项严格指标的限制：一是短上下文基准测试性能必须实现完全恢复；二是模型在当前长度下的“大海捞针（Needle-in-a-Haystack）”检索任务中必须达到接近完美的准确率4。在长上下文扩展的最后4000万（40M）Token训练期间，Llama 3.1采用了退火（Annealing）策略，将学习率线性衰减至零，同时对数据混合比例进行了动态干预，大幅上采样（Upsample）了极高质量的数据源，以稳固模型在超长距离上的注意力寻址能力3。  
与此形成对比的是，Qwen2.5-1M 采用了一种更为激进的五阶段渐进式扩展策略，以达到256K的训练上下文，并在推理阶段通过免训练的长度外推技术零样本扩展至100万 Token6。从4K基线开始，序列长度依次翻倍，经历32K、64K、128K，最终达到256K6。为了防止模型在超长序列上发生过拟合并丧失短距离局部连贯性，Qwen2.5-1M 在每个课程阶段的训练批次（Batch）中，严格维持了75%最大长度序列与25%短序列的混合比例，从而确保了跨尺度上下文泛化能力的平滑过渡7。  
DeepSeek-V3 则为128K扩展设计了极致压缩的双阶段课程。模型在仅仅1000个训练步（Steps）内即完成了从4K到32K的跨越，随后紧接着用另外1000步完成了32K到128K的扩展8。在随后的DeepSeek-V3.2-Exp实验中，这种快速适应能力得到了进一步挖掘：其稀疏注意力索引器仅使用21亿Token的密集计算（Dense）进行热身，随后立刻转入9437亿Token的稀疏训练阶段10。支撑这种极速收敛的核心秘诀在于，其持续预训练的数据分布与目标128K长上下文的真实数据分布实现了“完美对齐（Totally Aligned）”，确保了注意力头始终暴露在具有实际语义跨度的长距离依赖中，而非被毫无意义的填充符（Padding）所误导10。  
学术界的研究为这种低Token量、快速课程扩展提供了坚实的理论基础。在探讨有效长上下文缩放的重磅研究中，学者们证实了模型通过早期海量语料的预训练，实质上已经内化了在任意位置提取信息的能力12。只要严格保持原始预训练数据的领域平衡（Domain Balance），仅需5亿（500M）至50亿（5B）Token的持续预训练，就足以完全解锁模型在128K上下文下的检索能力13。

### **长上下文持续预训练的最优数据混合配比**

用于长上下文扩展的数据领域选择，将直接决定模型下游任务的性能表现。如果简单粗暴地对所有长文档（例如未经过滤的Common Crawl网页抓取数据）进行上采样，引入的结构性噪声将严重削弱模型的逻辑推理能力。  
在Llama 3.1 405B的底层预训练中，Meta采用了高达15万亿（15T）Token的庞大语料库2。其精确的数据混合配比为：50%的通用网络文本，25%的数学与推理语料，17%的多语言代码，以及8%的非英语多语言内容1。而在针对长上下文的消融实验中（如基于Llama 3架构的ProLong研究），研究人员对 SlimPajama 等开源数据集中的长文本分布进行了深度拆解16。分析表明，尽管书籍类数据提供了海量的连续长上下文（约332亿Token），但在GitHub代码仓库中，通过文件拼接形成的长文档蕴含着更为庞大的语料储备（约988亿Token）16。  
该研究得出了一个极为关键的结论：完全使用代码仓库进行长上下文训练，能够使模型在压力测试（如极端检索召回）中达到性能巅峰16。这是因为代码具有极其严格的语法树结构和跨文件变量追踪要求，强制注意力机制维持精确的长距离指针引用。相反，完全使用书籍数据进行训练，则能最大化提升上下文内学习（ICL）、长篇小说摘要和文档重排的性能，因为文学作品需要模型具备理解复杂叙事弧线和长距离语义连贯性的能力。最终的实验表明，将书籍与代码仓库以 1:1 的比例混合，作为占批次60%的长上下文核心数据，辅以40%的高质量短上下文混合数据，能够在所有128K评估任务中取得最完美的综合表现16。

| 数据来源 (SlimPajama / Stack) | 可用的长序列 Token总量 (≥ 64K) | 核心能力提升维度 |
| :---- | :---- | :---- |
| 代码仓库 (Code Repositories) | 988 亿 | 精确信息召回、长距离变量指针引用追踪 |
| 书籍文献 (Books) | 332 亿 | 上下文内学习 (ICL)、语义连贯性、长文摘要 |
| 网页抓取 (Common Crawl) | 153 亿 | 通用常识检索与基础知识覆盖 |
| 学术论文 (ArXiv Papers) | 52 亿 | 科学推理、跨图表多跳综合分析 |
| **最优长上下文混合配比** | **1:1 代码与书籍比例** | **实现硬性召回与柔性语义推理的完美平衡** |

### **数据打包策略与跨文档注意力污染控制**

为了将GPU的算力利用率推向极致，并彻底消除填充（Padding）Token带来的计算浪费，数据工程中通常将多个长度不一的独立文档拼接（Pack）成一个达到最大上下文长度的单一序列。然而，标准的文件打包技术会导致注意力机制跨越文档边界，产生严重的语义污染（Attention Contamination）。  
在长上下文持续预训练期间，Llama 3.1 采用了特殊的文档边界符来分隔拼接的文本，并令人意外地选择关闭了跨文档的因果注意力掩码（Cross-document Attention Mask）18。这一策略允许模型在整个128K的输入序列中自由进行全局注意力计算，其核心目的在于强制Transformer架构通过内在的语义理解来自动划定信息边界，而不是依赖人为施加的硬性物理掩码18。与此截然不同，DeepSeek-V3 极其看重“数据完整性（Data Integrity）”，在打包过程中严格实施了边界隔离，确保模型对结构化文档完整性的理解不会被相邻的无关文本池所毒化10。此外，DeepSeek还在文档层面大量应用了中间填充（Fill-in-the-Middle, FIM）格式，采用前缀-后缀-中间（PSM）的结构布局，并以0.1的概率应用此格式，以原生方式训练超长跨度的代码填补与逻辑修复能力8。  
包含变长文档的打包计算在数学层面上引入了严重的损失计算偏差。清华大学团队提出的 LongAlign 框架敏锐地指出了这一问题：标准的打包训练会将包含不同原始序列数量的数据包赋予相等的梯度权重，这导致模型在优化过程中不可避免地向那些偶然打包得极其密集的短文档群倾斜20。为了修正这一统计学偏差，LongAlign 引入了损失加权（Loss Weighting）机制，确保包内每个序列对最终梯度的贡献基于其真实长度得到平衡；同时，开发了排序批处理（Sorted Batching）算法，预先将长度相似的文档进行聚类，从而将批次内的闲置等待时间降至最低20。  
在这一技术路径的延伸上，NeurIPS 2025 所提出的层次化平衡打包（Hierarchical Balance Packing, HBP）算法，彻底解决了由注意力计算复杂度引发的系统级负载不均问题22。由于自注意力的计算复杂度随序列长度呈二次方暴增，包含少数几篇海量长文档的批次所消耗的浮点运算量，将指数级地高于包含大量短文档的同长度批次，这在分布式数据并行（DP）集群中会引发灾难性的流水线停滞22。HBP 通过注意力复杂度启发式算法，构建了多层级的打包组，成功将注意力平衡率（Attention Balance Ratio, ABR）从0.506断崖式降低至0.28823。同时，HBP引入了基于全局平均Token长度的稳定损失归一化器（Stable Loss Normalizer），完美中和了局部Token均值或样本均值归一化技术所带来的统计偏差23。

### **超长文本的质量过滤机制**

为了确保进入预训练流水线的每一兆Token都具备极高的信息密度，前沿实验室部署了多维度的自动化清洗矩阵。对于Llama 3.1，Meta不仅使用了fastText启发式分类器，还直接动用早期版本的Llama大模型进行质量打分（Quality Scoring）24。在处理网页抓取数据时，极度小心地剥离了Markdown标记以及容易引起干扰的HTML标签，但对于包含数学公式和代码逻辑的页面则采用特殊的解析器以保留其结构连贯性3。更为关键的是语义去重（Semantic Deduplication），团队利用RoBERTa模型对长文段落进行高维空间聚类，剔除余弦相似度极高的冗余文本，从源头上阻断了长上下文模型容易出现的复读机退化现象3。  
**核心趋势与技术演进：** 业界已经达成了高度共识：彻底摒弃从零开始耗费数千亿Token的纯长上下文预训练范式。当前的绝对主流是高侵入性、低Token量（10亿至100亿）的精准课程扩展，结合精心调配的1:1代码与书籍混合域。长度泛化不再被视为一种需要重新搭建神经回路的新能力，而是基础模型本就具备、只需通过局部数据分布偏移（Distribution Shift）结合损失加权打包技术即可轻易解锁的潜在本能。

## **第二支柱：监督微调（SFT）数据配方**

基础模型的上下文窗口一旦在预训练阶段被成功拓宽，随之而来的便是至关重要的监督微调（SFT）阶段。此阶段的核心任务是教导模型如何遵循长篇幅指令，如何在动辄数十万字的输入中综合多源文档，并彻底根除困扰早期模型的“中间迷失（Lost in the Middle）”顽疾。

### **数据构成与多阶段微调范式**

在SFT阶段，如果将长上下文指令数据与标准的短上下文指令数据同时混合注入，极易导致模型在处理日常简短对话时出现响应迟缓或格式崩溃。为化解这一冲突，前沿模型普遍采用了严密的多阶段SFT流水线。  
Qwen2.5-1M 模型家族在这方面实施了极其严格的双阶段SFT架构7。

* **第一阶段（短指令锁定）：** 团队首先使用长度被严格限制在32,768 Token以内的高质量短序列指令数据，对预训练模型进行首轮微调7。这一步旨在为模型打下坚实的指令遵循烙印，确立其在格式遵从和高频短对话模式下的绝对连贯性。  
* **第二阶段（混合上下文注入）：** 在核心行为范式被锁定后，训练数据无缝过渡到一种高度控制的混合模式，包含短指令（不超过32K）以及跨度高达256K的超长合成指令27。这种双阶段防御机制确保了模型在能力图谱上的全维泛化，使其在能够咀嚼海量分析报告的同时，绝不会遭受对短上下文能力的灾难性遗忘7。

在具体的数据比例调配上，Qwen2.5-Coder 展现了专门针对长上下文代码模型的配方艺术。其采用了 7:2:1 的精妙比例——70%的代码指令、20%的通用文本指令以及10%的数学逻辑指令29。引入20%和10%的非代码数据，不仅没有稀释其代码生成能力，反而成为维持其通用自然语言理解与长链条数理推理能力的关键锚点31。同样，在Cerebras公布的利用100亿Token扩展Llama3-8B上下文的开源配方中，其SFT数据呈现出高度倾斜的分布：77.5%为全合成长数据（基于RAFT构建的对话QA与RAG任务），仅有22.5%依赖于传统的开源长指令数据集（如LongDataCollections与LongAlpaca）32。

### **前沿模型的超长指令合成机制**

面对128K甚至1M的上下文窗口，依赖人类标注员进行高质量SFT数据的生产在经济和认知层面上均已失效。因此，现代长上下文SFT数据几乎完全依赖于由高级API驱动的自治智能体（Agent）合成。  
Meta的 Llama 3.1 405B 流水线代表了当前代码与数学推理合成数据的行业巅峰2。为了构建无懈可击的长上下文编程语料，Meta独创性地部署了“执行反馈（Execution Feedback）”闭环系统，成功合成了逾百万级的高质量代码对话：

1. **问题描述生成（Problem Description Generation）：** 利用较小参数量的模型（如8B或70B）从原始预训练语料库中随机抽样长尾代码片段，并基于这些片段逆向生成极其复杂的长篇编程需求说明2。  
2. **多文件解决方案生成（Solution Generation）：** 旗舰级405B模型接管任务，横跨多个文件系统生成包含完整逻辑链路的解决方案，同时被强制要求在代码注释中输出思维链（Chain-of-Thought）5。  
3. **多维正确性分析（Correctness Analysis）：** 生成的代码必须经过严酷的审查。静态分析阶段，利用解析器和代码检测工具（Linter）排查未初始化变量和类型定义错误；动态分析阶段，模型需自行编写单元测试，并在隔离的容器环境中执行，以捕捉深层的语义和运行时错误2。  
4. **错误反馈与自迭代修正（Error Feedback and Iterative Self-Correction）：** 一旦在上述任何环节发生崩溃，堆栈跟踪信息和Linter报错日志将被直接追加进模型的超长上下文输入中，迫使模型基于错误提示进行自我修正2。最终，包括试错、反思和成功修复在内的完整轨迹，被打包成一条极具价值的长上下文SFT样本。

对于强制要求模型在全上下文窗口内进行高频信息搜索的任务，研究人员高度依赖检索增强微调（Retrieval Augmented Fine-Tuning, RAFT）范式32。传统的问答数据集仅提供一段相关段落与问题；而在基于RAFT构建的长上下文合成数据中，提示词（Prompt）被故意设计为高噪音的结构：(真实段落, 干扰项\_1, ..., 干扰项\_k, 问题, 答案)32。这里的干扰项（Distractors）并非随机乱码，而是通过高维嵌入相似度计算得出的、与真实段落语义极为相近的混淆文本。真实段落随后会被随机插入到数以万计的Token矩阵中32。这种堪称“毒性”的合成构造，通过物理强制力惩罚了依赖位置偏见或浅层关键词匹配的模型，迫使Transformer的注意力头对全序列跨度进行极高强度的语义判别。

| 长上下文 SFT 任务类别 | 合成数据构建机制 | 核心评估能力 |
| :---- | :---- | :---- |
| **代码存储库级推理** | 执行反馈循环 (解析器静态分析 \+ 容器化单元测试动态分析 \+ 自纠错轨迹) | 跨文件上下文理解、Bug定位、长链执行追踪 |
| **抗干扰语义检索** | RAFT架构 (真实证据嵌入至基于语义相似度筛选的巨量强混淆干扰项中) | 克服“中间迷失”现象、抗位置偏见、全局注意力分配 |
| **多跳多文档综合** | 提取不连贯事实构建问答，强迫模型梳理信息碎片的实体拓扑关系 | 信息综合、矛盾消解、多点事实溯源 |
| **超长对话格式遵循** | 意图标记注入 (Intention-tagging)，使用拒绝采样保证回答调性与排版约束 | 格式忠实度、长程对话记忆连贯性 |

### **质量至上与LLM作为裁判（LLM-as-a-Judge）**

在长上下文合成数据域中，信噪比通常处于极低水平。为剔除劣质数据，Meta为Llama 3.1精心构建了立体式的剪枝机制3。他们开发了名为 Instag 的意图标记系统，调动70B参数模型去识别并标记合成提示词中蕴含的复杂意图层级，同时辅以独立的难度打分（Difficulty Scoring）网络3。在答案清洗环节，全面引入拒绝采样（Rejection Sampling），由专门训练的奖励模型（Reward Model）对每个提示词生成的10至30个候选输出进行严苛评审，仅保留最高分样本3。最后，利用RoBERTa模型进行特征聚类，执行激进的语义去重，将批次内余弦相似度超标的冗余数据彻底抹除3。  
Qwen2.5-Coder 团队对文本-代码对齐数据的处理同样苛刻，采用了一套迭代式的四阶过滤流程（4-Stage Filtering Process）。该流程大量使用经过高度优化的轻量级模型对代码基础数据进行清单打分（Checklist-based Scoring）29。实证数据表明，仅凭借这一套过滤流程，就将1.5B模型在 HumanEval 基准上的通过率从41.6%生生拉升至46.8%，无可辩驳地证明了在长上下文中，激进的数据剪枝带来的收益远胜于盲目的数据堆砌31。  
**核心趋势与技术演进：** 纯人类标注在长上下文SFT领域已被彻底淘汰。当前的战略高地已转移至基于执行反馈（如代码编译、环境交互）的闭环智能体合成管线。更加关键的是，业界已经摸索出了黄金法则——SFT必须实施严格的分层阶段化管理：先用高质量短文本锁定模型的行为对齐与基本逻辑，然后再注入包含对抗性干扰项的长序列合成数据。这不仅是拓展长度的必要手段，更是防止通用能力坍塌的最后防线。

## **第三支柱：强化学习（RL）任务与奖励设计**

人类反馈强化学习（RLHF）及其衍生变体是模型对齐的最终战役。然而，当上下文窗口飙升至10万甚至100万Token时，RL面临着致命的“信用分配（Credit Assignment）”危机。当模型的输出推理链绵延数千个Token时，究竟是哪个关键的中间步骤导致了最终奖励的得失？这在数学和工程上都构成了巨大的挑战。

### **长上下文RL向“数据中心”范式的回归**

历史上，学术界应对长上下文RL瓶颈的思路是构建越来越复杂的辅助奖励信号，例如试图评估推理中间步骤的过程监督奖励模型（Process-supervised Reward Models, PRMs）34。然而，来自Qwen团队的重磅研究《超越奖励工程：长上下文强化学习的数据配方（Beyond Reward Engineering: A Data Recipe for Long-Context Reinforcement Learning）》（2026年）彻底颠覆了这一路径。该论文强有力地证明：如果RL数据集构建得足够精妙，极其复杂的奖励工程完全是多此一举34。  
研究人员证实，使用最基础的、仅依赖结果的组相对策略优化（GRPO），只要配合高度提纯的1.4万条（14K）专属数据集，就能实现长上下文推理能力的跃升。该数据配方精准狙击了长上下文模型的三个致命死穴35：

1. **FuzzyNeedle（剥离词汇捷径的模糊检索）：** 传统的“大海捞针”任务由于依赖精准的字符串匹配，LLM往往利用浅层的局部注意力模式即可作弊通关34。FuzzyNeedle 彻底切断了词汇重叠，通过Wikidata的“IS-A”本体关系对隐蔽事实进行多重改写。例如，隐藏的证据是“爱丽丝非常喜欢三文鱼的口感”，而问题却刁钻地问“谁热衷于食用鱼类？”。模型被强迫在大量同构干扰句（例如关于其他蔬菜或肉类的喜好描述）中，进行深度的本体关系解析，从而彻底粉碎了基于关键词匹配的作弊可能34。  
2. **MultiNeedle（多目标序列计数与辨别）：** 该数据集将多个关于相似主题的独立对话无序地拼接成极长的上下文。提示词要求模型精确提取特定主题下的“第K个”对话。这迫使模型不能在找到第一个匹配项后就停止注意力搜索，而是必须穷尽遍历、在整个128K序列中维持一个隐式的计数器，并对高度同质化的对话进行微小的特征辨别34。  
3. **多证据跨度综合推理（Multi-evidence Synthesis and Reasoning）：** 此类任务要求模型将散落在上下文两端、相隔数万Token的孤立事实片段进行逻辑拼图与综合验证34。

在这一特定数据配方的驱动下，无需昂贵的PRMs加持，Qwen3模型在七个主流长上下文基准测试中取得了惊人的平均 \+7.2 分的跃升37。

### **长序列奖励建模、GRPO与可验证任务环境**

DeepSeek-V3及其主打推理的衍生体DeepSeek-R1，通过将训练数据合成与执行环境评估彻底融合，重新定义了强化学习数据的收集规模10。DeepSeek构建了一个堪称庞大的智能体合成矩阵，涵盖了逾1,800个独立的沙盒环境以及超过85,000个独特的交互提示词10。  
在强化学习优化算法的选择上，DeepSeek果断摒弃了需要维持庞大独立评论家（Critic）网络的近端策略优化（PPO）算法，全面转向更为轻量且高效的组相对策略优化（GRPO）11。针对长上下文与智能体多步执行任务，奖励信号被划分为截然不同的两极：

* **基于规则的确定性奖励（Rule-Based Rewards）：** 对于具备明确验证标准的领域（如数学证明、多步代码执行追踪），奖励是绝对刚性的。模型生成的代码补丁一旦通过了严格的沙盒编译，并且在执行所有单元测试后没有在长代码库中引发任何回归缺陷（Regressions），便能获得满分奖励10。由于环境提供了不可篡改的“绝对真理（Ground-truth）”信号，模型试图欺骗奖励函数（Reward Hacking）的漏洞被彻底封死。  
* **生成式奖励模型（Generative Reward Models）：** 针对那些主观性较强、缺乏执行环境反馈的长文总结或创意逻辑任务，DeepSeek引入了基于严密评价量表的生成式奖励模型10。为了防止AI陷入逻辑循环，平台强制要求领域内的人类专家在环（Human-in-the-loop）对量表进行高频审计，专门针对模型试图迎合奖励模型的边缘情况进行人工干预与重裁决10。

### **强化学习中震撼的“由短及长”泛化现象**

在近期的长上下文数据工程研究中，最具颠覆性的反直觉发现来自于Qwen2.5-1M的技术报告。研究团队在消融实验中证实：仅仅使用被严格限制在8,000 Token 以内的短文本进行强化学习训练，其带来的对齐效果能够近乎完美地泛化至100万 Token 的超长上下文推理中27。  
这一实证发现直接改写了长上下文RL的算力经济学。它雄辩地证明，诸如语气把控、有用性衡量、拒绝越狱指令以及分步推理的格式忠诚度等人类偏好对齐特征，本质上是一种完全“跨越长度域”的内在认知能力。只要模型在极端的预训练和SFT阶段已经搭建起了稳固的长距离机械注意力跨度，它就能将基于8K上下文习得的伦理和逻辑准则，本能地平移至1M上下文中去执行27。因此，未来的训练管线可以完全规避在超长序列上运行高昂RL算法所带来的海量显存（VRAM）压力，专注于短序列的极致对齐即可。  
**核心趋势与技术演进：** 长上下文强化学习的技术前沿正在经历一次重大的航向修正。行业正在摆脱算法层面的无底洞（如开发越来越庞大复杂的中间步骤评估模型PRMs），回归到最纯粹的“数据策展与环境设计”。通过将高度确定性的沙盒验证机制引入强化学习环境，并利用如FuzzyNeedle这种充满语义对抗陷阱的数据集，研究者发现，最基础的GRPO算法已足以引爆模型在长上下文下的推理潜力。而“对齐（短上下文RL）与跨度（长上下文预训练/SFT）可完全解耦”的发现，更是将大模型训练效率推向了一个前所未有的新纪元。

## **结论**

长上下文大语言模型的工程演进已经脱离了单纯的架构堆砌阶段，全面进化为一门对精确度要求极高的高级数据策展学科。综合截至2026年全球顶级开源与闭源模型的技术脉络，可提炼出以下确定性的数据工程蓝图：

1. **在预训练阶段：** 盲目的海量长文本训练已被证伪。仅需不到50亿Token的精准持续预训练，即可完全解锁模型的长上下文视野。关键在于严格维持代码仓库与书籍文献1:1的黄金配比，并强制引入基于损失加权的层次化平衡打包（HBP）机制，以根除计算复杂度引发的梯度偏见。  
2. **在监督微调（SFT）阶段：** 必须坚守分阶段微调的铁律。在固化短指令能力后，方可引入由智能体执行反馈闭环（编译验证+自纠错）生成的高质量合成数据。其中，带有密集干扰项的RAFT范式是根除模型位置偏见的最有效手段。  
3. **在强化学习（RL）阶段：** 面向长上下文的繁复奖励工程（如PRMs）已逐渐边缘化。最优解是依托基础的GRPO算法，配合不可篡改的代码执行沙盒以及深度去词汇化（FuzzyNeedle）的对抗性数据集。由于人类偏好对齐展现出卓越的“短及长泛化”特性，仅在8K序列上执行的RL便足以完美统治100万 Token 的广袤语义空间。

这三大数据工程支柱不仅为破解长上下文灾难性遗忘与注意力崩溃提供了标准答案，更标志着新一代生成式AI正从被动吞吐字符的“上下文阅读器”，真正蜕变为能够在无限信息视界中进行长程谋篇布局的“自治推理中枢”。

#### **引用的著作**

1. LLaMA 3.1 8B Model Overview \- Emergent Mind, [https://www.emergentmind.com/topics/llama-3-1-8b-model-2348e96d-21f2-48dc-88c2-90e5888c626b](https://www.emergentmind.com/topics/llama-3-1-8b-model-2348e96d-21f2-48dc-88c2-90e5888c626b)  
2. \[2407.21783\] The Llama 3 Herd of Models \- ar5iv \- arXiv, [https://ar5iv.labs.arxiv.org/html/2407.21783](https://ar5iv.labs.arxiv.org/html/2407.21783)  
3. Notes on 'The Llama 3 Herd of Models' | Fan Pu Zeng, [https://fanpu.io/blog/2024/llama-3.1-technical-report-notes/](https://fanpu.io/blog/2024/llama-3.1-technical-report-notes/)  
4. Papers Explained 187b: Llama 3.1 \- Ritvik Rastogi \- Medium, [https://ritvik19.medium.com/papers-explained-187b-llama-3-1-f0fb06898c59](https://ritvik19.medium.com/papers-explained-187b-llama-3-1-f0fb06898c59)  
5. arXiv Dive: How Meta Trained Llama 3.1 \- Oxen.ai, [https://ghost.oxen.ai/llama-3-1-herd-of-models/](https://ghost.oxen.ai/llama-3-1-herd-of-models/)  
6. Qwen2.5-1M Technical Report, [https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/Qwen2\_5\_1M\_Technical\_Report.pdf](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/Qwen2_5_1M_Technical_Report.pdf)  
7. Qwen2.5-1M: The First Open-Source AI Model with a 1 Million Token Context Window, [https://ajithp.com/2025/02/02/qwen2-5-1m-open-source-ai-1-million-token-context/](https://ajithp.com/2025/02/02/qwen2-5-1m-open-source-ai-1-million-token-context/)  
8. DeepSeek Explained 5: DeepSeek-V3-Base | by Shirley Li | Data Science Collective, [https://medium.com/data-science-collective/deepseek-explained-5-deepseek-v3-base-86c078ed5504](https://medium.com/data-science-collective/deepseek-explained-5-deepseek-v3-base-86c078ed5504)  
9. DeepSeek-V3 Technical Report \- arXiv, [https://arxiv.org/html/2412.19437v2](https://arxiv.org/html/2412.19437v2)  
10. Data Story: How the Corpus, Synthetic Pipelines, and Evaluation Shaped Deepseek V3.2, [https://kili-technology.com/blog/data-story-deepseek-v3-2](https://kili-technology.com/blog/data-story-deepseek-v3-2)  
11. Boosting Long-Context Efficiency with DeepSeek Sparse Attention, [https://aarnphm.xyz/thoughts/papers/DeepSeek\_V3\_2.pdf](https://aarnphm.xyz/thoughts/papers/DeepSeek_V3_2.pdf)  
12. Data Engineering for Scaling Language Models to 128K Context \- GitHub, [https://raw.githubusercontent.com/mlresearch/v235/main/assets/fu24d/fu24d.pdf](https://raw.githubusercontent.com/mlresearch/v235/main/assets/fu24d/fu24d.pdf)  
13. \[2402.10171\] Data Engineering for Scaling Language Models to 128K Context \- arXiv, [https://arxiv.org/abs/2402.10171](https://arxiv.org/abs/2402.10171)  
14. Data Engineering for Scaling Language Models to 128K Context \- Hugging Face, [https://huggingface.co/papers/2402.10171](https://huggingface.co/papers/2402.10171)  
15. Llama 3.1 Technical Report Overview | PDF | Artificial Intelligence \- Scribd, [https://www.scribd.com/document/824747336/Llama3](https://www.scribd.com/document/824747336/Llama3)  
16. How to Train Long-Context Language Models (Effectively) \- arXiv, [https://arxiv.org/html/2410.02660v4](https://arxiv.org/html/2410.02660v4)  
17. How to Train Long-Context Language Models (Effectively) \- arXiv, [https://arxiv.org/html/2410.02660v1](https://arxiv.org/html/2410.02660v1)  
18. From 128K to 4M: Efficient Training of Ultra-Long Context Large Language Models \- arXiv, [https://arxiv.org/html/2504.06214v1](https://arxiv.org/html/2504.06214v1)  
19. The Llama 3 Herd of Models. Paper Review | by Eleventh Hour Enthusiast | Medium, [https://medium.com/@EleventhHourEnthusiast/the-llama-3-herd-of-models-2f62252ce1c8](https://medium.com/@EleventhHourEnthusiast/the-llama-3-herd-of-models-2f62252ce1c8)  
20. arXiv:2401.18058v1 \[cs.CL\] 31 Jan 2024, [https://arxiv.org/pdf/2401.18058](https://arxiv.org/pdf/2401.18058)  
21. LongAlign/README.md at main \- GitHub, [https://github.com/THUDM/LongAlign/blob/main/README.md](https://github.com/THUDM/LongAlign/blob/main/README.md)  
22. Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM \- arXiv, [https://arxiv.org/html/2503.07680v1](https://arxiv.org/html/2503.07680v1)  
23. NeurIPS Poster Hierachical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM, [https://neurips.cc/virtual/2025/poster/119808](https://neurips.cc/virtual/2025/poster/119808)  
24. Llama 3.1 405B: Specifications and GPU VRAM Requirements \- ApX Machine Learning, [https://apxml.com/models/llama-3-1-405b](https://apxml.com/models/llama-3-1-405b)  
25. Introducing Llama 3.1: Our most capable models to date \- Meta AI, [https://ai.meta.com/blog/meta-llama-3-1/](https://ai.meta.com/blog/meta-llama-3-1/)  
26. LCLM-Horizon/A-Comprehensive-Survey-For-Long-Context-Language-Modeling \- GitHub, [https://github.com/LCLM-Horizon/A-Comprehensive-Survey-For-Long-Context-Language-Modeling](https://github.com/LCLM-Horizon/A-Comprehensive-Survey-For-Long-Context-Language-Modeling)  
27. Qwen2.5-1M: Deploy Your Own Qwen with Context Length up to 1M Tokens, [https://qwenlm.github.io/blog/qwen2.5-1m/](https://qwenlm.github.io/blog/qwen2.5-1m/)  
28. Alibaba's Qwen2.5-1M: Open-Source Model with 1M Token Contexts Released, [https://www.ainews.com/p/alibaba-s-qwen2-5-1m-open-source-model-with-1m-token-contexts-released](https://www.ainews.com/p/alibaba-s-qwen2-5-1m-open-source-model-with-1m-token-contexts-released)  
29. Qwen2.5-Coder Technical Report \- arXiv, [https://arxiv.org/html/2409.12186v1](https://arxiv.org/html/2409.12186v1)  
30. Qwen2.5 Model Family Overview \- Emergent Mind, [https://www.emergentmind.com/topics/qwen2-5-model-family](https://www.emergentmind.com/topics/qwen2-5-model-family)  
31. Qwen2.5-Coder Technical Report \[Quick Review\] \- Liner, [https://liner.com/review/qwen25coder-technical-report](https://liner.com/review/qwen25coder-technical-report)  
32. Extending LLM context with 99% less training tokens \- Cerebras, [https://www.cerebras.ai/blog/extending-llm-context-with-99-less-training-tokens](https://www.cerebras.ai/blog/extending-llm-context-with-99-less-training-tokens)  
33. The Llama 3 Herd of Models, [https://self-supervised.cs.jhu.edu/fa2024/files/presentations/9-12-llama3-Zhao-Huang.pdf](https://self-supervised.cs.jhu.edu/fa2024/files/presentations/9-12-llama3-Zhao-Huang.pdf)  
34. Beyond Reward Engineering: A Data Recipe for Long-Context Reinforcement Learning, [https://arxiv.org/html/2606.18831v1](https://arxiv.org/html/2606.18831v1)  
35. Beyond Reward Engineering: A Data Recipe for Long-Context Reinforcement Learning, [https://www.researchgate.net/publication/407241283\_Beyond\_Reward\_Engineering\_A\_Data\_Recipe\_for\_Long-Context\_Reinforcement\_Learning](https://www.researchgate.net/publication/407241283_Beyond_Reward_Engineering_A_Data_Recipe_for_Long-Context_Reinforcement_Learning)  
36. Beyond Reward Engineering: A Data Recipe for Long-Context Reinforcement Learning, [https://www.opentrain.ai/papers/beyond-reward-engineering-a-data-recipe-for-long-context-reinforcement-learning--arxiv-2606.18831/](https://www.opentrain.ai/papers/beyond-reward-engineering-a-data-recipe-for-long-context-reinforcement-learning--arxiv-2606.18831/)  
37. Beyond Reward Engineering: A Data Recipe for Long-Context Reinforcement Learning \- arXiv, [https://arxiv.org/pdf/2606.18831](https://arxiv.org/pdf/2606.18831)  
38. DeepSeek-V3 Technical Report \- The VITALab website, [https://vitalab.github.io/article/2025/02/11/DeepSeekV3.html](https://vitalab.github.io/article/2025/02/11/DeepSeekV3.html)