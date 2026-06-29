---
name: cluster-resources
description: VeriLong-RL 可用服务器、GPU 与已缓存模型资源
metadata:
  type: project
  originSessionId: 382b2af7-64ab-42e1-b374-e44bd8f47bc5
  observedAt: 2026-06-26
---

用户说明并经只读 SSH 到 `Tang-3-Wu` 查看 `/NAS/yesh` 确认：`Tang-1-Wu`、`Tang-2-Wu`、`Tang-3-Wu` 各有 `8×A40`，`Song-3-Wu` 有 `8×A100`。NAS 共享路径为 `/NAS/yesh`，各节点可连通进入。用户的工作文件夹在 `/NAS/yesh`，环境已配置；本地完成代码后可上传至服务器文件夹并调用显卡训练。

`/NAS/yesh/hf_cache/hub` 中已发现的可用模型/资源包括：

- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-7B`
- `meta-llama/Llama-3.1-8B`
- `meta-llama/Llama-2-7b-hf`
- `NousResearch/Llama-2-7b-hf`
- `selfrag/selfrag_llama2_7b`
- `facebook/contriever-msmarco`
- `sentence-transformers/all-MiniLM-L6-v2`

**Why:** VeriLong-RL 的训练实验需要现实可用的模型与 GPU 资源约束；Qwen2.5-7B 系列最适合作为 LoRA/QLoRA SFT 与 GRPO/RLVR 主线，Contriever/MiniLM 可作为检索/evidence baseline。

**How to apply:** 实施计划默认采用 Qwen2.5-7B / Qwen2.5-7B-Instruct 作为训练主线；API 模型做强模型对照；Llama/Self-RAG/检索模型作为可选 baseline。启动远程训练或上传代码前仍需明确当前步骤、路径和风险，不要无计划地占用 GPU。

Related: [[long-context-experiment-scope]], [[project-decisions]]
