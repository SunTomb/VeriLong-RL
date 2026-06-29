---
name: cluster-gpu-usage-tang-song
description: Tang/Song 集群 GPU 使用约定与 VeriLong-RL 环境配置
metadata:
  type: project
---

集群通过 SSH host alias 访问（如 `Tang-2-Wu`、`Song-3-Wu`），统一 IP 210.45.70.34，登录用户 wujcan。

**GPU 使用约定（重要）：** GPU 是共享资源，所有卡常被他人占用。提交作业前必须 `nvidia-smi` 看显存/利用率，**选显存占用最低的卡**共用，不要假设任何卡是空的。务必显式 `export CUDA_VISIBLE_DEVICES=<idx>` 锁定单卡。曾误判 GPU0 空闲（实为他人 25GB/100% 进程），应避免。用户 2026-06 指示：所有卡占用时可用 GPU7（显存最低）共用。完事或换卡前用 `pkill -f <script>` 释放自己的进程。

**环境（仿 MemUpdateBench 约定，勿重复下载环境包）：**
- 共享 conda 解释器 `/NAS/yesh/miniconda3`，复用 env **`gmsra`**（transformers 4.46.3、accelerate 1.13；无 vLLM）。已额外装入 pydantic 2.13 + pyyaml。
- **torch/驱动兼容（踩过的坑）：** Tang-2 驱动 535.129.03，最高支持 CUDA 12.2。gmsra 原本是 torch 2.12.0+cu130（需 CUDA 13），导致 `torch.cuda.is_available()=False`、模型回退 CPU 跑（34 核空转、不碰 GPU、出不了结果）。换卡无用（驱动整机级）。2026-06 用户批准在 gmsra 内降级 torch 到 cu121（torch 2.5.1+cu121 / torchvision 0.20.1）。降级前已备份 `pip freeze` 到 `/NAS/yesh/VeriLong-RL/gmsra_pip_freeze_backup.txt`（原始 torch==2.12.0 / torchvision==0.27.0 / triton==3.7.0）可回滚。
- 项目入口 `/NAS/yesh/VeriLong-RL/activate.sh`：activate gmsra + 设 `HF_HUB_CACHE=/NAS/yesh/hf_cache/hub`、`HF_HUB_OFFLINE=1`、`PYTHONPATH=/NAS/yesh/VeriLong-RL`。
- 模型缓存 `/NAS/yesh/hf_cache/hub`；用 repo id（如 `Qwen/Qwen2.5-7B-Instruct`）离线解析，**不要**传 cache 顶层目录路径（顶层无 config.json，真实文件在 snapshots/<hash>/）。
- NAS 项目目录 `/NAS/yesh/VeriLong-RL`（group users 可写）。

**代码同步方式：** 本地 Windows 无 rsync；用 `git archive --format=tar HEAD` + `scp` 到集群 `/tmp` 再 `tar -xf` 到 NAS 目录（tasks.jsonl 含长文档约 146MB，整包约 156MB）。单文件改动可直接 scp 覆盖。

Related: [[pilot-eval-findings-gemini]]
