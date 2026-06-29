---
name: long-context-experiment-scope
description: Long Context 考核项目实验场景选择与阶段门控
metadata:
  type: project
  originSessionId: 2f73dacc-70cf-4641-87de-f3d158f5f809
  migratedTo: VeriLong-RL
  migratedAt: 2026-06-26
---

本次 Long Context 数据策略最终考核项目采用“混合场景”实验设计，但阶段推进：先完成第 1 类“合成多文档证据任务”作为主线闭环；完成后再根据实际花费时间与结果质量，由用户决定是否继续扩展到第 2 类“真实论文/技术报告长文档任务”和第 3 类“代码仓库级长上下文任务”。

**Why:** 用户希望本次考核体现固定时间内的独立产出，而不是绑定既有项目；同时资源充足，目标是诊断型实验与工程化展示结合，但需要先保证可控、可验证、可展示的主线闭环。

**How to apply:** 规划实验、写设计文档、拆 implementation plan 时，必须先围绕合成多文档 evidence/RLVR 任务完成数据生成、评测、训练/验证与网页展示；不要在主线闭环完成前分散精力到真实论文或代码仓库任务。后两者作为阶段 2/3 扩展，需再次询问用户是否继续。

Related: [[project-decisions]], [[cluster-resources]]
