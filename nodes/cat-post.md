# 后训练与强化对齐

上级：[大模型核心知识体系 (LLM Core)](llm-root.md)

掌握度：熟悉 · 形态：知识点

> 从指令微调 SFT/LoRA 到 PPO、DPO、GRPO 的对齐范式演进

$$Base → SFT → RL Alignment$$

## 1 核心框架

Pre-training 注入世界知识，Post-training 解锁指令格式、边界拒答以及推理搜索能力（Test-Time Compute）。

## 2 RLVR（基于可验证奖励的强化学习）成主流 · 前沿演进

从单纯依赖人类偏好打分（RLHF）转向数学求解器、单元测试、形式化证明等可自动验证的客观奖励，大幅降低 Reward Hacking 风险并涌现长思考行为。

## 下级笔记

- [SFT 微调与 LoRA / QLoRA](node-sft.md) — 低秩旁路降显存，警惕数据配比与 EOS 缺失引发复读
- [PPO → DPO → GRPO 对齐演进](node-grpo.md) — 砍掉 Critic 模型，以组内相对优势 A_i 驱动可验证推理对齐
