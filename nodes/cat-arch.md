# 架构与注意力机制

上级：[大模型核心知识体系 (LLM Core)](llm-root.md)

掌握度：熟悉 · 形态：知识点

> KV 压缩、相对位置外推与归一化激活稳定性

$$Attn(Q,K,V) = softmax(QK^T / sqrt(d_k)) V$$

## 1 核心框架

现代 Decoder-Only Transformer（LLaMA / Qwen / DeepSeek 系）在 2017 原版架构上的三大核心演进方向：Pre-RMSNorm 稳梯度、SwiGLU 提表达、RoPE + GQA/MLA 降显存并支持超长窗。

## 2 Linear Attention 与 Hybrid 混合架构回归 (Mamba / Gated DeltaNet) · 架构设计

纯 Softmax Attention 的二次复杂度在 1M+ 长窗下成为瓶颈，越来越多前沿模型采用「每 3 层线性注意力/SSM + 1 层全注意力」的混合骨架，兼顾 O(1) 推理状态与精确检索 Recall。

## 下级笔记

- [MHA → MQA → GQA 注意力演进](node-gqa.md) — 共享 KV 头压缩显存带宽，平衡生成速度与表征能力
- [RoPE 旋转位置编码与外推](node-rope.md) — 复平面绝对旋转等价于相对内积，支持 NTK/YaRN 扩窗
- [RMSNorm (Pre-Norm) & SwiGLU](node-norm.md) — 去均值稳住残差梯度流，门控 FFN 以 8/3 d² 控参提效
