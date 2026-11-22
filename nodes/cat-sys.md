# 推理系统与显存优化

上级：[大模型核心知识体系 (LLM Core)](llm-root.md)

掌握度：熟悉 · 形态：知识点

> Prefill 计算受限与 Decode 访存受限下的 KV 管理与吞吐优化

$$AI = FLOPs / Bytes_HBM$$

## 1 核心框架

LLM 推理天生两阶段分裂：Prefill 并行吃满 Tensor Core（Compute-Bound），Decode 逐 Token 读取全部权重与 KV Cache（Memory-Bandwidth-Bound）。

## 2 Prefill-Decode 分离部署 (PD Disaggregation) · 架构设计

将 Prefill 节点与 Decode 节点物理拆开并按需配比，通过高速 RDMA 传递 KV Cache，彻底解决长输入 Prefill 抢占算力导致在线 Decode TTFT/TPOT 双抖动的问题。

## 下级笔记

- [KV Cache & vLLM PagedAttention](node-vllm.md) — 页表映射非连续 KV Block 消除显存碎片，撑起高并发批处理
