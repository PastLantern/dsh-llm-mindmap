# MHA → MQA → GQA 注意力演进

上级：[架构与注意力机制](cat-arch.md)

掌握度：熟悉 · 形态：知识点

> 共享 KV 头压缩显存带宽，平衡生成速度与表征能力

$$G = N_q / N_kv, repeat_kv(K, G)$$

> 附 1 条隔离讨论（未混入正文）

## 1 核心框架

MHA 每个 Q 头配独立 KV；MQA 所有 Q 头共享 1 组 KV 导致质量微降；GQA 将 Q 头分 G 组共享 KV 头，是现代开源模型标准折中。

## 2 DeepSeek MLA：超越 GQA 的低秩联合投影 · 前沿演进

MLA 不只是减少头数，而是把 K 和 V 联合投影进低维潜空间 c_KV，并通过矩阵吸收让推理时只需缓存 d_c + d_R 维向量，显存比 GQA 更小但表达力逼近 MHA。

## 复习卡

- [PyTorch 手撕带 KV Broadcast 的 GQA](../problems/prob-gqa-code.md)
- [MHA/GQA/MLA 的 KV 显存压缩率与算术强度推导](../problems/prob-kv-comp.md)
