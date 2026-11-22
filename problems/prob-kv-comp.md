# MHA/GQA/MLA 的 KV 显存压缩率与算术强度推导

[MHA → MQA → GQA 注意力演进](../nodes/node-gqa.md)

掌握度：熟悉

## 题眼 / 触发条件

- 对比 MHA、MQA、GQA 与 DeepSeek MLA 的显存和吞吐差异
- 解释为什么 Decode 阶段受限于显存带宽 (Memory-Bound)

## 破题固定步骤 / 核心公式

- 1. 写单 Token KV 要素：Elements = 2 * N_kv * d_head。
- 2. 算术强度推导：Decode 单步 FLOPs ≈ 4 * B * N_q * S * d_h，访存 Bytes ≈ 4 * N_kv * S * d_h。
- 3. 得出 AI = B * (N_q / N_kv) = B * G，GQA 将算术强度直接放大 G 倍。
- 4. MLA 更进一步将 KV 联合低秩压缩为 c_KV，并用矩阵吸收消除还原开销。

## 典型题与参考实现

70B 模型 L=80, H_q=64, d_h=128 (BF16)，在 S=32768、Batch=1 下对比 MHA 与 GQA(H_kv=8) 的 KV Cache 占用，并从 Roofline 解释加速原因。

1. MHA KV Cache = 2 * 80 * 64 * 128 * 32768 * 2B = 85.9 GB（单卡 80GB H100 直接 OOM）。
2. GQA(H_kv=8) KV Cache = 85.9 / 8 = 10.74 GB。
3. Roofline 算术强度：Decode 时 AI = FLOPs / Bytes ≈ B * G。MHA 在 B=1 时 AI=1，远低于 H100 拐点(~295)；GQA 把 AI 放大 8 倍并支持更大 Batch，直接缓解访存瓶颈。

## 持续追加的变式与连环追问

### 变式 1：DeepSeek MLA 推理能矩阵吸收，为什么训练还要还原 K/V？
- 追问：推理时把 W_UK 吸收进 W_Q 后只缓存 c_KV，训练时为何不直接用吸收形式？
- 关键变化：推理时 Q 仅 1 token 吸收不增算力；训练序列长 S 大且解耦 RoPE 需与多头分片对齐 FlashAttention。
- 解析：训练期间先解压多头 K/V 才能直接套用高度优化的 FlashAttention-2 Tensor Core 算子；推理 Decode 因访存是唯一瓶颈，吸收后只读 d_c=512 维潜在向量收益极大。
