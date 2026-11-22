# 大模型推理显存盘点与 KV Cache 容量/吞吐估算

[KV Cache & vLLM PagedAttention](../nodes/node-vllm.md)

掌握度：熟悉

## 题眼 / 触发条件

- 给定模型参数量、GPU 显存与带宽，估算最大并发 Batch 与 TPOT 延迟下界
- 解释 vLLM PagedAttention 如何将并发提升 2~4 倍

## 破题固定步骤 / 核心公式

- 1. 显存三分法：总显存 = 权重 + KV Cache + 激活与系统预留。
- 2. 单请求峰值 KV = 2 * L * N_kv * d_h * S_max * 2B。
- 3. PagedAttention 按 16-token Block 非连续分页分配，内部碎片趋近 0。
- 4. 延迟下界：Decode TPOT ≈ (Weight_Bytes / TP + KV_Read) / HBM_BW。

## 典型题与参考实现

8×H100 (80GB, 3.35TB/s) BF16 部署 Llama-3-70B (L=80, H_kv=8, d_h=128)，输入2048输出512，求极限并发与最小理论 TPOT。

总显存 640GB，权重 140GB，按 0.9 利用率剩余约 420GB 给 KV Cache；单请求 S=2560 占 0.80 GiB，PagedAttention 可支撑约 525 并发；每卡读取 17.5GB 权重理论 TPOT 下界 ≈ 5.22 ms/token。

## 持续追加的变式与连环追问

（暂无）
