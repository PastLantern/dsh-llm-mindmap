# KV Cache & vLLM PagedAttention

上级：[推理系统与显存优化](cat-sys.md)

掌握度：熟悉 · 形态：知识点

> 页表映射非连续 KV Block 消除显存碎片，撑起高并发批处理

$$KV = 2 * L * N_kv * d_h * S * B * 2B$$

## 1 核心框架

借鉴 OS 虚拟内存分页思想，把每个请求的 KV Cache 切分为固定 Block，按需映射到物理显存池，消除内部与外部碎片并支持 Copy-on-Write 共享前缀。

## 复习卡

- [大模型推理显存盘点与 KV Cache 容量/吞吐估算](../problems/prob-kv-roofline.md)
- [Prefix Caching (RadixAttention) 命中条件与缓存边界设计](../problems/prob-prefix-caching.md)
