# 长周期 Agent KV 命中与隔离

上级：[Agent 架构与 Harness 工程](cat-agent.md)

掌握度：已掌握 · 形态：知识点

> 只追加历史保前缀缓存，Subagent 隔离探索噪声防腐烂

$$Stable_Prefix + Immutable_Tools => High Hit$$

## 1 核心框架

长链路 Agent 成本与延迟的核心命门在于 Prefix Cache 复用：只要头部 System Prompt 与工具 Schema 保持字节级静态、历史只追加不原地篡改，就能实现极高缓存命中率。

## 2 命中率怎么算、又会在哪里掉 · 工程实践

### 记分方式
命中率按 token 算，不按请求算：

$$\text{hit} = \frac{\sum_i \text{hit\_tokens}_i}{\sum_i \text{prompt\_tokens}_i}$$

工业实现都是 block 粒度：vLLM 的 automatic prefix caching 给每个 block（默认 16 token）算 hash，SGLang 的 RadixAttention 用基数树做 token 前缀匹配。所谓「字节级一致」落到实处是「block 边界对齐 + hash 相同」，差一个字符就整块 miss。

### 省的是算力，不是显存
命中的那段 token 不再跑 prefill，省下的是 $t_{\text{prefill}}$；KV 本身照样驻留显存。所以**压缩上下文是省显存，前缀缓存是省算力**，两件事。把历史砍短并不会提高命中率，把头部钉死才会。

### 最常见的四种掉法
- **动态字段进了前缀**：时间戳、日期、剩余预算、任务 ID 写进 system prompt 或开头几条消息，每轮前缀都变，命中率直接归零。
- **序列化不稳定**：工具 schema 的 key 顺序、注册顺序、JSON 缩进一变，前几个 block 的 hash 就变了。
- **就地改写历史**：compaction 重写中间消息、把旧 tool result 裁短或删掉，该位置之后的前缀全部失效。要压缩就新开会话 + 摘要，别原地改。
- **多副本部署**：prefix cache 在 GPU 本地，同一会话被路由到另一个副本等于冷启动；要共享得做 KV 层间传输，代价另算。

### 一句话判据
「这一轮里，有多少 token 是上一轮末尾原样出现过的？」——这才是命中率，与你压了多少轮历史无关。

## 3 隔离的两条路：限制工具，还是单开会话 · 架构设计

### 两条路：限制工具，还是单开会话
1. **同上下文限制工具**：在同一个 agent 上 `tools.restrict({ allow: [...] })`，只留只读工具。省资源，但子任务的 tool 输出仍然追加在同一条历史里——上下文照样被灌满，只是灌不脏文件。
2. **独立会话**：新开 session，只把必要材料注入（本插件就是把选中的那一节正文序列化成 anchor 再提问），探索过程全部留在子会话。

隔离要防的不是「模型乱写文件」，而是**无关工具输出被永久追加进主历史**：它们既污染注意力，又把后面每一轮都要重算的前缀撑大。长链路腐烂几乎都从这里开始。

### 代价：隔离本身会杀掉命中率
子会话的 system prompt、工具集、cwd 若与父会话不一致，就拿不到父会话的 KV，第一次 prefill 全付费。所以隔离粒度该按**探索代价**切，不按任务个数切：
- 只读一份材料、答案确定的问题 → 同上下文限制工具就够，别新开会话；
- 要搜索、读文件、反复试错的探索 → 单开会话，哪怕冷启动一次。

### 顺序不能反
先静态化前缀（system prompt 拆静态/动态两段、工具 schema 排序稳定、历史只追加），再谈隔离。反过来做，隔离只是把每一次冷启动变得更贵。前缀这一层的要求见 [KV Cache & vLLM PagedAttention](node-vllm.md)。

### 可落地的四条
- 静态区在前、动态区（时间、预算、任务参数）放在最后一条 user message；
- 工具注册顺序固定，schema 序列化 sort_keys；
- 压缩走「新会话 + 摘要」，历史永不原地改写；
- 监控命中率指标（vLLM 的 `gpu_prefix_cache_hit_rate`），把它当 SLO 而不是日志。

## 复习卡

- [长周期 Agent 的 KV Cache 命中率保护与 Subagent 隔离](../problems/prob-harness-cache.md)
