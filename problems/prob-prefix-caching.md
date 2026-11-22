# Prefix Caching (RadixAttention) 命中条件与缓存边界设计

[KV Cache & vLLM PagedAttention](../nodes/node-vllm.md)

掌握度：未测

## 题眼 / 触发条件

- 出现 RadixAttention、vLLM Automatic Prefix Caching (APC)、Block Hash 前缀匹配
- 排查为什么加了一个动态时间戳导致 TTFT 突然变慢 5 倍

## 破题固定步骤 / 核心公式

- 1. 按固定 Block 大小（如 16 tokens）计算含全部前驱内容的链式前缀哈希。
- 2. 一旦第 i 个 Token 变化，其所在 Block 及之后所有 KV 缓存全部失效。
- 3. Prompt 编排严格遵循：静态系统指令 -> 固定工具 Schema -> 公共知识库 -> 用户动态提问。

## 典型题与参考实现

对比 SGLang RadixAttention 与 vLLM APC 的数据结构差异，并给出最大化 Prefix Cache 命中率的 Prompt 排序规范。

vLLM APC 用链式哈希表管理定长 Block；SGLang RadixAttention 用基数树（Radix Tree）管理可变长树形共享前缀，更适合多分支 Agent 树搜索，两者均通过 LRU 淘汰冷叶子节点。

## 持续追加的变式与连环追问

（暂无）
