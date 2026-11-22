# 长周期 Agent 的 KV Cache 命中率保护与 Subagent 隔离

[长周期 Agent KV 命中与隔离](../nodes/node-harness.md)

掌握度：已掌握

## 题眼 / 触发条件

- 多轮 Agent 工具调用中 Prefix Cache 命中率归零、延迟与成本暴涨
- 设计 Plan Mode、动态上下文与多 Agent 委派时的缓存边界

## 破题固定步骤 / 核心公式

- 1. 前缀字节不变原则：System Prompt 与 Tool Schemas 固定排序放最前。
- 2. Plan Mode 切模式不删工具定义，只靠指令约束，保住整个请求前缀哈希。
- 3. 动态时间或模式提醒走尾部 in-history 追加，绝不改写历史消息头。
- 4. Subagent 隔离大量检索噪声，仅把结构化结论回传父 Agent。

## 典型题与参考实现

设计 50 轮工具调用的 Agent Harness 时，如何避免破坏服务端 Prefix Cache？

保持头部 System Prompt + Tool Catalog 顺序绝对静态；动态上下文放进 user 增量快照；禁止就地修改旧 tool_result；长链探索交由 Subagent 消化后再回收摘要。

## 持续追加的变式与连环追问

（暂无）
