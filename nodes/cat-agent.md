# Agent 架构与 Harness 工程

上级：[大模型核心知识体系 (LLM Core)](llm-root.md)

掌握度：已掌握 · 形态：知识点

> 模型之外的编排外壳：通用循环 + 记忆设计 + Harness + RSI 演进

$$Model + Harness = Agent$$

> 附 4 条隔离讨论（未混入正文）

## 1 核心框架

【通用基础】Agent = 大模型大脑 + Harness 运行时底座（Perception 感知 → Planning 规划 → Memory 记忆 → Tool Action 执行 → Reflection 反思）。随着领域快速发展，可以在下方持续动态追加新的记忆架构、工程模式和前沿范式。

## 2 记忆设计 (Memory Architecture：工作 / 情景 / 语义 / 程序记忆) · 记忆系统

1. 工作记忆 (Working Memory)：当前上下文窗口内的活跃消息与临时状态，靠 Compaction 与结果裁剪控制体积；
2. 情景记忆 (Episodic Memory)：跨会话检索过去的成功轨迹与踩坑记录；
3. 语义记忆 (Semantic Memory)：沉淀成结构化知识图谱或双链笔记（正如本导图插件）；
4. 程序记忆 (Procedural Memory)：固化为 Skill、Hooks 与工具约束契约。

## 3 微内核 Harness 与多智能体架构设计 (Cordis + Subagent) · 架构设计

1. 一切皆插件 (Everything is a Plugin)：如 DeepSeek Harness 基于 Cordis 实现工具、沙箱、UI 与模型路由的时空可组合与安全卸载；
2. Subagent Spawn vs Fork：大量低信噪比的文件搜索/试错隔离到子 Agent 上下文（或 Fork 复用父 KV 前缀），主会话只回收精炼结论。

## 4 RSI 递归自我改进

子节点：[RSI 递归自我改进](node-rsi.md) — 让系统改进「产生产出的那套东西」，并让改进持续累积

## 5 Context Engineering（长程上下文工程与 Compaction 压缩） · 前沿演进

上下文不再是一次性静态模板，而是受限于注意力衰减（Lost in the Middle）与 KV Cache 成本的稀缺资源工程：
1. 工具结果按需裁剪与 Spill 落盘（大输出仅留路径句柄）；
2. 分层 Compaction：保留最近原貌 + 结构化总结早期探索轨迹；
3. 按需加载 Skills/MCP 描述，避免工具 Schema 爆炸挤占推理带宽。

## 下级笔记

- [长周期 Agent KV 命中与隔离](node-harness.md) — 只追加历史保前缀缓存，Subagent 隔离探索噪声防腐烂
