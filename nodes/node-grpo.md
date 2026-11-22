# PPO → DPO → GRPO 对齐演进

上级：[后训练与强化对齐](cat-post.md)

掌握度：待攻克 · 形态：知识点

> 砍掉 Critic 模型，以组内相对优势 A_i 驱动可验证推理对齐

$$A_i = (r_i - mean(r)) / std(r)$$

## 1 核心框架

强化对齐的演化主线一直在做减法：PPO 四模型太重 → DPO 砍掉在线采样与 RM（但受限于离线分布）→ GRPO 砍掉同尺寸 Critic，兼顾在线探索与低显存。

## 复习卡

- [PPO vs DPO vs DeepSeek GRPO 损失函数与架构取舍](../problems/prob-rl-grpo.md)
