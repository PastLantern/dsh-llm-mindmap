# PPO vs DPO vs DeepSeek GRPO 损失函数与架构取舍

[PPO → DPO → GRPO 对齐演进](../nodes/node-grpo.md)

掌握度：待攻克

## 题眼 / 触发条件

- 对比 PPO、DPO、GRPO 需要同时加载几个模型与各自显存瓶颈
- 写出 GRPO 组内优势估计公式与无偏 KL 估计器作用

## 破题固定步骤 / 核心公式

- 1. PPO 需 Actor/Ref/Reward/Critic 四模型，Critic 与策略同尺寸显存极重。
- 2. DPO 消去配分函数走离线二元偏好对比，但 Off-policy 易产生概率双降。
- 3. GRPO 同题采样 G 个输出，用组内相对归一化奖励替代 Critic 模型。

## 典型题与参考实现

对比 PPO、DPO 与 DeepSeek GRPO 的模型开销与 Loss 设计，说明 GRPO 为何契合数学/代码 RLVR？

GRPO 省去了与主策略同规模的 Critic 网络，对每个问题采样一组 {o_1..o_G} 计算相对优势 A_i = (r_i - mean(r)) / std(r)，再执行 Clipped Surrogate 与非负 K3 KL 正则。省下来的显存直接支持更长 CoT 采样，且代码/数学有确定性规则校验避免 Reward Hacking。

## 持续追加的变式与连环追问

### 变式 1：DPO 训练中 chosen 和 rejected 概率经常同时下降的原因
- 追问：为什么最大化 log π(y_w) - log π(y_l) 时 chosen 概率反而一起跌？
- 关键变化：Sigmoid 梯度只拉大差值，压低 y_l 比抬高 y_w 更容易把概率质量推向 OOD 词表区。
- 解析：工程常在 DPO Loss 里加一项 SFT NLL 正则 (RPO/CPO) 锚定 chosen 的绝对概率下限。

### 变式 2：GRPO 长思维链长度偏置 (Length Bias) 与 Dr.GRPO 修正
- 追问：原版 GRPO 目标函数为什么会导致负优势下的错误回答越写越长？Dr.GRPO 去掉哪两项归一化解决了这个问题？
- 关键变化：去掉响应内 1/|o_i|（改除以固定窗长常数）并去掉组标准差 std(r) 除法，消除负梯度稀释与低方差尖刺。
- 解析：当 A_i < 0 时，原版 1/|o_i| 让更长的废话分摊到更小的单 token 惩罚，模型学会“写长废话逃避扣分”；Dr.GRPO 只保留 r_i - mean(r) 并用固定长度常数归一化。
