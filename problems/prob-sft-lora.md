# SFT 复读机死循环、灾难性遗忘与 LoRA 调优排障

[SFT 微调与 LoRA / QLoRA](../nodes/node-sft.md)

掌握度：熟悉

## 题眼 / 触发条件

- 微调后模型不输出 EOS 死循环复读，或通用数学代码能力暴跌
- 询问 LoRA 该加在哪些线性层、Rank r 与 alpha 怎么配

## 破题固定步骤 / 核心公式

- 1. 复读机先查 pad_token == eos_token 时是否误把所有 eos_token Mask 成 -100。
- 2. 灾难性遗忘：降 LR (1e-5)、混入 15% 通用回放数据、减少 Epoch。
- 3. LoRA 同时挂 Attention + MLP 全部线性层，alpha = 2 * r，B=0 初始化。

## 典型题与参考实现

LoRA 微调 7B 模型出现停不下来复读机与通用能力暴跌，如何系统排查并修复？

1. 检查 Chat Template 的 EOS token 与 Label Masking，保证每条 Assistant 回复末尾的真实 EOS 参与交叉熵 Loss（只 Mask 后续 Padding）；
2. 混入 10%~20% 通用高质量多任务 SFT 语料并调低 LR，同时在 q/k/v/o + gate/up/down 全线性层挂 LoRA。

## 持续追加的变式与连环追问

### 变式 1：为什么 LoRA 初始化 B=0, A~N(0, σ²) 而不能全 0 或全随机？
- 追问：A、B 同时初始化为 0 会怎样？同时随机初始化会怎样？
- 关键变化：全 0 会导致第一步梯度全 0 无法启动；全随机则 Step 0 就破坏预训练权重起点。
- 解析：B=0 保证 ΔW=BA=0（完美起点），而 ∂L/∂B 依赖 Ax ≠ 0 可正常收到首步非零梯度。
