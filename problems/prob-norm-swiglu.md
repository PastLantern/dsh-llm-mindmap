# Pre-RMSNorm 梯度流与 SwiGLU 8/3 d² 参数计算

[RMSNorm (Pre-Norm) & SwiGLU](../nodes/node-norm.md)

掌握度：已掌握

## 题眼 / 触发条件

- 对比 Pre-Norm 与 Post-Norm 在百层网络下的训练稳定性
- 计算 SwiGLU 隐藏层维度为何设为 (8/3) * d 并对齐 128/256 倍数

## 破题固定步骤 / 核心公式

- 1. Pre-Norm 保留恒等残差路径 I，梯度直达浅层不经过 Norm 雅可比衰减。
- 2. RMSNorm 省去减均值规约，延迟降低约 10%~15%。
- 3. 令 3 * d * d_ff = 2 * d * 4d = 8d² => d_ff = 8/3 d。

## 典型题与参考实现

推导 Pre-Norm 稳住梯度的原因，并算清 SwiGLU 为什么取 d_ff = 8/3 d。

Post-Norm 反向梯度需连乘每层 Norm 的缩放矩阵导致浅层梯度消失；Pre-Norm 雅可比含单位阵 I，始终保持恒等通路。SwiGLU 有 Gate/Up/Down 三个矩阵，为匹配传统两矩阵 FFN 的 8d² 参数量，令 3 * d * d_ff = 8d² 解得 d_ff = 8/3 d。

## 持续追加的变式与连环追问

（暂无）
