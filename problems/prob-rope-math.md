# RoPE 复数旋转推导与 NTK/YaRN 长窗外推

[RoPE 旋转位置编码与外推](../nodes/node-rope.md)

掌握度：待攻克

## 题眼 / 触发条件

- 题目问“绝对位置旋转如何实现相对位置内积”或让写 rotate_half
- 上下文从 4K 外推到 128K 时 PI、NTK-aware、YaRN 各自的修正原理

## 破题固定步骤 / 核心公式

- 1. 将特征两两配对视为复数 z = x_0 + i x_1，在位置 m 乘 e^{i m theta}。
- 2. 证明内积 Re(q * e^{im theta} * conj(k * e^{in theta})) 只依赖 m - n。
- 3. 高频分量分辨局部相对词序，PI 统一压缩会毁掉高频精度；YaRN 按波长分段外推。

## 典型题与参考实现

写出 RoPE 基于 rotate_half 的 PyTorch 实现，并解释为什么线性位置插值 (PI) 效果不如 YaRN？

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

q_rot = q * cos + rotate_half(q) * sin
k_rot = k * cos + rotate_half(k) * sin

PI 将所有频率统一除以缩放比 s，挤扁了负责局部精准语法的高频旋转角；
YaRN 按波长 λ_j = 2π/θ_j 分三段：高频不插值保留局部精度、低频线性插值扩展长距、中频平滑过渡，并修正 Softmax 温度 sqrt(t)。

## 持续追加的变式与连环追问

### 变式 1：LLaMA 前半后半切分与相邻交替配对等价吗？
- 追问：chunk(2) 对半切分与原论文 (x_0, x_1) 相邻配对在权重转换时要注意什么？
- 关键变化：数学同构，但转换 HF 与 Megatron 权重时必须对 W_q/W_k 行重新置换，否则乱码。
- 解析：训练前特征维任意排列等价；推理加载现成权重必须严格匹配训练时的配对布局。
