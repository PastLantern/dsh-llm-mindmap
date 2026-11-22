# RoPE 旋转位置编码与外推

上级：[架构与注意力机制](cat-arch.md)

掌握度：待攻克 · 形态：知识点

> 复平面绝对旋转等价于相对内积，支持 NTK/YaRN 扩窗

$$\langle R_{\Theta, m}^d \mathbf{q},\; R_{\Theta, n}^d \mathbf{k} \rangle = \operatorname{Re}\!\left(\mathbf{q}_c \overline{\mathbf{k}_c}\, e^{i(m-n)\theta}\right)$$

> 附 5 条隔离讨论（未混入正文）

## 1 核心框架

### 复平面旋转的核心直觉
RoPE 把每个注意力头的 $d$ 维向量两两配对为 $d/2$ 个二维平面（等价于复数 $q_i = q_{2i} + \mathrm{i}q_{2i+1}$），对位置 $m$ 的向量按绝对角度 $m\theta_i$ 旋转；复共轭内积天然抵消公共绝对相位，使 Attention 点积**严格只依赖相对距离 $(m-n)$**。

### 为什么 RoPE 统治现代 LLM
- **绝对操作实现相对编码**：不改 Softmax 结构、无需显式构造 $O(S^2)$ 偏置矩阵。
- **KV Cache 完美兼容**：每个 token 写入 Cache 前仅旋转一次，后续 Decode 阶段历史 Key 无需重算。
- **正交性与远程衰减**：保持向量模长不变（$\|R_m\mathbf{q}\| = \|\mathbf{q}\|$），指数频率谱天然带来随距离增加的衰减上界。

## 2 2D 旋转矩阵与 rotate_half 零矩阵开销实现 · 原理推导

### 分块 $2\times 2$ 正交旋转矩阵
对第 $i$ 组二维特征，位置 $m$ 的正交旋转写作分块 $2 \times 2$ 矩阵：

$$R_{\theta_i, m}^{(2)} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

### $O(d)$ 逐元素实现与配对坑点
工程上利用稀疏结构化简为 **$O(d)$ 逐元素乘加**，无需构造大矩阵：

$$R_m \mathbf{x} = \mathbf{x} \odot \cos(m\Theta) + \operatorname{rotate\_half}(\mathbf{x}) \odot \sin(m\Theta)$$

```python
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

> ⚠️ **实现配对坑点**：原论文按相邻维 $(x_{2i}, x_{2i+1})$ 配对，HuggingFace / GPT-NeoX 则是前一半与后一半 $(x_i, x_{i+d/2})$ 配对；两者数学等价，但手写 Kernel 或转权重时必须严格对齐。

## 3 频率谱、波长 λ_i 与远程衰减 · 数学直觉

### 几何级数频率与波长分布
各二维子空间采用几何级数分布的旋转角速度与波长（默认底频 $b = 10000$）：

$$\theta_i = b^{-2i/d}, \qquad \lambda_i = \frac{2\pi}{\theta_i} = 2\pi b^{2i/d} \quad \left(i = 0, \dots, \frac{d}{2}-1\right)$$

- **高频维（$i$ 小、$\lambda_i$ 短）**：旋转快，负责捕捉相邻词序与局部句法搭配。
- **低频维（$i$ 大、$\lambda_i \gg L_{\text{train}}$）**：跨越数千 token 才转完不到一圈，编码段落级全局语义。

### 远程衰减（Abel 求和直觉）
各频率分量按幂律分布，对频率谱做分部求和可知，随距离 $|m-n|$ 增大产生相消干涉，天然抑制超远距离噪声。

## 4 长文本扩窗演进：PI → NTK-Aware → ABF + YaRN · 工程实践

### 主流扩窗方案横评
当目标长度 $L'$ 超出预训练窗长 $L$（倍率 $s = L'/L$），主流扩窗路线对比：

| 方案 | 核心改动 | 优势 | 短板 | 训练成本 |
| --- | --- | --- | --- | --- |
| 线性插值 (PI) | 统一缩位置 $m \to m/s$ | 相位落回训练域内 | 压扁高频分辨率，局部受损 | 需少步微调恢复 |
| NTK-Aware | 抬底频 $b' = b \cdot s^{d/(d-2)}$ | 高频少缩、低频拉长波长 | 低频外推不均，极长窗掉点 | 可免训 / 微调 |
| ABF 抬底频 | 续训把 $b$ 抬到 $5\times 10^5 \sim 10^6$ | 彻底拓宽预训练覆盖波长 | 依赖长文续训语料 | 中等续训算力 |
| YaRN 分段外推 | 按波长 $\lambda_i$ 斜坡混合 + 温度缩放 | 高频保真，32K~128K+ 最稳 | 需调斜坡区间 $\alpha, \beta$ | 约 400~1000 步微调 |

### YaRN 分段斜坡插值与现代配方
YaRN 按波数比 $r_i = \frac{L}{\lambda_i}$ 做斜坡插值并缩放 Softmax 注意力温度：

$$\theta_i' = (1 - \gamma(r_i))\frac{\theta_i}{s} + \gamma(r_i)\theta_i, \qquad \sqrt{\frac{1}{t}} \approx 0.1 \ln s + 1$$

- **主流配方**：先在续训阶段用 **ABF**（`rope_theta` 抬至 $5\times 10^5 \sim 10^6$）打底，再用 **YaRN** 冲击 $128\text{K} \sim 1\text{M}$。

## 5 DeepSeek MLA 解耦 RoPE 与多模态 M-RoPE · 前沿演进

### DeepSeek MLA 解耦 RoPE
标准 RoPE 会在点积 $q_m^\top k_n$ 中引入位置矩阵 $R_m^\top R_n = R_{n-m}$，阻断上投影矩阵 $W_{UK}$ 向 $W_Q$ 吸收。MLA 单独拆出极小的 $d_h^R = 64$ 维共享 `k_rope` 支路承载位置旋转，使主内容潜向量彻底免位置约束、实现推理矩阵吸收加速。

### 多模态 M-RoPE
Qwen2-VL 等多模态模型把旋转维度切分为时间、高度、宽度三段 $(T, H, W)$；纯文本退化为 1D RoPE，图文视频统一用三维坐标旋转对齐。

## 6 关于「KV Cache 完美兼容：…」的补充 · 通用知识

下面用真实数值演示锚点那句话说的情况：**“每个 token 写入 Cache 前仅旋转一次”** 意味着历史 Key 在后续所有 decode 步骤中都会成为“常量”。取 $d=4$、$b=10000$，两个平面 $\theta_0=1$、$\theta_1=0.01$。

#### Prefill：写入 Cache 前旋转一次

假设句子长度为 3（pos = 0,1,2），只关注 pos = 1 这个 token 的 key，其原始向量取 $k=(1,0,1,0)$：

$$R_1 k = (\cos 1,\ \sin 1,\ \cos 0.01,\ \sin 0.01) = (0.5403,\ 0.8415,\ 0.9999,\ 0.0100)$$

**写入 Cache 的就是这 4 个数**，原始的 $(1,0,1,0)$ 不会保留。

#### Decode：新 token 只旋转自己

到 pos = 3 时，query 也是 $q=(1,0,1,0)$，只需旋转一次：

$$R_3 q = (\cos 3,\ \sin 3,\ \cos 0.03,\ \sin 0.03) = (-0.9900,\ 0.1411,\ 0.9996,\ 0.0300)$$

然后直接与缓存中的点积，**cache 里那 4 个数不会有任何运算被施加到它们上**：

$$
\langle R_3 q,\ R_1 k\rangle
= \underbrace{(-0.9900)(0.5403) + (0.1411)(0.8415)}_{\cos(3-1)=\cos 2\ =\ -0.4161}
+ \underbrace{(0.9996)(0.9999) + (0.0300)(0.0100)}_{\cos(0.02)\ =\ 0.9998}
= 0.5837
$$

换成 pos = 2 的历史 key（距离 Δ=1），分数变成 $\cos 1 + \cos 0.01 = 0.5403 + 0.9999 = 1.5403$。注意低频平面（0.9999→0.9999）几乎没有改变，变化全部来自高频平面——这也印证了笔记里“高频维捕捉局部词序”的说法。

#### 与全量重算对照

全量重算即对整条序列重新计算。把 cache 换成“每步对历史拼接序列重算一遍”的原始 attention，`max|diff|` 为 0（仅浮点末位误差）。这不是近似——因为 $R_m^\top R_n = R_{n-m}$ 是恒等式，cache 里存的只是把 $R_n$ 提前乘掉而已。

#### 换算到真实规模

以 LLaMA-3.1-8B 为例：32 层、32 Q 头、8 KV 头、$d=128$、128K 上下文。

| 方案 | 每步 decode 的位置相关乘加 |
| --- | --- |
| 存旋转后 K（正确做法） | $32 \times (32+8) \times 128 \approx 1.6\times10^5$，**与上下文长度无关** |
| 存原始 K，每步重转历史 | $32 \times 8 \times 131072 \times 128 \approx 4.3\times10^9$ |

两者相差 **2.6 万倍**，而且是每生成一个 token 都要重新发生一次。相对于该步真正的 attention score 计算量（$1.7\times10^{10}$ 乘加），重转大约只多 25%；但这是**可以完全避免**的纯浪费，并且发生在访存受限的 decode 路径上。

#### 实现上的两条不变式

```python
# 写入：旋转后才进 cache
cache.append(rotate(k_new, pos))      # pos 是整数标量

# decode：只转新 token 的 q，历史一行不动
scores = rotate(q_new, pos) @ cache.T
```

- **Cache 存 $R_n k_n$，不存 $k_n$**；一旦存成 $k_n$，“只依赖相对距离”就退化成“每步都要现场恢复相位”。
- 位置号必须随 token 固定。因此 prefix cache 命中、vLLM 的 PagedAttention 块跨请求共享都能直接生效——复用前缀时位置相位本来就是对的，无需修正。这也是 RoPE 能和 KV 复用、投机解码同时存在的原因。

反例对照：T5 的加性相对位置偏置每步新增 $O(L)$ 个偏置值（按 $m-n$ 逐个查表），AliBi 的斜率至少还能预计算；纯相对偏置矩阵则完全不具备上述性质，这正是它输给 RoPE 的系统性原因。

> 这个“cache 不变量”是从锚点论点展开得到的一条可操作的判据，值得补进 `nb-rope-matrix` 旁边的工程块——你可以点「存到导图」提交。

## 复习卡

- [RoPE 复数旋转推导与 NTK/YaRN 长窗外推](../problems/prob-rope-math.md)
