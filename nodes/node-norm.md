# RMSNorm (Pre-Norm) & SwiGLU

上级：[架构与注意力机制](cat-arch.md)

掌握度：已掌握 · 形态：知识点

> 去均值稳住残差梯度流，门控 FFN 以 8/3 d² 控参提效

$$SwiGLU(x) = (SiLU(xW_1) ⊙ xW_3) W_2$$

## 1 核心框架

Pre-Norm 保住恒等残差梯度通道；RMSNorm 省去均值计算加速约 10%；SwiGLU 用双线性门控提升特征选择能力。

## 复习卡

- [Pre-RMSNorm 梯度流与 SwiGLU 8/3 d² 参数计算](../problems/prob-norm-swiglu.md)
