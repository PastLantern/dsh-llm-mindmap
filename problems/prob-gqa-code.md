# PyTorch 手撕带 KV Broadcast 的 GQA

[MHA → MQA → GQA 注意力演进](../nodes/node-gqa.md)

掌握度：熟悉

## 题眼 / 触发条件

- 题目出现 num_attention_heads != num_key_value_heads 或 group_size / n_rep
- 白板要求写出 LLaMA-3 / Qwen2 风格 Grouped-Query Attention 前向传播
- 要求同时兼容 Causal Mask 与增量推理缓存 past_key_value 拼接

## 破题固定步骤 / 核心公式

- 1. 校验 num_q_heads % num_kv_heads == 0，计算 n_rep = num_q_heads // num_kv_heads。
- 2. Q 投影为 H_q 个头，K/V 投影为 H_kv 个头，均 transpose 成 (B, H, S, D)。
- 3. 若传入 past_kv，沿序列维 dim=2 拼接历史 K/V。
- 4. 用 expand + reshape 完成 Zero-Copy KV 广播对齐后计算缩放点积注意力。

## 典型题与参考实现

请用 PyTorch 手写支持 num_q_heads 与 num_kv_heads 分组对齐及 past_kv 缓存的 GroupedQueryAttention 模块，并说明如何避免广播时的多余显存拷贝。

def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, h_kv, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h_kv, n_rep, s, d).reshape(b, h_kv * n_rep, s, d)

class GroupedQueryAttention(nn.Module):
    def forward(self, x, mask=None, past_kv=None):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.h_q, self.d).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.h_kv, self.d).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.h_kv, self.d).transpose(1, 2)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        k_exp, v_exp = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        scores = (q @ k_exp.transpose(-2, -1)) / math.sqrt(self.d)
        if mask is not None: scores = scores + mask
        out = (torch.softmax(scores, dim=-1) @ v_exp).transpose(1, 2).contiguous().view(b, s, -1)
        return self.o_proj(out), (k, v)

## 持续追加的变式与连环追问

### 变式 1：PyTorch SDPA enable_gqa=True 与 FlashAttention 原生索引
- 追问：使用 F.scaled_dot_product_attention(..., enable_gqa=True) 时不调 repeat_kv，底层 Kernel 如何节省 HBM 带宽？
- 关键变化：Kernel 内部直接按 q_head_idx // n_rep 映射 KV 指针偏移，完全省去广播后的 HBM 物化拷贝。
- 解析：K/V 保持 (B, H_kv, S, D) 传入 SRAM 分块，访存量直接下降至原来的 1 / n_rep。

### 变式 2：Tensor Parallel 切分下 TP > num_kv_heads 怎么办？
- 追问：模型有 8 个 KV Head，但在 16 张卡上做 TP=16，每张卡分不到 1 个完整 KV Head 时如何并行？
- 关键变化：每 2 张卡复制同一份 KV Head，Q Head 仍按 16 路均匀切分，避免跨卡读取 KV。
- 解析：vLLM/Megatron 均采用 KV Head 复制策略，确保单卡内 QKV 计算本地闭合，仅 O_proj 后执行一次 AllReduce。
