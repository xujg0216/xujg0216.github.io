# 从零手写 Qwen3 推理

> 仅依赖 PyTorch，从零实现 Qwen3 的完整推理流程。代码对应 `run_qwen3.py`，本节逐组件讲解。

Qwen3 是当前主流开源 LLM 之一，架构代表现代 LLM 的标配设计：

```
Token Embedding → RMSNorm → Attention (GQA + QK-Norm + RoPE) → RMSNorm → SwiGLU MLP
                                                    ↑_______残差_______↑    ↑___残差___↑

重复 num_layers 次后 → RMSNorm → lm_head → logits
```

下面按**从底层到顶层**的顺序，逐个拆解每个组件的原理和实现。

---

## 1. RMSNorm — 比 LayerNorm 更高效的归一化

### 原理

标准 LayerNorm 需要计算均值和方差：

$$
y = \frac{x - \mu}{\sigma} \cdot \gamma + \beta
$$

RMSNorm 只使用 RMS（Root Mean Square），省去均值计算和偏置参数：

$$
y = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}
$$

LLaMA 系列论文验证了去掉均值、去掉偏置不会损害效果，但计算速度更快。

### 实现

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (*, dim)
        x_float = x.float()                                                  # (*, dim)
        rms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps) # (*, 1)
        return (x_float * rms).to(x.dtype) * self.weight                    # (*, dim)
```

> 为什么要先转 float32？bfloat16 精度低，平方累加再开方容易累积误差，先转到 float32 算完再转回来。

---

## 2. Rotary Embedding — 旋转位置编码

详见 [04-ROPE.md](04-ROPE.md)。这里只回顾在 Qwen3 中的实际调用方式。

Qwen3 使用 base=1000000（比原始 Transformer 的 10000 大很多），目的是降低低频部分的旋转速度，让模型能感知到更远的距离。这对长上下文场景（Qwen3 支持 32K）至关重要。

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 1000000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids):
        # position_ids: (batch, seq_len)
        freqs = torch.outer(position_ids[0].float(), self.inv_freq)  # (seq_len, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)                       # (seq_len, head_dim)
        return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)         # 2 × (1, seq_len, head_dim)
```

RoPE 只作用于 Q 和 K，不作用于 V。Step 5 的注意力和 Step 5 对比图已经在 RoPE 博客中详细说明。

---

## 3. GQA（分组查询注意力）+ QK-Norm

### 三种注意力机制演变

```
MHA (Multi-Head)          MQA (Multi-Query)         GQA (Grouped Query)
Q: [H] heads              Q: [H] heads              Q: [H] heads (e.g. 32)
K: [H] heads              K: [1] head               K: [K] heads (e.g. 8)
V: [H] heads              V: [1] head               V: [K] heads (e.g. 8)

每个 Q 头有独立 KV         所有 Q 头共享 1 组 KV      每 G=H/K 个 Q 头共享 1 组 KV
参数最多，质量好             参数最少，质量可能下降      折中方案 ← Qwen3 采用
```

Qwen3-8B 配置：32 个 Q heads，8 个 KV heads → 每 4 个 Q 头共享一组 KV。

### QK-Norm（Qwen3 特有）

在 RoPE 之前，对 Q 和 K 每个头做一次 RMSNorm：

```python
q = self.q_norm(q)  # 对每个 head_dim 做 RMSNorm
k = self.k_norm(k)
q, k = apply_rotary_pos_emb(q, k, cos, sin)
```

QK-Norm 的目的：稳定训练、防止 Q 或 K 的范数过大导致的注意力崩溃。这是 Qwen3 区别于 LLaMA 的一个设计选择。

### 完整 Attention 实现

```python
class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.num_heads = config.num_attention_heads      # 32
        self.num_kv_heads = config.num_key_value_heads    # 8
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # 4
        self.head_dim = config.head_dim                   # 128
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask):
        # hidden_states: (B, S, hidden_size)
        # attention_mask: (1, 1, S, S)
        B, S, _ = hidden_states.shape

        # Step 1: 投影 Q, K, V → reshape 为多头格式
        # (B, S, hidden) -> (B, S, heads, head_dim) -> (B, heads, S, head_dim)
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        # q: (B, num_heads, S, head_dim)
        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # k: (B, num_kv_heads, S, head_dim)
        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # v: (B, num_kv_heads, S, head_dim)

        # Step 2: QK-Norm — 在 RoPE 之前归一化
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Step 3: 施加 RoPE（只作用于 Q 和 K）
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Step 4: GQA — 复制 KV 头以匹配 Q 头数量
        k = repeat_kv(k, self.num_kv_groups)                    # (B, num_heads, S, head_dim)
        v = repeat_kv(v, self.num_kv_groups)                    # (B, num_heads, S, head_dim)

        # Step 5: Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling  # (B, num_heads, S, S)
        attn_weights = attn_weights + attention_mask                         # (B, num_heads, S, S)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)                          # (B, num_heads, S, head_dim)

        # Step 6: 合并多头 → 投影输出
        attn_output = attn_output.transpose(1, 2).reshape(B, S, -1)          # (B, S, hidden)
        return self.o_proj(attn_output)                                      # (B, S, hidden)
```

### repeat_kv — 将 KV 头复制以匹配 Q 头

对于 32 个 Q 头和 8 个 KV 头，每组 4 个连续的 Q 头共享同一组 KV。用 `unsqueeze` + `expand` 实现：

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape                                       # h = 8
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d)       # 插入新维度再广播
             .reshape(b, h * n_rep, s, d)                       # → h = 32
```

`expand` 不会复制内存数据，只是改变 view，所以 GQA 几乎不增加 KV 显存。

---

## 4. SwiGLU MLP

### 原理

标准 Transformer MLP：

$$
y = W_2 \cdot \text{ReLU}(W_1 \cdot x)
$$

SwiGLU MLP（Qwen3 采用）：

$$
y = W_{down} \cdot (\text{SiLU}(W_{gate} \cdot x) \odot W_{up} \cdot x)
$$

其中 $\text{SiLU}(x) = x \cdot \sigma(x)$（也叫 Swish 激活），$\odot$ 是逐元素乘法。

三个投影层：
- **gate_proj**：门控，经 SiLU 激活后作为"开关"
- **up_proj**：信息主路，被 gate 的值调节
- **down_proj**：将中间维度压缩回 hidden_dim

### 实现

```python
class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        # x: (B, S, hidden_size)
        gate = F.silu(self.gate_proj(x))    # (B, S, intermediate_size)
        up   = self.up_proj(x)               # (B, S, intermediate_size)
        return self.down_proj(gate * up)     # (B, S, hidden_size)
```

用 SwiGLU 而非 ReLU 的好处：门控机制让模型选择性通过信息，表达能力更强。LLaMA、Qwen 等主流模型全部采用 SwiGLU。

---

## 5. Transformer Decoder Layer — Pre-Norm 架构

### Pre-Norm vs Post-Norm

```
Pre-Norm (Qwen3):                Post-Norm (原始 Transformer):
                                 
x → Norm → Attention → + → x'    x → Attention → + → Norm → x'
    |________________|               |__________|

x → Norm → MLP → + → x'            x → MLP → + → Norm → x'
    |___________|                      |_______|
```

Pre-Norm 训练更稳定，收敛更快，已成为现代 LLM 的标配。

### 实现

```python
class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask, position_embeddings):
        # hidden_states: (B, S, hidden_size)
        # 子层 1: 自注意力 + 残差
        residual = hidden_states                                           # (B, S, hidden_size)
        hidden_states = self.input_layernorm(hidden_states)                # (B, S, hidden_size)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
                                                                            # (B, S, hidden_size)
        hidden_states = residual + hidden_states                           # (B, S, hidden_size)

        # 子层 2: MLP + 残差
        residual = hidden_states                                           # (B, S, hidden_size)
        hidden_states = self.post_attention_layernorm(hidden_states)       # (B, S, hidden_size)
        hidden_states = self.mlp(hidden_states)                            # (B, S, hidden_size)
        hidden_states = residual + hidden_states                           # (B, S, hidden_size)

        return hidden_states
```

---

## 6. Qwen3Model — Transformer Backbone

将所有 Decoder Layer 堆叠起来，加上 Embedding 和最终 Normalization：

```python
class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

    def forward(self, input_ids):
        # input_ids: (B, S)
        B, S = input_ids.shape

        # Token Embedding
        hidden_states = self.embed_tokens(input_ids)                      # (B, S, hidden_size)

        # 位置 ID（无 KV cache 时从 0 开始）
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
                                                                          # (B, S)

        # 计算 RoPE cos/sin（所有层共享）
        position_embeddings = self.rotary_emb(position_ids)               # 2 × (1, S, head_dim)

        # Causal mask：上三角为 -inf，确保看不到未来 token
        causal_mask = torch.full((S, S), float("-inf"), device=input_ids.device, dtype=hidden_states.dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1)                 # (S, S)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)               # (1, 1, S, S)

        # 逐层前向
        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_mask, position_embeddings)
                                                                          # (B, S, hidden_size)

        return self.norm(hidden_states)                                   # (B, S, hidden_size)
```

---

## 7. Qwen3ForCausalLM — 完整语言模型

在 backbone 之上加一个 `lm_head`（线性投影），将隐藏状态映射到词表大小：

```python
class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids):
        hidden_states = self.model(input_ids)           # (B, S, hidden)
        return self.lm_head(hidden_states)              # (B, S, vocab_size)
```

## 8. 权重加载

从 HuggingFace 预训练模型加载权重的流程：

1. 用 `AutoModelForCausalLM.from_pretrained` 加载 HF 模型到 CPU
2. 提取 state_dict
3. 用 `load_state_dict(strict=False)` 加载到我们手写的模型中
4. 释放 HF 模型节省内存

```python
def load_weights_from_hf(model, model_name, device, dtype):
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map="cpu")
    result = model.load_state_dict(hf_model.state_dict(), strict=False)
    del hf_model
    torch.cuda.empty_cache()
    model.to(device=device, dtype=dtype)
```

`strict=False` 是因为 HF 模型中有一些不需要的 key（如 `inv_freq` 会被自动重建），同时忽略意外的 key。

---

## 9. 自回归生成

生成文本的核心循环：

```
Step 0: 输入 prompt "What is"      → forward(2 tokens) → 采样 token3 "the"
Step 1: 输入 prompt + "the"         → forward(3 tokens) → 采样 token4 "capital"
Step 2: 输入 prompt + "the capital" → forward(4 tokens) → 采样 token5 "of"
...
```

**没有 KV cache 时，每步都重新计算整个序列的注意力**。总计算量 $O(S^2)$，后续课程会讲 KV cache 将每步降为 $O(1)$。

```python
@torch.no_grad()
def generate(model, input_ids, max_new_tokens=128, temperature=0.7,
             top_k=50, top_p=0.9, eos_token_id=151645):
    generated = input_ids.clone()                                       # (B, S_in)

    for step in range(max_new_tokens):
        logits = model(generated)                                       # (B, S_cur, vocab_size)
        next_logits = logits[:, -1, :]                                  # (B, vocab_size)

        if temperature == 0:
            next_token = next_logits.argmax(dim=-1, keepdim=True)  # 贪心
        else:
            next_logits = next_logits / temperature
            # Top-K: 只保留概率最高的 K 个
            topk_vals = torch.topk(next_logits, min(top_k, next_logits.size(-1))).values
            next_logits = next_logits.masked_fill(
                next_logits < topk_vals[..., -1:], float("-inf"))
            # Top-P (Nucleus): 保留累积概率不超过 p 的 token
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumprobs > top_p
            mask[..., 1:] = mask[..., :-1].clone()  # 至少保留一个 token
            mask[..., 0] = False
            remove = mask.scatter(-1, sorted_idx, mask)
            next_logits = next_logits.masked_fill(remove, float("-inf"))
            # 按概率采样
            probs = F.softmax(next_logits, dim=-1)                           # (B, vocab_size)
            next_token = torch.multinomial(probs, num_samples=1)          # (B, 1)

        generated = torch.cat([generated, next_token], dim=-1)            # (B, S_cur+1)
        if next_token.item() == eos_token_id:
            break

    return generated
```

**温度 (temperature)**：控制生成多样性。T=0 时贪心选择概率最高的 token；T 越大输出越随机。

**Top-K**：只保留概率最高的 K 个 token，截断低概率的"噪音"。

**Top-P**：按概率从高到低取 token，直到累积概率超过阈值 P。这种"动态截断"比固定 K 更灵活。

---

## 10. Qwen3 模型规格

| 规格 | 层数 | hidden_dim | Q heads | KV heads | GQA 分组 |
|------|------|-----------|---------|----------|---------|
| 0.6B | 28 | 1024 | 16 | 8 | 每 2 个 Q 共享 1 组 KV |
| 1.7B | 28 | 2048 | 16 | 8 | 每 2 个 Q 共享 1 组 KV |
| 4B | 36 | 2560 | 32 | 8 | 每 4 个 Q 共享 1 组 KV |
| 8B | 36 | 4096 | 32 | 8 | 每 4 个 Q 共享 1 组 KV |
| 14B | 40 | 5120 | 40 | 8 | 每 5 个 Q 共享 1 组 KV |
| 32B | 64 | 5120 | 40 | 8 | 每 5 个 Q 共享 1 组 KV |

所有规格共用 base=1000000 的 RoPE，支持 32K 上下文（可扩展）。

---

## 总结

从零手写 Qwen3 需要实现的核心组件：

| 组件 | 作用 | 关键点 |
|------|------|--------|
| RMSNorm | 归一化（替代 LayerNorm） | 省去均值计算，更快 |
| RoPE | 位置编码 | 旋转 Q/K 向量，天然编码相对位置 |
| GQA | 注意力机制 | 多个 Q 头共享一组 KV，节省缓存 |
| QK-Norm | 稳定训练 | Qwen3 特有，RoPE 前归一化 Q/K |
| SwiGLU MLP | 前馈网络 | 门控 + 主路，比 ReLU 表达力更强 |
| Pre-Norm | 层架构 | 先归一化再计算，训练更稳定 |
| Causal Mask | 防止看到未来 | 上三角为 -inf |
| 温度 + Top-K + Top-P | 采样策略 | 控制生成质量和多样性 |
