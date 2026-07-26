# miniGPT

>来自karpathy Lets build GPT_ from scratch, in code, spelled out课程，训练字符级gpt,部分步骤与bigram相同


## 1. 超参数定义

```python
batch_size = 64  # 并行训练多少段文本
block_size = 256 # 最大上下文长度
n_embd = 384 # token 表示维度
n_head = 6 # 自注意力头
n_layer = 6 # transformer的层数
dropout = 0.2 # 正则化参数
```

> 字符级词表构建，构建训练、验证集、batch采用均与上节内容相同



## 2. 总体过程

```text
input idx
 ↓
Token Embedding + Position Embedding
 ↓
[ Transformer Block × 6 ]
 ↓
LayerNorm
 ↓
Linear → vocab_size
 ↓
CrossEntropyLoss
```

## 3. Token Embedding

```python
self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

```

`Token Embedding Table`是一个可学习的查表矩阵
- 每一行：一个 token（字符）
- 每一列：该 token 在某个语义维度上的分量

在此代码中一共有65个`token`， 也就是`vocab_size`的长度， `n_embd`是语义维为384,每一个 token 被映射成一个 384 维向量

**与Bigram embeddingde不同**

Bigram 中的 embedding

`nn.Embedding(vocab_size, vocab_size)`

- 输入 token id
- 直接输出 **下一个 token 的 logits**
    
`token i → [P(next=0), P(next=1), ..., P(next=64)]`

👉 **这是一个“查概率表”的模型**
GPT 中的 embedding

`nn.Embedding(vocab_size, n_embd)`

- token → **语义表示向量**
    
- 不直接预测
    
- 后面还要经过 **Attention + FFN**
    

👉 **这是“特征表示”，不是概率**


## 4. Position Embedding

```python
self.position_embedding_table = nn.Embedding(block_size, n_embd)
```
```text
(block_size, n_embd) = (256, 384)
```

- 每一行 = 序列中的一个 **位置**
    
- 第 0 个 token、第 1 个 token、第 2 个 token……


**为什么Position Embedding**


> **Attention 本身不感知顺序**

```text
"ABC" 和 "CBA" → Attention 看到的是同一组 token
```

```python
x = tok_emb + pos_emb
```

这一步让模型知道：

> “这是第几个 token”

## 5. GPTLanguageModel模型结构

`Head`：**单头自注意力（Self-Attention Head）**


```python
class Head(nn.Module):
    """ one head of self-attention """
```

表示 **一个 attention head**
GPT 中通常是 **多头注意力 = 多个 Head 并行**

---

初始化函数

```python
def __init__(self, head_size):
    super().__init__()
```

`head_size`：

* 每个 head 的维度
* 通常：`head_size = n_embd // n_head`

---

Key / Query / Value 线性层

```python
self.key = nn.Linear(n_embd, head_size, bias=False)
self.query = nn.Linear(n_embd, head_size, bias=False)
self.value = nn.Linear(n_embd, head_size, bias=False)
```

> 注意：
>
> * **GPT 是 self-attention** → Q、K、V 都来自同一个 `x`
> * `bias=False` 是标准 Transformer 设置


下三角 mask（因果掩码）

```python
self.register_buffer(
    'tril',
    torch.tril(torch.ones(block_size, block_size))
)
```

作用：**防止“偷看未来”**

举例（T=5）：

```
1 0 0 0 0
1 1 0 0 0
1 1 1 0 0
1 1 1 1 0
1 1 1 1 1
```

表示：

* 第 t 个 token **只能看到 ≤ t 的 token**
* GPT 是 **自回归语言模型（causal LM）**

📌 `register_buffer`：

* 不参与训练
* 会自动 `.to(device)`
* 会被保存进 `state_dict`



Dropout

```python
self.dropout = nn.Dropout(dropout)
```

防止 attention 过拟合

---

forward 过程（重点）

```python
def forward(self, x):
    B, T, C = x.shape
```

* B：batch size
* T：序列长度（time steps）
* C：embedding dim


计算 Q、K

```python
k = self.key(x)   # (B, T, hs)
q = self.query(x) # (B, T, hs)
```


Attention 权重计算（核心公式）

```python
wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
```

$$
\text{Attention}(Q, K, V)
= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

维度变化：

```
(B, T, hs) @ (B, hs, T) → (B, T, T)
```

每个 token 对 **所有 token** 的相似度


Causal Mask

```python
wei = wei.masked_fill(
    self.tril[:T, :T] == 0,
    float('-inf')
)
```

* 把未来位置设为 `-inf`
* softmax 后概率为 0

Softmax

```python
wei = F.softmax(wei, dim=-1)
```

得到：

```
每个 token 对过去 token 的注意力分布
```

Dropout

```python
wei = self.dropout(wei)
```
加权求和（Value）

```python
v = self.value(x)     # (B, T, hs)
out = wei @ v         # (B, T, hs)
```

每个 token = 所有 token 的 value 加权和

---

`MultiHeadAttention`：**多头注意力**


初始化

```python
self.heads = nn.ModuleList(
    [Head(head_size) for _ in range(num_heads)]
)
```

* 并行多个 Head
* 每个 head 关注不同模式

输出映射

```python
self.proj = nn.Linear(head_size * num_heads, n_embd)
```

把拼接后的结果映射回 `n_embd`


forward

```python
out = torch.cat([h(x) for h in self.heads], dim=-1)
```

维度：

```
num_heads × (B, T, hs) → (B, T, n_embd)
```


```python
out = self.dropout(self.proj(out))
```
==单头注意力和多头注意力公式推导==


单头注意力（Single-Head Attention）



给定输入 $(X \in \mathbb{R}^{L \times d_{\text{model}}})$ ：

$$
\begin{aligned}
Q &= XW_Q \\
K &= XW_K \\
V &= XW_V
\end{aligned}
$$

然后计算 **缩放点积注意力**：

$$
\text{Attention}(Q, K, V)
= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

输出维度仍然是：

$$
L \times d_v
$$


多头注意力（Multi-Head Attention）

计算方式

假设有 $h$ 个头（head）：

（1）线性投影（每个头一套参数）

$$
\begin{aligned}
Q_i &= XW_Q^{(i)} \
K_i &= XW_K^{(i)} \
V_i &= XW_V^{(i)}
\end{aligned}
\quad i = 1,2,\dots,h
$$

通常：
$$
d_k = d_v = \frac{d_{\text{model}}}{h}
$$

（2）每个头独立算注意力

$$
\text{head}_i
= \text{Attention}(Q_i, K_i, V_i)
$$


（3）拼接 + 线性变换

$$
\text{MultiHead}(Q,K,V)
= \text{Concat}(\text{head}_1,\dots,\text{head}_h)W_O
$$

输出维度仍是：
$$
L \times d_{\text{model}}
$$


---
FeedForward`：**逐 token 的 MLP**

> 注意：**不做 token 间通信**
> 只是对每个 token 独立做非线性变换


结构

```python
nn.Linear(n_embd, 4 * n_embd)
nn.ReLU()
nn.Linear(4 * n_embd, n_embd)
```

这是 GPT 经典结构：

* **升维 → 非线性 → 降维**
* 增强表达能力



forward

```python
return self.net(x)
```

Block`：**完整 Transformer Block**

> Transformer =
> **Attention（通信） + FFN（计算）**

---

初始化

```python
head_size = n_embd // n_head
self.sa = MultiHeadAttention(n_head, head_size)
self.ffwd = FeedFoward(n_embd)
```
LayerNorm（Pre-LN）

```python
self.ln1 = nn.LayerNorm(n_embd)
self.ln2 = nn.LayerNorm(n_embd)
```

⚠️ GPT 使用 **Pre-LN**（比 Post-LN 稳定）

forward（重点）

```python
x = x + self.sa(self.ln1(x))
```

1. LayerNorm
2. Self-Attention
3. **Residual（残差连接）**

```python
x = x + self.ffwd(self.ln2(x))
```

1. LayerNorm
2. FeedForward
3. Residual


整体数据流

```
Token Embedding + Position Embedding
        ↓
[ Block 1 ]
    LN → Multi-Head Attention → +res
    LN → FeedForward        → +res
        ↓
[ Block 2 ]
        ↓
...
        ↓
LayerNorm
Linear → vocab_size
```

---




### 5.1 Token Embedding（字符 → 向量）

```python
self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
```

**作用**

* 把 **离散的 token id** 映射为 **连续向量**
* 每个 token 对应一个 `n_embd` 维向量

**参数形状**

```text
(vocab_size, n_embd)
```


### 5.2 Position Embedding（位置 → 向量）

```python
self.position_embedding_table = nn.Embedding(block_size, n_embd)
```

**作用**

* 给 Transformer 注入 **序列顺序信息**
* 第 0 个 token、第 1 个 token……都有独立表示

**参数形状**

```text
(block_size, n_embd)
```


### 5.3 Transformer Blocks

```python
self.blocks = nn.Sequential(
    *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
)
```

* 堆叠 `n_layer` 个 Transformer Block
* 每个 Block 包含：

  * Multi-Head Self-Attention
  * FeedForward
  * Residual + LayerNorm


### 5.4 最后的 LayerNorm

```python
self.ln_f = nn.LayerNorm(n_embd)
```

**作用**

* 在输出 logits 前稳定特征分布
* GPT 的标准做法（Pre-LN 架构）


### 5.5 Language Model Head（向词表投影）

```python
self.lm_head = nn.Linear(n_embd, vocab_size)
```

**作用**

* 把隐藏向量映射成 **下一个 token 的 logits**
* 每个位置都是一个 `vocab_size` 分类问题

**输出形状**

```text
(B, T, vocab_size)
```

### 5.6️ 权重初始化

```python
self.apply(self._init_weights)
```

* 对模型中所有子模块递归调用 `_init_weights`
* 使用 GPT 常用初始化方式



**Linear 层初始化**

```python
if isinstance(module, nn.Linear):
    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

* 权重：高斯分布
* 标准差 `0.02`（GPT 论文经验值）

```python
if module.bias is not None:
    torch.nn.init.zeros_(module.bias)
```

* bias 全 0


**Embedding 初始化**

```python
elif isinstance(module, nn.Embedding):
    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

* token embedding 和 position embedding 都用同样规则


### 5.7 Transformer Blocks

```python
x = self.blocks(x)
```

* 多层 Attention + FFN
* 融合历史上下文信息

**形状保持不变**

```text
(B, T, n_embd)
```


### 5.8 `generate`：自回归文本生成

```python
def generate(self, idx, max_new_tokens):
```
**输入**

* `idx`: 当前上下文 `(B, T)`
* `max_new_tokens`: 要生成多少 token


**循环生成**

```python
for _ in range(max_new_tokens):
```


**截断上下文**

```python
idx_cond = idx[:, -block_size:]
```

* 只保留最近 `block_size` 个 token
* 防止位置 embedding 越界


**前向预测**

```python
logits, loss = self(idx_cond)
```
**取最后一个时间步**

```python
logits = logits[:, -1, :]
```

**形状**

```text
(B, vocab_size)
```

> 用当前上下文预测下一个 token


**Softmax → 概率**

```python
probs = F.softmax(logits, dim=-1)
```

---

**采样下一个 token**

```python
idx_next = torch.multinomial(probs, num_samples=1)
```

**输出**

```text
(B, 1)
```
**拼接到序列末尾**

```python
idx = torch.cat((idx, idx_next), dim=1)
```
