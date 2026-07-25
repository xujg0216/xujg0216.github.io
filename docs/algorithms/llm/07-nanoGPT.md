# nanogpt

> nanogpt来自Andrej Karpathy的开源项目，旨在让开发者能够以最直观的方式理解、训练和微调中等规模的 GPT（生成式预训练 Transformer）模型


## 1. 总览

nanoGPT是对 Karpathy minGPT 项目的重写，也就是[05-gpt_Karpathy.md](05-gpt_Karpathy.md), Karpathy 认为 minGPT 太过侧重于教育而牺牲了性能，因此 nanoGPT 在保持代码高度可读的同时，大幅优化了训练效率。
整个核心模型定义（model.py）和训练循环（train.py）通常只有几百行代码，非常适合人类阅读和修改。
**性能卓越**： 支持分布式训练（DDP）、混合精度训练（FP16/BF16）以及 PyTorch 2.0 的 torch.compile 编译加速，显著提升了训练速度。

**功能完备**：
- 从零预训练（Pre-training）： 可以从头开始训练类似 GPT-2 规模的模型。

- 微调（Fine-tuning）： 支持加载 OpenAI 发布的 GPT-2 权重，并根据自己的数据集进行微调。 

- 生成采样： 包含一个简单的脚本用于从训练好的模型中生成文本。



## 2. model.py

顺着数据的流向（从输入一个单词 ID 到输出预测概率），可以将 GPT 类中的核心前向传播过程拆解为以下三个关键阶段：

### 2.1 输入编码阶段：Token 与 Position

在 `GPT.forward` 方法中，第一步是将离散的单词索引变成连续的向量

```python

self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
tok_emb = self.transformer.wte(idx) # shape: (Batch, Time, n_embd)
pos_emb = self.transformer.wpe(pos) # shape: (Time, n_embd)
x = self.transformer.drop(tok_emb + pos_emb)
```
`wte (Word Token Embedding)`：gpt2 中使用的词表（$50304 \times 768$）。每个单词 ID 对应一行向量，在`gpt2`中共有`50304`个`token`, `embedding`维度为`768`

`wpe (Word Position Embedding)`：GPT-2 的位置特征。由于 Transformer 并行处理所有单词，它无法识别顺序。在这里为每个位置（0 到 1023）训练了一个专门的向量。此外，这里也决定了gpt2的上下文最大是1024

## 2.2 核心计算阶段：Block 的循环迭代

数据经过 `Embedding` 后，会进入 `n_layer`（通常是 `12` 层）个重复的 `Block`。

```python
self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
# GPT.forward 中的循环
for block in self.transformer.h:
    x = block(x)
```
每个 Block 内部逻辑如下：
```python
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```
其中， `self.ln_1`与`self.ln_2`为`LayerNorm()`, `self.attn` 为因果自注意力 (Causal Self-Attention)，`self.mlp`为前馈网络，下面分别介绍自注意力和前馈网络。

### 2.2.1 Causal Self-Attention

```python
# Causal：只能看当前位置之前的 token
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        # embedding 维度必须能被 head 数整除
        assert config.n_embd % config.n_head == 0

        # 三合一c_attn, 三个独立的 Linear 层分别给 Q、K、V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 输出投影（多头到单embedding）
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # 是否支持 Flash Attention，自动检测，PyTorch ≥ 2.0 才有 scaled_dot_product_attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 构造 causal mask（下三角）
            # torch.tril(torch.ones(config.block_size, config.block_size)) 下三角全1矩阵， register_buffer则表明不参与训练
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # x: (B, T, n_embd)
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # 一次性算q, k, v
        # q: (B, T, n_embd)
        # k: (B, T, n_embd)
        # v: (B, T, n_embd)
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        # k: (B, T, n_head, n_embd//n_head) -> (B, n_head, T, n_embd//n_head) 
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        # q: (B, T, n_head, n_embd//n_head) -> (B, n_head, T, n_embd//n_head) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        # v: (B, T, n_head, n_embd//n_head) -> (B, n_head, T, n_embd//n_head) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # 使用 Flash Attention， 效率更高
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # 手写 
            # att : (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # 未来位置用 -inf
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            # 未来位置经过softmax后变为0， 在最后一维（时间维度）归一化
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            # 矩阵乘法
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, n_embd)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # 线性映射回 embedding
        y = self.resid_dropout(self.c_proj(y))
        return y
```

此外，需要注意的是， `self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)`

- 输入：`(B, T, n_embd)`
- 输出：`(B, T, 3 * n_embd)`
- 一次性算 Q、K、V（效率高）
- 后面用 `.split()` 拆成三份
这是 GPT 系列的经典写法


==dropout==

`att = self.attn_dropout(att)`
Dropout 会 随机把一部分 attention 权重置 0， 防止依赖某 token

`y = self.resid_dropout(self.c_proj(y))`

放在residual之前， 削弱子模块（Attention）的支配力，防止模块过强
`model.eval()`推理阶段， 所有`dropout`自动关闭


### 2.2.2 MLP

```python
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
```
MLP对每个 token，先扩维、再激活、再压缩


==为什么是 4 * n_embd==

这是 Transformer 的经验设定（来自原论文）,维度太小表达能力不够,太大则算力& 显存爆炸，4C 是一个性价比最优点。GPT / BERT / ViT 都用这个比例



```python
self.gelu = nn.GELU()
```

GELU = **Gaussian Error Linear Unit**

公式（近似）：

$$
GELU(x) ≈ x · Φ(x)
$$

> **值越大，越“可信”，越容易通过**

比 ReLU 更“平滑”，对大模型训练更友好


## 3. bench.py

benchmark脚本，train.py 的精简版，只用来跑性能



1. 构建一个 **GPT 模型**
2. 准备 **真实或伪造的数据**
3. 跑一个 **标准训练 step**：

   ```text
   forward → loss → backward → optimizer.step
   ```
4. **不追求收敛**，只关心：

   * 每步多快
   * GPU 利用率（MFU）
   * Flash Attention / bf16 / torch.compile 的效果



```python
batch_size = 12 
block_size = 1024 # 上下文长度， Attention 的计算量（O(T²)）
device = 'cuda' # 强制用 GPU
# 优先用bf16， 否则退化到fp16
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# 使用 torch.compile（PyTorch 2.0）
compile = True 
# 允许用命令行 / 配置文件覆盖参数
exec(open('configurator.py').read())
# 随机数种子 & CUDA 设置
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# 允许矩阵乘法和卷积使用tf32,几乎不掉精度，但大幅提速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# GPU 上启用 AMP
# forward 用 bf16/fp16
# backward 自动处理缩放
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(...)
with ctx:
    logits, loss = model(X, Y)
real_data = True
dataset = 'openwebtext'
# memmap：不把整个数据加载进内存
train_data = np.memmap('train.bin', ...)

# 随机选 batch_size 个起点
ix = torch.randint(len(data) - block_size, (batch_size,))
# x: 上下文长度的输入， y是对应target
x = data[i : i+block_size]
y = data[i+1 : i+1+block_size]
# 异步拷贝到 GPU
x.pin_memory().to(device, non_blocking=True)

# 模型构建
gptconf = GPTConfig(
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
)
model = GPT(gptconf).to(device)
optimizer = model.configure_optimizers(...)

# 自动把 PyTorch 模型编译成更快的执行版本
if compile:
    model = torch.compile(model)

```


这是一个 **GPT-2 Small 级别模型**：

| 参数     | 值     |
| ------ | ----- |
| 层数     | 12    |
| hidden | 768   |
| heads  | 12    |
| 参数量    | ~124M |


## 4. sample.py

>文本生成，推理任务,加载一个已训练好的 GPT（自己训的 or GPT-2），给定一个 prompt，用自回归方式采样生成文本

```text
加载配置 & 权重
        ↓
准备 tokenizer（encode / decode）
        ↓
把 prompt 编码成 token
        ↓
循环采样生成 token
        ↓
decode 成文本并打印

```

```python
"""
Sample from a trained GPT model
This version only shows the structure and purpose of each block,
without actually running generation.
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# ------------------------- 配置参数 -----------------------------------------
# -----------------------------------------------------------------------------
init_from = 'resume'       # 模型初始化方式：'resume' = 自己训练的模型，'gpt2-xl' = GPT-2 预训练模型
out_dir = 'out'            # 模型 checkpoint 保存目录
start = "\n"               # 生成起始 prompt，可以是字符串或文件，例如 "FILE:prompt.txt"
num_samples = 10           # 生成样本数量
max_new_tokens = 500       # 每个样本生成的最大 token 数
temperature = 0.8          # 温度控制采样随机性：<1 更保守，>1 更随机
top_k = 200                # top-k 采样：只保留概率最高的 k 个 token
seed = 1337                # 随机种子，保证可复现
device = 'cuda'            # 设备: 'cpu', 'cuda', 'cuda:0' ...
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 推理精度
compile = False            # 是否用 PyTorch 2.0 torch.compile 加速
exec(open('configurator.py').read())  # 可以通过配置文件或命令行覆盖上面参数

# -----------------------------------------------------------------------------
# ------------------------- 随机种子 & 数值精度 --------------------------------
# -----------------------------------------------------------------------------
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# 允许 TF32 加速矩阵乘法
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 根据设备类型设置混合精度上下文
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# ------------------------- 模型加载 -----------------------------------------
# -----------------------------------------------------------------------------
if init_from == 'resume':
    # 从自己训练的 checkpoint 加载
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)

    # 使用 checkpoint 的模型参数重建模型结构
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    # 处理可能存在的前缀
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    # 加载权重
    model.load_state_dict(state_dict)

elif init_from.startswith('gpt2'):
    # 使用官方 GPT-2 预训练模型
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

# 设置推理模式
model.eval()
model.to(device)
if compile:
    model = torch.compile(model)  # PyTorch 2.0 可选加速

# -----------------------------------------------------------------------------
# ------------------------- Tokenizer / 编码系统 --------------------------------
# -----------------------------------------------------------------------------
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)

if load_meta:
    # 使用训练时保存的 tokenizer
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]  # 字符串 → token ids
    decode = lambda l: ''.join([itos[i] for i in l])  # token ids → 字符串
else:
    # 否则默认 GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# -----------------------------------------------------------------------------
# ------------------------- Prompt 编码 ---------------------------------------
# -----------------------------------------------------------------------------
if start.startswith('FILE:'):
    # 如果 prompt 是文件，读取文件内容
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
# 把字符串 prompt 转为 token id
start_ids = encode(start)
# 增加 batch 维度，得到 (1, T)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# -----------------------------------------------------------------------------
# ------------------------- 推理 / 生成 ----------------------------------------
# -----------------------------------------------------------------------------
with torch.no_grad():
    with ctx:
         for k in range(num_samples):
             y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
             print(decode(y[0].tolist()))
             print('---------------')

# 核心逻辑：
# 1. 自回归生成：每次生成一个 token，并追加到输入序列
# 2. temperature：控制采样随机性
# 3. top_k：只保留概率最高 k 个 token
# 4. decode：把 token id 转为文本输出

```





