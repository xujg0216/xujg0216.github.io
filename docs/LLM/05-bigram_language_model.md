# bigram language model 

>这是一个字符级 Bigram 语言模型, 来自 Karpathy 的教学示例。只根据当前字符预测下一个字符，没有上下文建模、没有注意力。但它已经具备完整 LLM 训练 & 生成流程

## 1. 项目目标
- 使用 **字符级建模（character-level）**
    
- 基于 **Bigram 假设**：
    > 下一个字符只依赖当前字符
- 学习内容：
    - 数据准备
    - 词表构建
    - 模型定义
    - 训练与评估
    - 文本生成

## 2. 超参数定义

```python
batch_size = 32      # 每个 batch 的样本数
block_size = 8       # 上下文长度（T）
max_iters = 3000     # 训练迭代次数
eval_interval = 300  # 评估间隔
learning_rate = 1e-2
eval_iters = 200     # 评估时的 batch 数
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337) # 随机种子
```


## 3. 数据读取

```python
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
```

- `text` 是一个**长字符串**
    
- 数据集：Tiny Shakespeare


## 4. 字符级词表构建

### 4.1 获取所有字符

```python
chars = sorted(list(set(text)))
vocab_size = len(chars)
```

- `chars`：所有出现过的字符
    
- `vocab_size`：字符总数（≈ 65）

Tiny Shakespeare 文本里字符主要包括：
1. **小写字母**：`a-z` → 26 个
2. **大写字母**：`A-Z` → 26 个
3. **标点符号**：`, . ; : ' " ! ? -` 等 → 10 左右
4. **空格和换行**：空格 + `\n` → 2
5. **其他符号**：如 `(`, `)` 等 → 1-2
    

---

### 4.2 编码与解码映射

```python
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
```

```python
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
```

- **encode**：字符 → 整数
    
- **decode**：整数 → 字符



---

## 5. 构建训练/验证集

```python
data = torch.tensor(encode(text), dtype=torch.long)
```
- 整个文本 → 整数序列
```python
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```
- 90% 训练
- 10% 验证
    
## 6. batch 采样


```python
def get_batch(split):
```

> 随机从文本中采样一批  
> `(x, y)` 用于训练

---


```python
data = train_data if split == 'train' else val_data
ix = torch.randint(len(data) - block_size, (batch_size,))
```

- `ix`：随机起始位置
    

```python
x = torch.stack([data[i:i+block_size] for i in ix])
y = torch.stack([data[i+1:i+block_size+1] for i in ix])
```

- `x`：输入字符序列
    
- `y`：目标字符序列（右移一位）
    

```python
x, y = x.to(device), y.to(device)
return x, y
```

## 7. Bigram 语言模型定义


```python
class BigramLanguageModel(nn.Module):
```


```python
self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
```

- 输入：当前字符 id
    
- 输出：下一个字符的 logits
    
- 等价于一个：
    
    ```
    [vocab_size × vocab_size] 的查表矩阵
    ```

- 每个字符直接对应一个 **长度为 vocab_size 的向量**
    
- 这个向量就是 **预测下一个字符的 logits**
    
- 换句话说：
    

```text
self.token_embedding_table[i]  # i 是当前字符的 id
```

- 得到的向量长度 = 65
    
- 每个元素表示 **预测下一个字符为某个字符的分数**
    
- **这个向量直接就是 softmax 前的 logits**
    

所以 Bigram 核心公式就是：

```
P(next_char | current_char) = softmax(token_embedding_table[current_char])
```



---

### 7.1 forward（训练）

```python
logits = self.token_embedding_table(idx)
```

- 输入：`(B, T)`
    
- 输出：`(B, T, vocab_size)`
    

```python
logits = logits.view(B*T, C)
targets = targets.view(B*T)
loss = F.cross_entropy(logits, targets)
```

---

### 7.2 generate（文本生成）

```python
logits = logits[:, -1, :] # (B, C)
probs = F.softmax(logits, dim=-1) # (B, C)， 转化为概率分布
idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
```
  `torch.multinomial`：**从概率分布中随机采样**
    
- 输入：`probs`，每行是概率分布
- 输出：`idx_next` → 下一个字符的 **id**
- `num_samples=1` → 每行采一个样本
- 形状：`(B, 1)`
- 只看最后一个字符

```python
idx = torch.cat((idx, idx_next), dim=1)
```


💡 小技巧：

- 可以用 `temperature` 控制生成随机性：
    

```python
temperature = 0.8
probs = F.softmax(logits / temperature, dim=-1)
idx_next = torch.multinomial(probs, 1)
```

- `temperature < 1` → 选高概率字符更多 → 文本更“保守”
    
- `temperature > 1` → 低概率字符也会被选 → 文本更“多样化”
    


## 总体代码

```python 
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # 独立的句子，并行计算数量
block_size = 8 #    上下文长度
max_iters = 3000 #  训练迭代次数
eval_interval = 300 # 评估间隔
learning_rate = 1e-2 # 学习率
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 # 评估时的batch数
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text))) # 所有出现过的字符
vocab_size = len(chars) # 词表的长度 65

stoi = { ch:i for i,ch in enumerate(chars) } # 映射字典， key: char, value: idx
itos = { i:ch for i,ch in enumerate(chars) } # 映射字典， key: idx, value: char
encode = lambda s: [stoi[c] for c in s] # encoder: 给一个字符串，输出对应索引
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: 给定索引，解码字符串

# 划分训练和验证集 （ 9：1）
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]

# 数据加载
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 随机获取batch_size个起始字符索引
    x = torch.stack([data[i:i+block_size] for i in ix]) # x输入字符序列
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # y：目标字符序列（右移一位）
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx：当前上下文字符序列（整数索引），形状 (B, T)
        for _ in range(max_new_tokens):
            # get the predictions, 调用forward
            logits, loss = self(idx)
            # 取最后一个时间步，因为 Bigram 模型只依赖当前最后一个字符
            logits = logits[:, -1, :] # becomes (B, C)
            # 转成概率分布
            probs = F.softmax(logits, dim=-1) # (B, C)
            # 根据概率采样下一个字符
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # 拼接到现有序列
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx # 形状 (B, T + max_new_tokens)

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # 每 eval_interval = 300 步，评估训练集和验证集的平均 loss
    # estimate_loss() 是前面定义的无梯度函数
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # 获取一个batch的数据
    xb, yb = get_batch('train')

    # 前向 + loss
    logits, loss = model(xb, yb)
    # 清除上一步梯度
    optimizer.zero_grad(set_to_none=True)
    # 反向传播
    loss.backward()
    # 梯度更新
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


```

## 结果

```python
(yolo) xujg@xujg-ASUS:~/code/ng-video-lecture$ python bigram.py 
step 0: train loss 4.7305, val loss 4.7241
step 300: train loss 2.8110, val loss 2.8249
step 600: train loss 2.5434, val loss 2.5682
step 900: train loss 2.4932, val loss 2.5088
step 1200: train loss 2.4863, val loss 2.5035
step 1500: train loss 2.4665, val loss 2.4921
step 1800: train loss 2.4683, val loss 2.4936
step 2100: train loss 2.4696, val loss 2.4846
step 2400: train loss 2.4638, val loss 2.4879
step 2700: train loss 2.4738, val loss 2.4911



CEThik brid owindakis b, bth

HAPet bobe d e.
S:
O:3 my d?
LUCous:
Wanthar u qur, t.
War dXENDoate awice my.

Hastarom oroup
Yowhthetof isth ble mil ndill, ath iree sengmin lat Heriliovets, and Win nghir.
Swanousel lind me l.
HAshe ce hiry:
Supr aisspllw y.
Hentofu n Boopetelaves
MPOLI s, d mothakleo Windo whth eisbyo the m dourive we higend t so mower; te

AN ad nterupt f s ar igr t m:

Thin maleronth,
Mad
RD:

WISo myrangoube!
KENob&y, wardsal thes ghesthinin couk ay aney IOUSts I&fr y ce.
J
```