# 从单文件到模块化框架 — run_qwen3.py 重构

## 概述

在 [05-run-qwen3.md](05-run-qwen3.md) 中实现了一个完整的 Qwen3 推理引擎。它能跑，但存在三个核心问题：

1. **依赖 `nn.Module`** — PyTorch 的 `nn.Module` 为训练设计（梯度追踪、参数注册、hook 机制），推理用不到这些
2. **依赖 `transformers`** — 用 `AutoConfig` 读配置、用 `AutoModelForCausalLM` 中转权重，多加载了一份完整模型到内存
3. **单文件不可扩展** — 所有代码堆在一个文件里，无法独立替换某一层的实现

本文档的目标是：**保持推理逻辑完全不变**，仅改造代码架构，将 `run_qwen3` 重构为一个模块化的推理框架。

## 改造前后对比

```
改造前：run_qwen3.py (单文件)           改造后：模块化包
─────────────────────────────          ─────────────────────────────────
nn.Module 基类                         → BaseOP 自定义基类（无梯度开销）
nn.Parameter 包装权重                   → torch.empty() 原始张量
nn.Linear (独立 Q/K/V/O)               → LinearQKVMerged (QKV 融合投影)
nn.Linear (独立 gate/up)               → LinearColParallelMerged (gate+up 融合)
RMSNorm (纯 PyTorch)                   → RMSNorm + RMSNormFused (flashinfer 加速)
F.silu(gate) * up                      → silu_and_mul (flashinfer fused kernel)
register_buffer (RoPE)                 → StateLessOP + 预计算 cache
AutoConfig.from_pretrained()           → AutoConfig + ModelConfig.from_hf()
AutoModelForCausalLM → state_dict 中转 → safetensors 直接加载 + fused 层自动打包
generate() 独立函数                     → LLM 类 + Engine + Scheduler
forward 传参                            → 全局 Context 传递 batch 信息
```

## 目录结构

```
推理框架/
├── __init__.py                导出 LLM, Sampler, SamplingParams
├── __main__.py                CLI 入口
├── core.py                    Context, Batch, Req, SamplingParams
├── engine/                    执行引擎
│   ├── __init__.py
│   ├── engine.py              Engine 类 (模型创建 + KV cache + 前向)
│   ├── sample.py              Sampler 类
│   └── graph.py               GraphRunner (CUDA Graph 加速)
├── llm/                       用户接口
│   ├── __init__.py
│   └── llm.py                 LLM 入口类
├── models/                    模型实现
│   ├── __init__.py            create_model() 工厂函数
│   ├── base.py                BaseLLMModel(ABC, BaseOP)
│   ├── config.py              ModelConfig dataclass + from_hf() / from_json()
│   ├── qwen3.py               完整 Qwen3 模型 (5 个类)
│   └── weight.py              safetensors 直接加载 + fused 层自动打包
├── layers/                    基础算子层
│   ├── __init__.py
│   ├── base.py                BaseOP / StateLessOP / OPList
│   ├── linear.py              Linear / LinearQKVMerged / LinearColParallelMerged
│   ├── norm.py                RMSNorm / RMSNormFused (flashinfer 加速)
│   ├── rotary.py              RotaryEmbedding(StateLessOP) (预计算 cos/sin)
│   ├── attention.py           rotate_half / apply_rotary_pos_emb / repeat_kv
│   ├── activation.py          silu_and_mul (flashinfer fused kernel)
│   └── embedding.py           Embedding + LMHead (含 tie_word_embeddings)
├── attention/                 注意力后端 (FlashInfer)
├── kvcache/                   KV Cache 管理
└── scheduler/                 调度器 (continuous batching)
    ├── scheduler.py
    ├── table.py
    ├── cache.py
    ├── prefill.py
    ├── decode.py
    └── common.py
```

---

## 第 1 步：创建 BaseOP 基类替换 nn.Module — `layers/base.py`

### 为什么推理不需要 nn.Module

`nn.Module` 在幕后做了大量工作：
- 维护 `_parameters` 字典追踪所有 `nn.Parameter`
- 维护 `_modules` 字典追踪所有子模块
- 维护 `_buffers` 字典追踪所有 buffer
- 支持 `register_forward_hook`、`register_backward_hook`
- 支持 `train()`/`eval()` 模式切换
- `nn.Parameter` 自动设置 `requires_grad=True`

**推理时这些全部不需要。** 我们只需要两件事：
1. 存储权重（`torch.Tensor`）
2. 执行前向计算（`forward()`）

### BaseOP 的极简设计

```python
class BaseOP:
    @abstractmethod
    def forward(self, *args, **kwargs): ...   # 唯一的抽象接口

    def state_dict(self, *, prefix=""):       # 递归收集所有 Tensor 属性
        for name, param in self.__dict__.items():
            if name.startswith("_"):          # 跳过私有属性（非权重）
                continue
            if isinstance(param, torch.Tensor):
                result[key] = param           # 直接收集
            elif isinstance(param, BaseOP):
                param.state_dict(...)         # 递归进入子模块

    def load_state_dict(self, state_dict):    # 递归加载，pop 消费每个 key
        ...                                   # 最终校验 state_dict 为空
```

**核心规则：**
- 公有属性（不以 `_` 开头）如果是 `torch.Tensor` → 权重，参与 `state_dict`
- 公有属性如果是 `BaseOP` → 子模块，递归处理
- 私有属性（以 `_` 开头） → 跳过（用于存放 scale、cache 等非权重数据）

### key 的拼接规则

```
Qwen3ForCausalLM                     prefix = ""
  ├── model (Qwen3Model)             prefix = "model"
  │     ├── embed_tokens (Embedding) prefix = "model.embed_tokens"
  │     │     └── weight             key = "model.embed_tokens.weight"  ✓ 对应 safetensors
  │     ├── layers (OPList)          prefix = "model.layers"
  │     │     ├── [0] (DecoderLayer) prefix = "model.layers.0"
  │     │     │     ├── self_attn    prefix = "model.layers.0.self_attn"
  │     │     │     │     ├── qkv_proj → key = "model.layers.0.self_attn.qkv_proj.weight"
```

这些 key **恰好**与 safetensors 文件中的 key 一致，所以可以直接加载。

### 三个基类

```python
class BaseOP:
    """替代 nn.Module 的推理基类"""

class StateLessOP(BaseOP):
    """无参数的算子（如 RotaryEmbedding），state_dict 永远为空"""

class OPList(BaseOP, Generic[T]):
    """替代 nn.ModuleList，用整数索引做 prefix"""
```

---

## 第 2 步：替换 nn.Linear，引入融合投影 — `layers/linear.py`

与 run_qwen3.py 的关键区别：**Q、K、V 三个投影合并为一个 QKV 融合投影**，**gate 和 up 两个投影合并为一个 gate_up 融合投影**。

```python
class Linear(BaseOP):
    """单路线性投影，替代 nn.Linear"""
    def __init__(self, input_size, output_size, has_bias=False):
        self.weight = torch.empty(output_size, input_size)
        self.bias = torch.empty(output_size) if has_bias else None

class LinearQKVMerged(Linear):
    """Q、K、V 三路融合投影，布局为 [Q | K | V]"""
    def __init__(self, hidden_size, q_size, kv_size, has_bias=False):
        super().__init__(hidden_size, q_size + 2 * kv_size, has_bias)

class LinearColParallelMerged(Linear):
    """gate、up 两路融合投影，布局为 [gate | up]"""
    def __init__(self, input_size, output_sizes, has_bias=False):
        super().__init__(input_size, sum(output_sizes), has_bias)
```

> **为什么融合？** 一次矩阵乘法同时算出 QKV（或 gate+up），比三次独立乘法更高效，且为后续 tensor-parallel 切分做准备。

---

## 第 3 步：RMSNorm 用 flashinfer 加速 — `layers/norm.py`

run_qwen3.py 用纯 PyTorch 实现，重构版使用 flashinfer 的高性能 CUDA kernel：

```python
class RMSNorm(BaseOP):
    def __init__(self, size, eps=1e-6):
        from flashinfer import rmsnorm
        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    def forward(self, x):
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x):
        """就地归一化，节省一次内存分配"""
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    """融合残差 + 归一化：一步完成 residual_add + rmsnorm"""
    def __init__(self, size, eps=1e-6):
        from flashinfer import fused_add_rmsnorm, rmsnorm
        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(self, x, residual=None):
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
```

`RMSNormFused` 是 Decoder Layer 使用的版本：残差加法和 RMS 归一化合并在一个 kernel 中完成，避免中间 tensor 的读写。

---

## 第 4 步：RotaryEmbedding 改为预计算 — `layers/rotary.py`

与 run_qwen3.py 的关键区别：**预计算所有位置的 cos/sin，forward 时直接查表**，而非每次现场计算。

```python
class RotaryEmbedding(StateLessOP):
    def __init__(self, head_dim, max_position_embeddings, base=1000000.0):
        super().__init__()
        # 强制在 CPU 上计算 (meta device 不能做数学运算)
        with torch.device("cpu"):
            inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
            t = torch.arange(max_position_embeddings, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cache = emb.cos()  # (max_pos, head_dim)
            self._sin_cache = emb.sin()

    def forward(self, position_ids):
        cos = self._cos_cache[position_ids]  # (B, S, head_dim)
        sin = self._sin_cache[position_ids]
        return cos.unsqueeze(1), sin.unsqueeze(1)  # (B, 1, S, head_dim)
```

- 继承 `StateLessOP`（无学习参数，state_dict 为空）
- `_cos_cache` / `_sin_cache` 以 `_` 前缀命名 → 跳过 state_dict
- `with torch.device("cpu")` 强制在 CPU 计算，避免 meta device 无法做数学运算的问题
- `set_device()` 在加载完权重后将 cache 移到 GPU

---

## 第 5 步：激活函数用 flashinfer fused kernel — `layers/activation.py`

```python
def silu_and_mul(x, out=None):
    """flashinfer 的 fused silu+multiply kernel"""
    from flashinfer import silu_and_mul as flashinfer_silu_and_mul
    return flashinfer_silu_and_mul(x, out=out)
```

比 `F.silu(gate) * up` 更高效：一次 kernel 完成激活和逐元素乘法。

---

## 第 6 步：Embedding 和 LMHead — `layers/embedding.py`

```python
class Embedding(BaseOP):
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.empty(num_embeddings, embedding_dim)

    def forward(self, input_ids):
        return F.embedding(input_ids, self.weight)


class LMHead(BaseOP):
    def __init__(self, num_embeddings, embedding_dim,
                 tie_word_embeddings=False, tied_embedding=None):
        self._tie_word_embeddings = tie_word_embeddings
        self._tied_embedding = tied_embedding
        if not tie_word_embeddings:
            self.weight = torch.empty(num_embeddings, embedding_dim)

    def forward(self, x):
        ctx = get_global_ctx()
        batch = ctx.batch
        if batch.is_prefill:
            # prefill 阶段只取每个序列最后一个 token 的 hidden state
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        w = self._tied_embedding.weight if self._tie_word_embeddings else self.weight
        return F.linear(x, w)
```

`LMHead` 的关键设计：
- **tie_word_embeddings**：小模型（如 0.6B）的 `lm_head.weight` 与 `embed_tokens.weight` 共享，tie 模式下直接 pop 掉 safetensors 中多余的 key
- **prefill 优化**：prefill 阶段只取每个序列最后一个 token 的 hidden state 做 logits 计算，不处理整个序列

---

## 第 7 步：全局 Context — `core.py`

run_qwen3.py 的 forward 通过参数传递 mask、position_ids 等。重构版改为**全局 Context**模式：

```python
@dataclass
class Context:
    page_size: int
    page_table: torch.Tensor
    attn_backend: BaseAttnBackend
    kv_cache: MHAKVCache
    _batch: Batch | None

    @contextmanager
    def forward_batch(self, batch):
        """设置当前 batch，forward 完成后自动清除"""
        self._batch = batch
        try:
            yield
        finally:
            self._batch = None


def get_global_ctx() -> Context:
    """任意层的 forward 方法中可直接调用，获取当前 batch 信息"""
```

这样模型各层的 `forward()` 不需要传 `attention_mask`、`position_ids` 等参数，直接从 Context 中读取。减少传参层数，各层自治。

---

## 第 8 步：模型配置 — `models/config.py`

```python
@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    hidden_act: str
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool

    @classmethod
    def from_hf(cls, config) -> "ModelConfig":
        """从 transformers config 对象加载 (主路径)"""

    @classmethod
    def from_json(cls, model_path: str) -> "ModelConfig":
        """从 config.json 文件加载 (fallback)"""
```

`frozen=True` 表示配置创建后不可修改，避免运行时意外改动。

---

## 第 9 步：模型实现 — `models/qwen3.py`

核心改动点（相对于 run_qwen3.py）：

### Qwen3Attention — QKV 融合 + flashinfer 注意力

```python
class Qwen3Attention(BaseOP):
    def __init__(self, config, layer_idx):
        self.num_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self._scale = config.head_dim ** -0.5
        self._layer_idx = layer_idx

        # Q、K、V 融合为一个投影层
        self.qkv_proj = LinearQKVMerged(config.hidden_size,
                                         self.num_heads * self.head_dim,
                                         self.num_kv_heads * self.head_dim)
        self.o_proj = Linear(self.num_heads * self.head_dim, config.hidden_size)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings):
        total_tokens, _ = hidden_states.shape
        qkv = self.qkv_proj.forward(hidden_states)
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(total_tokens, self.num_heads, self.head_dim)
        k = k.view(total_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(total_tokens, self.num_kv_heads, self.head_dim)

        # QK-Norm: 就地归一化 (inplace，省内存)
        self.q_norm.forward_inplace(q)
        self.k_norm.forward_inplace(k)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 注意力计算委托给 flashinfer 后端
        ctx = get_global_ctx()
        attn_output = ctx.attn_backend.forward(q, k, v, self._layer_idx, ctx.batch)
        return self.o_proj.forward(attn_output.reshape(q.size(0), -1))
```

关键变化：
- QKV 一次乘法完成 → `qkv_proj.forward()` → `split()` 拆分
- QK-Norm 用 `forward_inplace` 就地归一化，节省内存
- 注意力计算通过 `ctx.attn_backend.forward()` 委托给 FlashInfer，支持 PageAttention

### Qwen3MLP — gate+up 融合

```python
class Qwen3MLP(BaseOP):
    def __init__(self, config):
        # gate 和 up 融合为一个投影层
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size, [config.intermediate_size, config.intermediate_size]
        )
        self.down_proj = Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        gate_up = self.gate_up_proj.forward(x)      # 一次乘法得到 gate+up
        return self.down_proj.forward(silu_and_mul(gate_up))  # fused kernel
```

### Decoder Layer — 融合残差 + 归一化

```python
class Qwen3DecoderLayer(BaseOP):
    def __init__(self, config, layer_idx):
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, residual):
        # RMSNormFused 一步完成 residual_add + rmsnorm
        hidden_states, residual = self.input_layernorm.forward(hidden_states, residual)
        hidden_states = self.self_attn.forward(hidden_states, position_embeddings)
        hidden_states, residual = self.post_attention_layernorm.forward(hidden_states, residual)
        hidden_states = self.mlp.forward(hidden_states)
        return hidden_states, residual
```

`RMSNormFused` 将残差加法和 RMS 归一化合并在一个 CUDA kernel 中完成，避免中间结果的内存分配和读写。

### Qwen3Model — 全局 Context 驱动

```python
class Qwen3Model(BaseOP):
    def forward(self):
        ctx = get_global_ctx()
        input_ids = ctx.batch.input_ids
        batch = ctx.batch

        hidden_states = self.embed_tokens.forward(input_ids)
        position_embeddings = self._rotary_emb.forward(batch.positions)

        residual = None
        for layer in self.layers.op_list:
            hidden_states, residual = layer.forward(
                hidden_states, position_embeddings, residual
            )
        return self.norm.forward(hidden_states, residual)[0]
```

forward 不再接收参数，直接从 `get_global_ctx()` 获取输入。

---

## 第 10 步：safetensors 直接加载 + fused 层自动打包 — `models/weight.py`

这是消除 `transformers` 权重依赖的关键步骤。与 run_qwen3.py 中加载完整 HF 模型再提取 state_dict 不同，这里直接读 safetensors 文件，并自动处理 fused 层的打包：

```python
# fused 层映射: 模型中的 fused 层 → HF checkpoint 中的独立层
packed_modules_mapping = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}

def load_weights(model, model_path, device, dtype):
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

    # 建立 key → file 索引
    index = _checkpoint_index(files)

    # 遍历模型的 state_dict key，从 safetensors 加载对应张量
    fused_state_dict = {}
    for target_name in model.state_dict():
        source_names = _packed_source_names(target_name)
        if source_names is None:
            # 普通层: 直接加载
            tensor = _read_tensor(index, target_name)
        else:
            # fused 层: 从多个 source 读取，在 dim=0 拼接
            tensor = torch.cat([_read_tensor(index, name) for name in source_names], dim=0)
        fused_state_dict[target_name] = tensor.to(device=device, dtype=dtype)

    model.load_state_dict(fused_state_dict)
```

**工作原理**：当模型的 key 包含 `.qkv_proj.` 时，`_packed_source_names` 自动将其替换为 `("q_proj", "k_proj", "v_proj")` 三个 HF 名称，分别读取后沿 dim=0 拼接。

### 内存对比（Qwen3-0.6B, bfloat16）

| 方式 | 峰值内存 | 说明 |
|------|---------|------|
| run_qwen3.py | ~2.4 GB | HF 模型 1.2G + 我们的模型 1.2G 同时在内存 |
| safetensors 直读 | ~1.2 GB | 只加载到我们的模型，省去 HF 中转 |

---

## 第 11 步：Engine + LLM + Scheduler

重构版引入了完整的推理引擎架构（run_qwen3.py 的 `generate()` 函数只是一个简单循环）：

```
LLM.generate()
    └── Scheduler (continuous batching 调度)
        └── Engine.forward_batch()
            ├── set_global_ctx (设置当前 batch)
            ├── model.forward()
            │   └── 各层通过 get_global_ctx() 获取 batch 信息
            ├── Sampler.sample() (采样)
            └── clear_global_ctx
```

### Engine

```python
class Engine:
    def __init__(self, *, model_path, model_config, dtype, device, ...):
        # 1. 在 meta device 上创建模型骨架（零内存）
        with torch.device("meta"):
            self.model = create_model(model_path, model_config)
        # 2. 从 safetensors 直接加载权重
        load_weights(self.model, model_path, device, dtype)
        # 3. 将 RoPE cache 移到 GPU
        self.model.model._rotary_emb.set_device(device)
        # 4. 创建 KV cache
        self.kv_cache = MHAKVCache(...)
        # 5. 设置 flashinfer 注意力后端
        self.ctx.attn_backend = FlashInferBackend(model_config)
```

### LLM 入口

```python
class LLM:
    def generate(self, prompts, sampling_params=None, ...):
        """Continuous-batching 生成"""
        for ids, sp in zip(all_input_ids, params_list):
            scheduler.add_request(ids, sp)

        while scheduler.has_work:
            forward_input = scheduler.schedule_next_batch()
            batch = forward_input.batch
            next_tokens = self.engine.forward_batch(batch)
            scheduler.process_batch_output(forward_input, next_tokens)

        return scheduler.collect_results(self.tokenizer)
```

---

## 架构演进总结

| 层面 | run_qwen3.py | 重构后 |
|------|-------------|--------|
| 算子基类 | nn.Module | BaseOP |
| 线性层 | 独立 Q/K/V/O + gate/up/down | QKV 融合 + gate_up 融合 |
| RMSNorm | 纯 PyTorch | flashinfer kernel + fused residual add |
| 激活函数 | F.silu(gate) * up | flashinfer silu_and_mul |
| 注意力 | 手写 softmax | FlashInfer PageAttention |
| 前向参数 | 函数传参 | 全局 Context |
| 配置读取 | AutoConfig | from_hf() / from_json() |
| 权重加载 | HF 中转 → state_dict | safetensors 直读 + fused 层自动打包 |
| 生成方式 | 简单自回归循环 | Scheduler continuous batching |
| KV Cache | 无 | Paged KV Cache |

每一步改造都保持推理结果与 run_qwen3.py 一致，但架构清晰、性能大幅提升。
