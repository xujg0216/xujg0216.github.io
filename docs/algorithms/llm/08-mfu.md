# MFU

> Model FLOPs Utilization 模型算力利用率, 模型实际用了 GPU 理论峰值算力的多少比例

## 统计模型参数数量

```python
def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        # 统计模型里 所有参数的数量
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
```
==为什么要减 position embedding？==

wpe位置嵌入,只在 embedding 查表时用不参与大规模矩阵乘法
而 MFU 关注的是真正消耗 FLOPs 的参数

## MFU Flops计算

模型结构参数
```python
L = cfg.n_layer        # Transformer 层数
H = cfg.n_head         # 注意力头数
Q = cfg.n_embd // cfg.n_head  # 每个 head 的维度
T = cfg.block_size     # 序列长度
N = 参数总量

def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

```

**6N**

这里的 $N$ 是模型参数量。对于 Transformer 每一个参数，在一次训练迭代中：
- **前向传播（Forward）**：分别进行 1 次乘和加运算（约 **2** 个 FLOPs）。
- **反向传播（Backward）**：由于需要计算对**输入**的梯度和对**权重**的梯度，其计算量通常是前向传播的 **2 倍**（约 **4** 个 FLOPs）。

因此，每个 Token 在处理权重矩阵乘法时，总开销约为 $2 + 4 = 6N$


**12LHQT**

这一部分专门计算 **Self-Attention 中 $Q \cdot K^T$ 和 $\text{Score} \cdot V$** 的计算量，因为这两步不涉及权重矩阵（它们是激活值之间的乘法），所以不包含在 $N$ 里。

对于每一层 ($L$) 的每一个头 ($H$)：

1. **$Q \cdot K^T$**：形状是 $(T, Q) \times (Q, T)$。计算量是 $2 \cdot T^2 \cdot Q$。
    
2. **$\text{Score} \cdot V$**：形状是 $(T, T) \times (T, Q)$。计算量也是 $2 \cdot T^2 \cdot Q$。
    
3. **前向 + 反向**：同样乘以 3 倍，反向是正向的2倍
    
4. **汇总**：$L \times H \times (2T^2Q + 2T^2Q) \times 3 = 12LHQT$。
    

> **注意**：由于 $Q = \text{n\_embd} / H$，公式也可以简化为 $12 \cdot L \cdot T^2 \cdot \text{n\_embd}$。这部分开销随序列长度 $T$ 的**平方**增长。



$$MFU = \frac{\text{实际计算速度}}{\text{显卡理论峰值速度}}$$

- **`flops_achieved`**：通过公式算出的总计算量除以实际消耗的时间 $dt$。
    
- **`312e12`**：这是 NVIDIA A100 GPU 在使用 **bfloat16** 格式时的理论峰值（312 TFLOPS）。


    
