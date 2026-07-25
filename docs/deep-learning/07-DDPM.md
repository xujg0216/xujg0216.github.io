# 扩散模型

## 1. 前向扩散训练过程（Forward Diffusion / Noise Addition）

### 1.1 公式

前向扩散过程：


$$q(x_t | x_0) = \mathcal{N}\Big(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I \Big) $$


其中：

$$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s, \quad \alpha_s = 1 - \beta_s$$

然后训练目标是预测噪声 ($\epsilon$)：

$$\text{loss} = | \epsilon - \epsilon_\theta(x_t, t) |^2$$

---

### 1.2 伪代码
```text
输入: 原图 x0, 总步数 T, beta schedule (β1 ~ βT)
输出: MSE loss

1. 随机采样时间步 t ∈ [0, T-1]，每个样本独立
2. 生成噪声 noise ~ N(0, I)
3. 根据公式生成 x_t:
       x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * noise
4. 预测噪声 epsilon_pred = model(x_t, t)
5. 计算 loss = MSE(noise, epsilon_pred)
6. 返回 loss
```
### 1.3 Python 代码

```python

def extract(v, t, x_shape):
    """
    从 v 中根据时间步 t 提取对应的系数，并 reshape 为可以广播的形状。
    
    输入:
        v: [T] 或 [T, ...]，每个时间步对应的系数，比如 sqrt_alphas_bar
        t: [batch_size]，每个样本的时间步索引
        x_shape: 输入数据的形状，用于广播
    返回:
        [batch_size, 1, 1, ..., 1]，可以直接乘到 x_t 上
    """
    device = t.device
    # # 按索引 t 从 v 中取值，得到每个样本对应时间步的系数，out = [v[2], v[5], v[7]] = [2, 5, 7]
    out = torch.gather(v, index=t, dim=0).float().to(device) 
    # reshape 为 [batch_size, 1, 1, ...]，方便广播乘到 [B, C, H, W] 的 x_t
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        # UNet
        self.model = model
        # 总时间步长
        self.T = T
        # betas: 存储一个长度为T， 元素从beta_1线性变化到beta_T
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0) # 累积 ᾱ_t = ∏_{s=1}^{t} α_s

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # 计算前向扩散所需系数
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar)) # sqrt(ᾱ_t)
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar)) # sqrt(1-ᾱ_t)

    def forward(self, x_0):
        """
        Algorithm 1. 扩散步骤， 输入原始样本 x_0，返回训练损失
        """ 
        # 随机选择每个样本的时间步 t
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)  # 生成噪声 ε ~ N(0,1)
        # 根据公式 q(x_t | x_0) = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        # 训练目标: 预测噪声 ε，损失为 MSE
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss
```

## 2️. 反向采样过程（Reverse Sampling / Denoising）

### 2.1 公式

DDPM 反向采样：

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

其中：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\Big) $$

$$\sigma_t^2 = \beta_t \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}$$

采样公式：

$$x_{t-1} = \mu_\theta(x_t, t) + \sqrt{\sigma_t^2} \cdot \text{noise}, \quad \text{noise} \sim N(0, I)$$

最后一步 (t=0) 不加噪声。

---

### 2.2 伪代码

```text
输入: 噪声图 x_T, 总步数 T, beta schedule
输出: 生成图 x0

1. x_t = x_T
2. 对 time_step 从 T-1 到 0:
       t = [time_step] * batch_size
       eps = model(x_t, t)             # 预测噪声
       mean = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1-alpha_bar_t)) * eps)
       var = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
       if time_step > 0:
           noise = N(0, I)
       else:
           noise = 0
       x_t = mean + sqrt(var) * noise
3. return clip(x_t, -1, 1)
```

---

### 2.3 Python 代码

```python

class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T
        # 线性betas
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # ᾱ_{t-1}，方便计算后向采样
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        # coeff1, coeff2 用于从 x_t 预测 x_{t-1} 的均值 μ
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        # 去噪过程方差 σ²_t
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        """
        根据公式 μ = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε) 预测 x_{t-1} 均值
        """
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        """
        返回每一步的 μ 和 σ²
        """
        # 选择每个时间步对应的方差
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T):
        """
        Algorithm 2. 采样过程: 从 x_T ~ N(0,1) 逐步采样到 x_0
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)
            # 构造时间步张量 t
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t)
            # t>0 时加噪声，否则 t=0 不加噪声
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            # 采样公式: x_{t-1} = μ + sqrt(σ²) * ε
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)   

```