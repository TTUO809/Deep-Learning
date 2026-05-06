# Paper Notes: DDPM

> **Title:** Denoising Diffusion Probabilistic Models  
> **Authors:** Jonathan Ho, Ajay Jain, Pieter Abbeel (UC Berkeley)  
> **Venue:** NeurIPS 2020  
> **Code:** https://github.com/hojonathanho/diffusion

---

## 核心貢獻

1. 證明 diffusion model 能生成高品質圖像（FID 3.17 on CIFAR10，當時 SOTA）
2. 建立 **ε-prediction** 與 denoising score matching 的等價關係
3. 提出簡化 loss $L_\text{simple}$，比完整 ELBO 訓練效果更好

---

## 數學框架

### Forward Process（加噪，固定無參數）

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t;\ \sqrt{1-\beta_t},\mathbf{x}_{t-1},\ \beta_t \mathbf{I})$$

任意時間步 $t$ 的閉合形式（不需逐步計算）：

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t;\ \sqrt{\bar\alpha_t},\mathbf{x}_0,\ (1-\bar\alpha_t)\mathbf{I})$$

其中 $\alpha_t = 1 - \beta_t$，$\bar\alpha_t = \prod_{s=1}^{t} \alpha_s$

```python
# 閉合形式取樣（reparameterization）
x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * eps
# eps ~ N(0, I)
```

### Reverse Process（去噪，有參數 θ）

$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1};\ \boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\ \sigma_t^2 \mathbf{I})$$

- **Variance**：固定為常數，不學習
    - $\sigma_t^2 = \beta_t$（上界）或 $\sigma_t^2 = \tilde\beta_t$（下界），效果相似
- **Mean**：用 ε-prediction 參數化：

$$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}},\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)$$

---

## Loss Function

### 完整 ELBO（不推薦用於訓練）

$$L = L_T + \sum_{t>1} L_{t-1} + L_0$$

其中 $L_T$ 為常數（forward process 無參數），可忽略。

### 簡化 Loss $L_\text{simple}$（**推薦，效果最好**）

$$L_\text{simple}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\left|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta!\left(\sqrt{\bar\alpha_t},\mathbf{x}_0 + \sqrt{1-\bar\alpha_t},\boldsymbol{\epsilon},\ t\right)\right|^2\right]$$

本質上是 **MSE between predicted noise and actual noise**，去掉了各 $t$ 的加權係數。

> 💡 去掉加權使模型更專注於 large $t$（高噪聲）的困難任務，提升樣本品質

---

## 訓練與 Sampling 演算法

### Training（Algorithm 1）

```
repeat:
  x_0 ~ q(x_0)                          # 從資料集取樣
  t ~ Uniform({1, ..., T})              # 隨機取時間步
  ε ~ N(0, I)                           # 隨機取噪聲
  
  x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε   # 閉合形式加噪
  
  取梯度下降步：∇_θ ‖ε - ε_θ(x_t, t)‖²
until converged
```

### Sampling（Algorithm 2）

```
x_T ~ N(0, I)
for t = T, T-1, ..., 1:
  z ~ N(0, I) if t > 1, else z = 0
  x_{t-1} = (1/√α_t) * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t, t)) + σ_t * z
return x_0
```

> ⚠️ 原始 DDPM sampling 需要 **T=1000 步**，每步都要跑一次 UNet，很慢

---

## 模型架構（實驗細節）

|項目|設定|
|---|---|
|Backbone|**UNet**（基於 PixelCNN++ / Wide ResNet）|
|Normalization|**Group Normalization**（取代 weight norm）|
|Time embedding|**Transformer sinusoidal positional embedding**，注入每個 ResBlock|
|Attention|**Self-attention at 16×16 feature map resolution**|
|ResBlocks per level|2|
|解析度 levels|32×32 model: 4 levels（32→16→8→4）；256×256: 6 levels|
|Parameters|CIFAR10: ~35.7M；LSUN/CelebA-HQ: ~114M|

---

## Hyperparameters（最佳設定）

|參數|值|
|---|---|
|T|1000|
|β schedule|Linear，$\beta_1 = 10^{-4}$ → $\beta_T = 0.02$|
|Optimizer|**Adam**，lr = $2 \times 10^{-4}$（大圖用 $2 \times 10^{-5}$）|
|Batch size|128（CIFAR10）；64（256×256）|
|EMA decay|**0.9999**|
|Dropout|0.1（CIFAR10）；0（其他）|
|圖像 scaling|像素值線性縮放至 $[-1, 1]$|

---

## Ablation 重要結論

|設定|IS|FID|
|---|---|---|
|$\tilde\mu$ prediction + full $L$|8.06|13.22|
|$\epsilon$ prediction + full $L$|7.67|13.51|
|**$\epsilon$ prediction + $L_\text{simple}$（推薦）**|**9.46**|**3.17**|
|$\epsilon$ prediction + learned $\Sigma$|不穩定|—|

→ **結論：ε prediction + $L_\text{simple}$ + fixed variance 是最佳組合**

---

## 對本 Lab 的直接應用

- **架構**：UNet + Group Norm + Sinusoidal time embedding + Self-attn at 16×16
- **Loss**：用 $L_\text{simple}$（noise prediction MSE），不用 full ELBO
- **Variance**：固定為 $\sigma_t^2 = \beta_t$ 或 $\tilde\beta_t$，不學習
- **Schedule**：Linear β schedule，$T=1000$
- **Condition**：原論文是 unconditional，本 lab 需額外加 label embedding 注入每個 ResBlock
- **Sampling**：原版需 1000 步；可考慮改用 DDIM（參考論文三）加速

---

## 參考連結

- [arXiv 2006.11239](https://arxiv.org/abs/2006.11239)
- [GitHub: hojonathanho/diffusion](https://github.com/hojonathanho/diffusion)