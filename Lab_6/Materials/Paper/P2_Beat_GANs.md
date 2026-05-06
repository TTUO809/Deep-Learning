# Paper Notes: Diffusion Models Beat GANs on Image Synthesis

> **Title:** Diffusion Models Beat GANs on Image Synthesis  
> **Authors:** Prafulla Dhariwal, Alex Nichol (OpenAI)  
> **Venue:** NeurIPS 2021  
> **Code:** https://github.com/openai/guided-diffusion  
> **縮寫：** 模型稱為 **ADM**（Ablated Diffusion Model），加 guidance 稱為 **ADM-G**

---

## 核心貢獻

1. 透過 **UNet 架構系統性 ablation**，找出能大幅提升 FID 的改進組合
2. 提出 **Classifier Guidance**：用預訓練分類器的梯度引導 diffusion 採樣，以 diversity 換 fidelity
3. 同時適用於 DDPM（stochastic）與 DDIM（deterministic）兩種 sampler

---

## Part 1：架構改進（ADM）

### 最終採用的最佳架構設定

|改動|說明|FID 效果|
|---|---|---|
|**深度 → 寬度**|增加 channel width 而非 depth|✅ 更快收斂|
|**Multi-head attention**|從 1 head → 多 head，或固定 64 ch/head|✅ FID -0.97~-1.36|
|**Multi-resolution attention**|Attention at **32×32, 16×16, 8×8**（原本只有 16×16）|✅ FID -0.72|
|**BigGAN residual blocks**|用於 up/downsampling（取代 strided conv）|✅ FID -1.20（最大提升）|
|Rescale residual connections|乘以 $\frac{1}{\sqrt{2}}$|❌ 略有退步|

> **最終預設架構**：variable width + 2 ResBlocks/resolution + multi-head (64 ch/head) + attention at 32/16/8 + BigGAN up/down + **AdaGN**

### AdaGN（Adaptive Group Normalization）⭐

這是本論文最重要的 condition 注入機制，對 Lab 直接有用：

$$\text{AdaGN}(h, y) = y_s \cdot \text{GroupNorm}(h) + y_b$$

- $h$：ResBlock 第一個 conv 後的 feature map
- $y = [y_s, y_b]$：由 **timestep embedding + class embedding 的線性投影** 得到
- 比 DDPM 原版的 addition + GroupNorm 明顯更好（FID 13.06 vs 15.08）

```python
# AdaGN 實作概念
class ResBlock(nn.Module):
    def forward(self, x, t_emb, c_emb):
        # 合併 time 與 condition embedding
        y = self.proj(torch.cat([t_emb, c_emb], dim=-1))  # → [y_s, y_b]
        y_s, y_b = y.chunk(2, dim=-1)
        
        h = self.conv1(x)
        h = self.norm(h)                    # GroupNorm
        h = h * (1 + y_s[..., None, None]) + y_b[..., None, None]  # AdaGN
        h = self.act(h)
        h = self.conv2(h)
        return h + self.shortcut(x)
```

### Learned Variance（可選）

原始 DDPM 使用 fixed variance；本論文採用 **learned variance**，插值於 $\beta_t$ 和 $\tilde\beta_t$ 之間：

$$\Sigma_\theta(x_t, t) = \exp!\left(v \log \beta_t + (1-v) \log \tilde\beta_t\right)$$

- $v$ 由 UNet 額外輸出 head 預測
- 對 **少步數採樣（< 1000 步）有明顯幫助**
- Training 用 hybrid loss：$L = L_\text{simple} + \lambda L_\text{vlb}$（$\lambda = 0.001$）

---

## Part 2：Classifier Guidance

### 核心思想

用預訓練分類器 $p_\phi(y|x_t)$ 的梯度來修改 diffusion 的 sampling distribution：

$$p_{\theta,\phi}(x_t | x_{t+1}, y) \propto p_\theta(x_t | x_{t+1}) \cdot p_\phi(y | x_t)$$

### Algorithm 1：Classifier Guided DDPM Sampling

```
Input: class label y, gradient scale s
x_T ~ N(0, I)
for t = T, ..., 1:
    μ, Σ = μ_θ(x_t), Σ_θ(x_t)          # diffusion model prediction
    g = ∇_{x_t} log p_φ(y | x_t)        # classifier gradient
    x_{t-1} ~ N(μ + s·Σ·g, Σ)           # shifted mean
return x_0
```

數學推導：將 $\log p_\phi(y|x_t)$ 在 $x_t = \mu$ 處做 Taylor 展開，得到 mean 位移量為 $\Sigma g$。

### Algorithm 2：Classifier Guided DDIM Sampling ⭐（Lab 推薦用）

DDIM 不能直接用 mean shift，改用 score-based 方式：

$$\hat\epsilon(x_t) = \epsilon_\theta(x_t) - \sqrt{1-\bar\alpha_t} \cdot \nabla_{x_t} \log p_\phi(y|x_t)$$

```
Input: class label y, gradient scale s
x_T ~ N(0, I)
for t = T, ..., 1:
    ε_hat = ε_θ(x_t) - s · √(1-ᾱ_t) · ∇_{x_t} log p_φ(y|x_t)
    x_{t-1} = √ᾱ_{t-1} · (x_t - √(1-ᾱ_t)·ε_hat) / √ᾱ_t + √(1-ᾱ_{t-1})·ε_hat
return x_0
```

```python
# Lab 實作：用 pretrained evaluator 做 classifier guidance
def classifier_guided_step(x_t, t, labels, eps_model, classifier, s):
    with torch.enable_grad():
        x_in = x_t.detach().requires_grad_(True)
        # evaluator 輸出 logits
        logits = classifier(x_in)   # shape: (B, 24)
        # 計算 log p(y|x_t)，labels 為 one-hot
        log_prob = (F.log_softmax(logits, dim=-1) * labels).sum(dim=-1).sum()
        grad = torch.autograd.grad(log_prob, x_in)[0]
    
    eps_pred = eps_model(x_t, t)
    sqrt_one_minus_alpha = (1 - alpha_bar[t]).sqrt()
    eps_hat = eps_pred - s * sqrt_one_minus_alpha * grad
    return eps_hat
```

### Gradient Scale $s$ 的效果

|Scale $s$|Fidelity（Precision/IS）|Diversity（Recall）|
|---|---|---|
|0（無 guidance）|低|高|
|1.0|中|中|
|**>1（推薦）**|**高**|**低**|
|>>10|極高，接近 adversarial|極低|

> 💡 $s > 1$ 等效於在 sharpened distribution $p(y|x)^s$ 下採樣，讓生成更聚焦於目標 class
> 
> 💡 **Lab 建議**：從 $s = 1\sim5$ 開始調，太大容易過度擬合

### Conditional vs Unconditional + Guidance（ImageNet 256×256）

|設定|FID|
|---|---|
|Unconditional，no guidance|26.21|
|Unconditional + guidance (s=10)|12.00|
|Conditional，no guidance|10.94|
|**Conditional + guidance (s=1)**|**4.59** ✅|

---

## 對本 Lab 的直接應用

### 架構改進採用清單

- [ ] Attention at **32×32, 16×16, 8×8**（不只 16×16）
- [ ] **BigGAN-style up/downsampling residual blocks**
- [ ] **AdaGN**：將 time embedding + label embedding 一起投影，透過 scale+shift 注入 GroupNorm
- [ ] Multi-head attention，64 ch/head

### Classifier Guidance 採用（可選，但可大幅提升分數）

- Evaluator（ResNet18）**不可修改**，但可繼承後 `requires_grad_(True)` 計算梯度
- 用 **DDIM + classifier guidance（Algorithm 2）** 是最實用組合
- gradient scale `s` 是最重要的超參數，建議搜尋 `[1, 2, 3, 5]`
- ⚠️ Evaluator 是在 **clean image** 上訓練，不是在 noisy $x_t$ 上 → guidance 效果在小 $t$（低噪聲）時更準

---

## 參考連結

- [arXiv 2105.05233](https://arxiv.org/abs/2105.05233)
- [GitHub: openai/guided-diffusion](https://github.com/openai/guided-diffusion)