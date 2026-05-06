# Paper Notes: HuggingFace Diffusion Models Course

> **來源：** https://github.com/huggingface/diffusion-models-class  
> **課程網站：** https://huggingface.co/learn/diffusion-course  
> **性質：** 實作導向教學，以 🤗 Diffusers 函式庫為主，搭配 from-scratch 實作對照

---

## 課程結構（與 Lab 相關 Units）

|Unit|主題|Lab 相關性|
|---|---|---|
|**Unit 1**|Introduction to Diffusion Models|⭐⭐⭐ 核心架構與訓練流程|
|**Unit 2**|Fine-Tuning, Guidance & Conditioning|⭐⭐⭐ Conditional DDPM 實作|
|Unit 3|Stable Diffusion|參考用（Latent Diffusion）|
|Unit 4|Advanced Techniques（DDIM Inversion 等）|選讀|

---

## 🤗 Diffusers 三大核心 API

```
Pipelines  → 高階封裝，快速生成（DDPMPipeline 等）
Models     → 模型架構（UNet2DModel, UNet2DConditionModel）
Schedulers → 控制 noise schedule 與 sampling（DDPMScheduler, DDIMScheduler）
```

> 💡 Scheduler 與 Model **解耦**，可以自由替換，不需改 model 就能切換 DDPM ↔ DDIM

---

## Unit 1：Diffusers UNet2DModel 使用方式

### 建立 UNet

```python
from diffusers import UNet2DModel

model = UNet2DModel(
    sample_size=64,              # 目標圖像解析度
    in_channels=3,               # RGB 輸入
    out_channels=3,              # 輸出維度
    layers_per_block=2,          # 每個 block 的 ResNet 層數
    block_out_channels=(128, 256, 256, 512),  # 各 level 的 channel 數
    down_block_types=(
        "DownBlock2D",           # 普通 ResNet downsampling
        "DownBlock2D",
        "AttnDownBlock2D",       # 含 self-attention 的 downsampling
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)
```

> ⚠️ AttnDownBlock2D / AttnUpBlock2D = 含 self-attention，對應論文的「attention at low resolution」 若要對應 P2 論文的 multi-resolution attention（32/16/8），可在多個 level 都使用 `AttnDownBlock2D`

### DDPMScheduler（訓練 + Sampling）

```python
from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="linear",       # 'linear' | 'scaled_linear' | 'squaredcos_cap_v2'(cosine)
    beta_start=1e-4,
    beta_end=0.02,
)

# 訓練：加噪
noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

# 評估/Sampling：去噪一步
output = noise_scheduler.step(model_output, t, x_t)
x_prev = output.prev_sample
```

### 標準訓練 Loop

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for batch in dataloader:
        clean_images = batch["images"].to(device)
        
        # 1. 隨機取 timestep
        timesteps = torch.randint(0, 1000, (B,), device=device).long()
        
        # 2. 加噪（forward process）
        noise = torch.randn_like(clean_images)
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        
        # 3. 預測噪聲
        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
        
        # 4. L_simple：MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

### DDPM Sampling Loop

```python
# 從純 noise 出發
x = torch.randn(batch_size, 3, 64, 64).to(device)
noise_scheduler.set_timesteps(1000)

for t in noise_scheduler.timesteps:   # T → 0
    with torch.no_grad():
        noise_pred = model(x, t).sample
    x = noise_scheduler.step(noise_pred, t, x).prev_sample
```

---

## Unit 2：Conditional Diffusion Model 實作

### 方法一：Channel Concatenation（最簡單）

將 label embedding 擴展成圖像大小，與 noisy image 在 channel 維度拼接後送入 UNet：

```python
class ClassConditionedUnet(nn.Module):
    def __init__(self, num_classes=24, class_emb_size=4):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, class_emb_size)
        self.model = UNet2DModel(
            sample_size=64,
            in_channels=3 + class_emb_size,  # ← 增加 input channels
            out_channels=3,
            ...
        )

    def forward(self, x, t, class_labels):
        bs, ch, w, h = x.shape
        cond = self.class_emb(class_labels)                        # (B, emb_size)
        cond = cond.view(bs, -1, 1, 1).expand(bs, -1, w, h)       # (B, emb_size, H, W)
        net_input = torch.cat([x, cond], dim=1)                    # (B, 3+emb, H, W)
        return self.model(net_input, t).sample
```

> ⚠️ **本 Lab 是 multi-label（one-hot 24 dims），不是 single class label**  
> 需改用 one-hot 直接投影：`nn.Linear(24, class_emb_size)` 而非 `nn.Embedding`

**Multi-label 版本：**

```python
class MultiLabelConditionedUnet(nn.Module):
    def __init__(self, num_classes=24, class_emb_size=4):
        super().__init__()
        self.class_proj = nn.Linear(num_classes, class_emb_size)  # one-hot → emb
        self.model = UNet2DModel(
            in_channels=3 + class_emb_size,
            ...
        )

    def forward(self, x, t, labels):  # labels: (B, 24) one-hot float
        bs, ch, w, h = x.shape
        cond = self.class_proj(labels)                           # (B, emb_size)
        cond = cond.view(bs, -1, 1, 1).expand(bs, -1, w, h)
        return self.model(torch.cat([x, cond], dim=1), t).sample
```

### 方法二：AdaGN 注入（推薦，效果最好）

參見 [[P2_Beat_GANs]] 的 AdaGN 段落，將 label embedding 與 time embedding 一起注入 ResBlock。

---

## DDIMScheduler（快速 Sampling）

DDIM 可以大幅減少 sampling 步數（1000 → 50~200 步）：

```python
from diffusers import DDIMScheduler

scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_schedule="linear",
)
scheduler.set_timesteps(50)  # 只需 50 步

x = torch.randn(B, 3, 64, 64).to(device)
for t in scheduler.timesteps:
    with torch.no_grad():
        noise_pred = model(x, t, labels).sample
    x = scheduler.step(noise_pred, t, x).prev_sample
```

|Scheduler|步數|特性|
|---|---|---|
|`DDPMScheduler`|1000 步|Stochastic，品質最高但最慢|
|`DDIMScheduler`|50~200 步|幾乎 deterministic，速度快|
|`DDIMScheduler(eta=1.0)`|—|等同 DDPM（加回隨機性）|

---

## Guidance（Unit 2）

Diffusers 課程介紹的 guidance 方式與 P2 論文一致，核心為在 sampling 每步加入 classifier 梯度：

```python
# DDIM + guidance 的手動 loop
for t in scheduler.timesteps:
    x_in = x.detach().requires_grad_(True)
    
    # Diffusion model prediction
    noise_pred = unet(x_in, t, labels).sample
    
    # Classifier guidance gradient
    with torch.enable_grad():
        log_prob = classifier(x_in, labels).log_softmax(-1)
        grad = torch.autograd.grad(log_prob.sum(), x_in)[0]
    
    # Modify noise prediction（DDIM 版）
    eps_hat = noise_pred - guidance_scale * (1 - alpha_bar[t]).sqrt() * grad
    
    x = scheduler.step(eps_hat, t, x).prev_sample
```

---

## 對本 Lab 的直接應用

### 推薦實作路線（由簡到強）

**Level 1（基礎）：Channel Concat Conditioning**

- 用 `nn.Linear(24, 4)` 把 one-hot 投影成 embedding
- Expand 成 (B, 4, H, W) 後 cat 到 noisy image
- 使用 `DDPMScheduler`（linear, T=1000）
- Loss: `F.mse_loss(noise_pred, noise)`

**Level 2（進階）：AdaGN Conditioning + DDIM**

- 參考 P2 論文，將 label+time embedding 用 AdaGN 注入每個 ResBlock
- Sampling 改用 `DDIMScheduler`（50~100 步），加快推論
- 可在 report 中比較兩種方法的 accuracy 差異（extra discussion 15%）

**Level 3（最強）：DDIM + Classifier Guidance**

- 繼承 evaluator class，在 sampling loop 加入 gradient guidance
- 調整 `guidance_scale` 超參數（建議搜尋 1, 2, 3, 5）
- 預期可將 accuracy 從 0.6~0.7 提升至 0.8+

### Beta Schedule 選擇

```python
# 使用 cosine schedule（比 linear 更適合 low-res 圖像）
DDPMScheduler(beta_schedule="squaredcos_cap_v2")

# 或 linear（與 DDPM 論文完全一致）
DDPMScheduler(beta_schedule="linear", beta_start=1e-4, beta_end=0.02)
```

---

## 參考連結

- [Unit 1: Diffusion from Scratch](https://huggingface.co/learn/diffusion-course/en/unit1/3)
- [Unit 2: Class-Conditioned Diffusion](https://huggingface.co/learn/diffusion-course/en/unit2/3)
- [Unit 2: Fine-Tuning and Guidance](https://huggingface.co/learn/diffusion-course/en/unit2/2)
- [Annotated Diffusion Model（深度程式碼解析）](https://huggingface.co/blog/annotated-diffusion)
- [DDPMScheduler API](https://huggingface.co/docs/diffusers/api/schedulers/ddpm)
- [DDIMScheduler API](https://huggingface.co/docs/diffusers/api/schedulers/ddim)