# Model Architecture

> 綜合來源：[[P1_DDPM]](./Paper/P1_DDPM.md) · [[P2_Beat_GANs]](./Paper/P2_Beat_GANs.md) · [[P3_HF_Diffusion_Course]](./Paper/P3_HF_Diffusion_Course.md)
> **目標**：64×64 RGB，conditional DDPM，條件為 24-dim multi-label one-hot

---

## 整體設計決策

|項目|選擇|論文依據|
|---|---|---|
|操作空間|**Pixel space**|64×64 不需 latent|
|Backbone|**UNet**|P1 標準選擇|
|Normalization|**Group Normalization**|P1 取代 weight norm|
|Condition 注入|**AdaGN**（time + label 一起）|P2：比 add+GN 好（FID 13.06 vs 15.08）|
|Attention resolution|**32×32, 16×16, 8×8**|P2：multi-res 比只在 16×16 好（FID -0.72）|
|Up/Downsampling|**BigGAN-style ResBlock**|P2：最大單一 FID 提升（-1.20）|
|Attention heads|**固定 64 ch/head**|P2：FID -1.36，與 Transformer 對齊|

---

## UNet 架構（64×64 輸入）

```
Input: x_t (B,3,64,64)  t (B,)  labels (B,24)
            │
     ┌──────▼──────┐
     │  Init Conv  │ → (B, 128, 64, 64)
     └──────┬──────┘
            │                              ← skip 1
  ┌─────────▼──────────────────────────────────────┐ ENCODER
  │ Down1: ResBlock×2  128ch  64×64   [no attn]    │ ← skip 2,3
  │ Down2: ResBlock×2  256ch  32×32 + AttnBlock     │ ← skip 4,5
  │ Down3: ResBlock×2  256ch  16×16 + AttnBlock     │ ← skip 6,7
  │ Down4: ResBlock×2  512ch   8×8  + AttnBlock     │ ← skip 8,9
  └─────────┬──────────────────────────────────────┘
            │
  ┌─────────▼──────┐ BOTTLENECK
  │ ResBlock + Attn + ResBlock  512ch  8×8           │
  └─────────┬──────┘
            │
  ┌─────────▼──────────────────────────────────────┐ DECODER
  │ Up4:  ResBlock×2  512ch   8×8  + AttnBlock     │ (+ skip 8,9)
  │ Up3:  ResBlock×2  256ch  16×16 + AttnBlock     │ (+ skip 6,7)
  │ Up2:  ResBlock×2  256ch  32×32 + AttnBlock     │ (+ skip 4,5)
  │ Up1:  ResBlock×2  128ch  64×64  [no attn]      │ (+ skip 2,3)
  └─────────┬──────────────────────────────────────┘
            │
     ┌──────▼──────┐
     │  GroupNorm + SiLU + Conv  │ → (B, 3, 64, 64)  ε̂
     └─────────────┘
```

每個 ResBlock 都透過 **AdaGN** 接收 `time_emb ‖ label_emb`

---

## 1. Time Embedding（P1）

```python
import math, torch, torch.nn as nn

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):                        # t: (B,) LongTensor
        half  = self.dim // 2
        freqs = torch.exp(-math.log(10000) *
                          torch.arange(half, device=t.device) / half)
        args  = t.float()[:, None] * freqs[None] # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)

class TimeEmbeddingMLP(nn.Module):
    def __init__(self, base_dim=128, time_emb_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalTimeEmbedding(base_dim),
            nn.Linear(base_dim, time_emb_dim), nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
    def forward(self, t): return self.net(t)     # (B, time_emb_dim)
```

---

## 2. Label Embedding（Multi-label, 24 classes）

> ⚠️ **本 Lab 是 multi-label one-hot，不是 single class index，不能用 `nn.Embedding`**

```python
class LabelEmbedding(nn.Module):
    """(B, 24) float one-hot → (B, label_emb_dim)"""
    def __init__(self, num_classes=24, label_emb_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes, label_emb_dim), nn.SiLU(),
            nn.Linear(label_emb_dim, label_emb_dim),
        )
    def forward(self, labels): return self.net(labels)
```

---

## 3. AdaGN ResBlock（P2 核心）⭐

$$\text{AdaGN}(h,, y) = y_s \cdot \text{GroupNorm}(h) + y_b, \quad y=[y_s,y_b]=\text{Linear}(t_emb | l_emb)$$

```python
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, num_groups=32, dropout=0.1):
        super().__init__()
        self.norm1    = nn.GroupNorm(num_groups, in_ch)
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2    = nn.GroupNorm(num_groups, out_ch)
        self.conv2    = nn.Sequential(nn.Dropout(dropout),
                                      nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.act      = nn.SiLU()
        # AdaGN: cond → scale & shift for norm2
        self.adagn    = nn.Sequential(nn.SiLU(),
                                      nn.Linear(cond_dim, 2 * out_ch))
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):              # cond: (B, cond_dim)
        h = self.conv1(self.act(self.norm1(x)))
        scale, shift = self.adagn(cond).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.act(h))
        return h + self.shortcut(x)
```

---

## 4. Multi-Head Attention Block（P1 + P2）

P1：只在 16×16 加 self-attn（1 head）  
P2：擴展到 **32/16/8**，多 head，**64 ch/head**

```python
class AttentionBlock(nn.Module):
    def __init__(self, ch, num_groups=32):
        super().__init__()
        num_heads     = max(1, ch // 64)      # 64 ch/head（P2 建議）
        self.norm     = nn.GroupNorm(num_groups, ch)
        self.attn     = nn.MultiheadAttention(ch, num_heads, batch_first=True)
        self.proj_out = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H*W).permute(0, 2, 1)  # (B, HW, C)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + self.proj_out(h)
```

---

## 5. BigGAN-style Up/Downsampling（P2）

```python
class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        return self.conv(nn.functional.interpolate(x, scale_factor=2, mode='nearest'))
```

---

## 6. 完整 ConditionalUNet

```python
class ConditionalUNet(nn.Module):
    def __init__(
        self,
        in_ch        = 3,
        base_ch      = 128,
        ch_mults     = (1, 2, 2, 4),           # 128, 256, 256, 512
        num_res      = 2,
        attn_res     = {32, 16, 8},             # P2 multi-resolution attn
        time_emb_dim = 512,
        label_emb_dim= 256,
        num_classes  = 24,
        dropout      = 0.1,
    ):
        super().__init__()
        cond_dim = time_emb_dim + label_emb_dim

        self.time_emb  = TimeEmbeddingMLP(base_ch, time_emb_dim)
        self.label_emb = LabelEmbedding(num_classes, label_emb_dim)
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # ---------- Encoder ----------
        self.downs = nn.ModuleList()
        ch, res = base_ch, 64
        skip_chs = [ch]

        for i, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            for _ in range(num_res):
                self.downs.append(ResBlock(ch, out_ch, cond_dim, dropout=dropout))
                ch = out_ch
                skip_chs.append(ch)
                if res in attn_res:
                    self.downs.append(AttentionBlock(ch))
            if i != len(ch_mults) - 1:
                self.downs.append(Downsample(ch))
                skip_chs.append(ch)
                res //= 2

        # ---------- Bottleneck ----------
        self.mid = nn.ModuleList([
            ResBlock(ch, ch, cond_dim),
            AttentionBlock(ch),
            ResBlock(ch, ch, cond_dim),
        ])

        # ---------- Decoder ----------
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mults))):
            out_ch = base_ch * mult
            for j in range(num_res + 1):
                skip_ch = skip_chs.pop()
                self.ups.append(ResBlock(ch + skip_ch, out_ch, cond_dim, dropout=dropout))
                ch = out_ch
                if res in attn_res:
                    self.ups.append(AttentionBlock(ch))
            if i != 0:
                self.ups.append(Upsample(ch))
                res *= 2

        # ---------- Output ----------
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch), nn.SiLU(),
            nn.Conv2d(ch, in_ch, 3, padding=1),
        )

    def forward(self, x, t, labels):
        # labels: (B, 24) float one-hot
        cond = torch.cat([self.time_emb(t), self.label_emb(labels)], dim=-1)

        h = self.init_conv(x)
        skips = [h]

        for layer in self.downs:
            if isinstance(layer, (ResBlock,)):
                h = layer(h, cond)
            else:
                h = layer(h)
            if not isinstance(layer, Downsample):
                skips.append(h)

        for layer in self.mid:
            h = layer(h, cond) if isinstance(layer, ResBlock) else layer(h)

        for layer in self.ups:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, skips.pop()], dim=1)
                h = layer(h, cond)
            else:
                h = layer(h)

        return self.out(h)
```

---

## 7. Diffusers 快速替代（Baseline 用）

若要先跑通再優化，可用 `UNet2DModel` + channel concat：

```python
from diffusers import UNet2DModel

class CondUNetDiffusers(nn.Module):
    def __init__(self, num_classes=24, label_emb_dim=4):
        super().__init__()
        self.label_proj = nn.Linear(num_classes, label_emb_dim)
        self.unet = UNet2DModel(
            sample_size=64,
            in_channels=3 + label_emb_dim,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 256, 256, 512),
            down_block_types=("DownBlock2D", "AttnDownBlock2D",
                              "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=  ("AttnUpBlock2D", "AttnUpBlock2D",
                              "AttnUpBlock2D", "UpBlock2D"),
            attention_head_dim=64,            # P2 建議
        )

    def forward(self, x, t, labels):
        B, _, H, W = x.shape
        cond = self.label_proj(labels)[:, :, None, None].expand(-1, -1, H, W)
        return self.unet(torch.cat([x, cond], dim=1), t, return_dict=False)[0]
```

---

## 架構路線選擇

|路線|難度|預期 Accuracy|Report 討論空間|
|---|---|---|---|
|Diffusers + channel concat|★☆☆|0.5~0.7|Baseline|
|自製 UNet + AdaGN|★★★|0.7~0.8|高（設計選擇可討論）|
|自製 UNet + AdaGN + Classifier Guidance|★★★|**0.8+**|最高（論文級）|

---

## 參考

- [[P1_DDPM]](./Paper/P1_DDPM.md)：UNet + GroupNorm + Sinusoidal time embedding
- [[P2_Beat_GANs]](./Paper/P2_Beat_GANs.md)：AdaGN、multi-res attention、BigGAN up/down
- [[P3_HF_Diffusion_Course]](./Paper/P3_HF_Diffusion_Course.md)：Diffusers API、channel concat conditioning 實作