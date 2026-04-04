# Lab 2 — Technical Details

Binary Semantic Segmentation on Oxford-IIIT Pet dataset using UNet and ResNet34-UNet.

---

## Table of Contents

- [Dataset](#dataset)
- [Data Preprocessing & Augmentation](#data-preprocessing--augmentation)
- [Model Architectures](#model-architectures)
  - [UNet](#unet)
  - [ResNet34-UNet](#resnet34-unet)
  - [CBAM Attention Module](#cbam-attention-module)
- [Loss Functions](#loss-functions)
- [Training Strategy](#training-strategy)
- [Inference & TTA](#inference--tta)
- [Results](#results)

---

## Dataset

**Oxford-IIIT Pet Dataset** — 37 breeds, ~7,400 images with pixel-wise trimap annotations.

| Split | Source | Size |
|-------|--------|------|
| `train.txt` | Kaggle provided | ~3,680 |
| `val.txt` | Kaggle provided | ~920 |
| `test_unet.txt` | Kaggle provided | ~3,669 |
| `test_res_unet.txt` | Kaggle provided | ~3,669 |

**Mask binarization**: trimap label `1` (foreground) → `1`; labels `2` (background) and `3` (border) → `0`.

---

## Data Preprocessing & Augmentation

All images are resized to **388×388** and normalized with ImageNet statistics
(`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

Purpose:
- Resize to a fixed spatial size so samples can be batched together.
- Keep the input size aligned with the UNet padding/output pipeline.
- Normalize input scale and distribution for more stable optimization.

Tensor conversion:
- Images and masks are converted from PIL images to PyTorch tensors before training.
- This is required for batching, GPU transfer, normalization, and loss computation.

**Training augmentations** (image and mask transformed identically):

| Augmentation | Probability | Parameters |
|---|---|---|
| Random horizontal flip | 50% | — |
| Random vertical flip | 50% | — |
| Random affine | 50% | rotation ±15°, translate ±10%, scale 0.9–1.1 |
| Color jitter (image only) | 50% | brightness/contrast/saturation ±20%, hue ±5% |

Purpose:
- Increase effective data diversity without collecting new images.
- Reduce overfitting to fixed pose, position, scale, or lighting.
- Improve generalization to unseen test images.

**No augmentation is applied during validation or inference.**

---

## Model Architectures

### UNet

Faithful re-implementation of [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597).

```
Input (3, 388, 388)
  │
  ├─ Pre-padding before UNet forward
  │   └─ reflect pad 92 px per side → Input (3, 572, 572)
  │
  ├─ Encoder
  │   ├─ DoubleConv(3→64)    [(64,  568, 568) skip 1] + MaxPool2d → (64,  284, 284)  
  │   ├─ DoubleConv(64→128)  [(128, 280, 280) skip 2] + MaxPool2d → (128, 140, 140)
  │   ├─ DoubleConv(128→256) [(256, 136, 136) skip 3] + MaxPool2d → (256,  68,  68)
  │   └─ DoubleConv(256→512) [(512,  64,  64) skip 4] + MaxPool2d → (512,  32,  32)
  │
  ├─ Bottleneck
  │   └─ DoubleConv(512→1024) → (1024, 28, 28)
  │
  ├─ Decoder
  │   ├─ ConvTranspose2d(1024→512) [(512,  56,  56)] + center_crop(skip 4) + DoubleConv(1024→512) → (512,  52,  52)
  │   ├─ ConvTranspose2d(512→256)  [(256, 104, 104)] + center_crop(skip 3) + DoubleConv(512→256)  → (256, 100, 100)
  │   ├─ ConvTranspose2d(256→128)  [(128, 200, 200)] + center_crop(skip 2) + DoubleConv(256→128)  → (128, 196, 196)
  │   └─ ConvTranspose2d(128→64)   [(64,  392, 392)] + center_crop(skip 1) + DoubleConv(128→64)   → (64, 388, 388)
  │
  └─ Conv1x1(64→1) → Output (1, 388, 388)
```

**Key details:**
- `DoubleConv`: two consecutive Conv 3×3 → ReLU (no padding, no BatchNorm — original paper faithful)
- Encoder feature maps are larger than decoder inputs due to no-padding convolutions; skip connections are aligned via **center crop**
- Pipeline uses external reflect padding before UNet forward so model output can align with 388×388 masks

---

### ResNet34-UNet

ResNet34 encoder + lightweight decoder with CBAM attention.

**Encoder** (ResNet34 from scratch, Kaiming He initialization):

| Stage | Operation | Output Channels | Stride |
|-------|-----------|-----------------|--------|
| conv1 | Conv 7×7 + BN + ReLU | 64 | 2 |
| conv2_x | MaxPool + 3 × BasicBlock | 64 | 1 |
| conv3_x | 4 × BasicBlock | 128 | 2 |
| conv4_x | 6 × BasicBlock | 256 | 2 |
| conv5_x | 3 × BasicBlock | 512 | 2 |

`BasicBlock`: Conv 3×3 → BN → ReLU → Conv 3×3 → BN → Add(shortcut) → ReLU.  
Shortcut uses Conv 1×1 + BN when stride ≠ 1 or channels change.

**Bottleneck**: Conv 3×3 (512→256) + BN + ReLU, then concatenated with layer4 output → **768ch**.

**Decoder** (5 × DecoderBlock, all output 32ch):

| Block | Input Channels | Skip Source | Output |
|-------|----------------|-------------|--------|
| dec1 | 768 | layer3 (256ch) | 32+256 = 288 |
| dec2 | 288 | layer2 (128ch) | 32+128 = 160 |
| dec3 | 160 | layer1 (64ch) | 32+64 = 96 |
| dec4 | 96 | — | 32 |
| dec5 | 32 | — | 32 |

Each `DecoderBlock`: Bilinear upsample (×2) → Conv 3×3 → BN → ReLU → **CBAM** → concat skip → output.

**Final**: Conv 1×1 (32→1), then interpolate to original input size.

---

### CBAM Attention Module

Applied after the conv block in each DecoderBlock.

**Channel Attention**:
$$\mathbf{M}_c = \sigma\!\left(W_1\!\left(\text{AvgPool}(F)\right) + W_1\!\left(\text{MaxPool}(F)\right)\right)$$

Shared MLP: reduction ratio = 16. Output multiplied element-wise with input.

**Spatial Attention**:
$$\mathbf{M}_s = \sigma\!\left(\text{Conv}_{7\times7}\!\left([\text{AvgPool}_c(F);\,\text{MaxPool}_c(F)]\right)\right)$$

Channel-wise avg and max pooled feature maps are concatenated → Conv 7×7 → Sigmoid. Output multiplied element-wise with input.

---

## Loss Functions

### Dice Score

$$\text{Dice} = \frac{2|A \cap B| + \epsilon}{|A| + |B| + \epsilon}, \quad \epsilon = 10^{-6}$$

- **Hard Dice** (evaluation): predictions binarized at threshold before computing
- **Soft Dice** (training): raw sigmoid probabilities used to preserve gradients

### FocalDiceLoss (used in training)

$$\mathcal{L} = 0.5 \cdot \mathcal{L}_\text{Focal} + 0.5 \cdot (1 - \text{SoftDice})$$

Focal Loss with $\alpha=0.5$, $\gamma=2.0$:

$$\mathcal{L}_\text{Focal} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Focuses training on hard-to-classify (mis-predicted) pixels; down-weights confident correct predictions.

### BCEDiceLoss (alternative)

$$\mathcal{L} = 0.5 \cdot \mathcal{L}_\text{BCE} + 0.5 \cdot (1 - \text{SoftDice})$$

---

## Training Strategy

| Component | Configuration |
|-----------|---------------|
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| LR Warmup | LinearLR, 5 epochs (factor 1e-3 → 1.0) |
| LR Schedule | CosineAnnealingLR, T_max=195, η_min=1e-6 |
| LR Combiner | SequentialLR (warmup → cosine) |
| Mixed Precision | `torch.amp.autocast` + `GradScaler` |
| Gradient Clipping | `max_norm=1.0` |
| Early Stopping | Monitor val Hard Dice, patience=20 |
| Random Seed | 42 (fully deterministic: Python / NumPy / PyTorch / cuDNN) |

LR schedule overview:

```
Epoch:   0───────5──────────────────────────────────────────200
LR:      ~0  ──▶ 5e-4  ──▶ (cosine decay) ──▶ 1e-6
          ↑ warmup ↑         ↑ cosine annealing ↑
```

The best model (by val Dice) is saved to `saved_models/{model}_best.pth` automatically.

---

## Inference & TTA

**Threshold**: `0.5` for UNet, `0.45` for ResNet34-UNet (selected via `--auto_threshold` sweep on val set).

**Test Time Augmentation (TTA)** — 4 forward passes averaged:

| Pass | Transform |
|------|-----------|
| 1 | Original |
| 2 | Horizontal flip → predict → flip back |
| 3 | Vertical flip → predict → flip back |
| 4 | H+V flip → predict → flip back |

Final probability = mean of 4 passes → binarize at threshold.

---

## Results

| Model | Epochs | Time/epoch | Best Val Dice | Kaggle Dice |
|-------|--------|------------|---------------|-------------|
| UNet | 145 | ~2:36 | 0.9080 | 0.91442 |
| ResNet34-UNet | 168 | ~0:49 | 0.9267 | 0.92958 |

UNet is slower per epoch mainly because the training/evaluation pipeline pads 388×388 inputs to 572×572 before forward, so much larger feature maps are processed.

ResNet34-UNet converges faster per epoch because the decoder outputs a fixed 32-channel representation, keeping computation lightweight despite the deeper encoder.
