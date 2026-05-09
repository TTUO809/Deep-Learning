# Implementation Checklist

## Data Pipeline

- [x] 實作 `ICLEVRDataset`（讀取 train.json + 圖像）
- [x] Multi-label → one-hot vector（24 dims）
- [x] Image normalization：`Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))`
- [x] DataLoader with shuffle, batch
- [x] DataLoader (Test/New_Test): shuffle=False

## Condition Embedding

- [x] Label embedding（Linear / MLP）
- [x] Time embedding（Sinusoidal + MLP）
- [x] 確認 condition 正確注入 UNet（AdaGN）

## Noise Schedule

- [x] 選擇 cosine schedule（`squaredcos_cap_v2`）
- [x] 預計算 $\alpha_t$、$\bar{\alpha}_t$、$\beta_t$ 等（透過 diffusers scheduler）

## Model（UNet）

- [x] ResBlock（with time + condition embedding via AdaGN）
- [x] Attention block（multi-head self-attn at 32×32, 16×16, 8×8）
- [x] Downsampling blocks（BigGAN-style strided conv）
- [x] Bottleneck（ResBlock → Attention → ResBlock）
- [x] Upsampling blocks（nearest-neighbor + conv，with skip connections）
- [x] Output Conv → 3 channels（zero-initialized）

## Training

- [x] Forward process（`scheduler.add_noise`）
- [x] Noise prediction loss（MSE）
- [x] Optimizer（AdamW lr=2e-4 + CosineAnnealingLR）
- [x] EMA model（decay=0.9999，warmup after 2000 steps）
- [x] Checkpoint 儲存（every 10 epochs + last.pt）
- [x] Mixed-precision training（torch.amp）
- [x] 200 epochs 完整訓練（final loss ≈ 0.0012）

## Sampling

- [x] DDPM sampler（1000 steps）
- [x] DDIM sampler（configurable steps: 25/50/100/200）
- [x] Classifier guidance（x₀-space BCE gradient injection，`ddim_guided`）
- [x] Denormalization（`[-1,1] → [0,1]`）

## Evaluation

- [x] 載入 pretrained evaluator（ResNet18，frozen）
- [x] 對 `test.json` 生成 32 張圖像並計算 accuracy（94.44% with DDIM guided）
- [x] 對 `new_test.json` 生成圖像並計算 accuracy（91.67% with DDIM guided）
- [x] 完整 ablation log 記錄（notrain_all.log）

## Output

- [x] `images/test/0.png ... 31.png`
- [x] `images/new_test/0.png ... 31.png`
- [x] Synthetic image grid（test.json，8×4）→ `results/test_grid.png`
- [x] Synthetic image grid（new_test.json，8×4）→ `results/new_test_grid.png`
- [x] Denoising process grid（label: red sphere / cyan cylinder / cyan cube，8 steps）→ `results/denoise_process.png`
- [x] Ablation grids（sampler × steps × guidance scale × EMA，共 ~30 variants）

## Report

- [x] Introduction
- [x] Implementation details（架構 / noise schedule / embedding / loss / sampling）
- [x] Result grids + accuracy 截圖
- [x] Extra experiments discussion（sampler 比較、guidance scale sweep、EMA vs raw）
- [x] Reference

## 繳交

- [x] 壓縮成 `DL_LAB6_B11107122_凃岳霖.zip`
- [x] 包含：report (.pdf) + source code (.py) + images/
- [x] **不包含** dataset
- [x] 2026-05-12 23:59 前上傳
