# Implementation Checklist

## Data Pipeline

- [ ] 實作 `ICLEVRDataset`（讀取 train.json + 圖像）
- [ ] Multi-label → one-hot vector（24 dims）
- [ ] Image normalization：`Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))`
- [ ] DataLoader with shuffle, batch

## Condition Embedding

- [ ] Label embedding（Linear / MLP）
- [ ] Time embedding（Sinusoidal + MLP）
- [ ] 確認 condition 正確注入 UNet（AdaGN / concat / cross-attn）

## Noise Schedule

- [ ] 選擇 linear 或 cosine schedule
- [ ] 預計算 $\alpha_t$、$\bar{\alpha}_t$、$\beta_t$ 等

## Model（UNet）

- [ ] ResBlock（with time + condition embedding）
- [ ] Attention block（self-attn at low resolution）
- [ ] Downsampling blocks
- [ ] Bottleneck
- [ ] Upsampling blocks（with skip connections）
- [ ] Output Conv → 3 channels

## Training

- [ ] Forward process（q_sample）
- [ ] Noise prediction loss（MSE）
- [ ] Optimizer（AdamW + LR scheduler）
- [ ] EMA model（optional but recommended）
- [ ] Checkpoint 儲存

## Sampling

- [ ] DDPM 或 DDIM sampler
- [ ] Classifier guidance（optional, 可提升分數）
- [ ] Denormalization（`[-1,1] → [0,1]`）

## Evaluation

- [ ] 載入 pretrained evaluator
- [ ] 對 `test.json` 生成 32 張圖像並計算 accuracy
- [ ] 對 `new_test.json` 生成圖像並計算 accuracy
- [ ] 截圖 accuracy 結果

## Output

- [ ] `images/test/0.png ... N.png`
- [ ] `images/new_test/0.png ... N.png`
- [ ] Synthetic image grid（test.json，8×4）
- [ ] Synthetic image grid（new_test.json，8×4）
- [ ] Denoising process grid（label: red sphere / cyan cylinder / cyan cube，≥8 steps）

## Report

- [ ] Introduction
- [ ] Implementation details（架構 / noise schedule / embedding / loss / sampling）
- [ ] Result grids + accuracy 截圖
- [ ] Extra experiments discussion（至少比較 1-2 個設定）

## 繳交

- [ ] 壓縮成 `DL_LAB6_YourStudentID_YourName.zip`
- [ ] 包含：report (.pdf) + source code (.py) + images/
- [ ] **不包含** dataset
- [ ] 2026-05-12 23:59 前上傳