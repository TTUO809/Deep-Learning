# DL Lab 6 — Conditional DDPM on iCLEVR

**Student ID:** B11107122  
**Name:** 凃岳霖 (Yue-Lin Tu)  
**Course:** Deep Learning, Spring 2026  
**Due:** 2026-05-12 23:59

---

## Results

| Split | Sampler | Steps | Guidance Scale | Accuracy |
|-------|---------|-------|----------------|----------|
| test | DDIM + classifier guidance | 100 | s=1.0 | **98.61%** |
| new_test | DDIM + classifier guidance | 100 | s=1.0 | **96.43%** |
| test | DDPM | 1000 | — | 97.22% |
| new_test | DDPM | 1000 | — | 92.86% |

Both splits exceed the 80% full-credit threshold.

---

## Directory Structure

```
DL_LAB6_B11107122_凃岳霖/
├── src/
│   ├── dataset.py          # iCLEVR dataset & dataloader (multi-label one-hot)
│   ├── model.py            # Conditional UNet (AdaGN ResBlocks + self-attention)
│   ├── sampler.py          # DDPM / DDIM / DDIM + classifier guidance samplers
│   ├── utils.py            # EMA, seed, denormalize, cosine LR scheduler
│   ├── train.py            # training entry point (AMP, EMA, wandb, checkpoint)
│   ├── evaluate.py         # image generation + accuracy via pretrained evaluator
│   └── denoise_process.py  # denoising visualisation (timestep grid)
├── images/
│   ├── test/               # 32 generated images for test.json (0.png … 31.png)
│   └── new_test/           # 32 generated images for new_test.json
└── LAB6_B11107122_Report.pdf
```

---

## Environment Setup

```bash
conda create -n DL_LAB6 python=3.10 -y
conda activate DL_LAB6
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install diffusers accelerate tqdm pillow wandb
```

The iCLEVR training images should be placed at `iclevr/` (not included in submission).

---

## Usage

```bash
cd src/

# Train from scratch (200 epochs, AMP, cosine LR, EMA)
python train.py --epochs 200 --batch_size 64 --amp

# Resume from checkpoint
python train.py --epochs 200 --batch_size 64 --amp --resume ../checkpoints/last.pt

# Evaluate — generate images/test/ and images/new_test/, print accuracy
python evaluate.py --ckpt ../checkpoints/last.pt \
    --sampler ddim_guided --steps 100 --guidance_scale 1.0

# Denoising visualisation (saves results/denoise_process.png)
python denoise_process.py --ckpt ../checkpoints/last.pt \
    --scheduler ddpm --snapshots 8

# Full pipeline (train → eval → denoise → ablation)
cd .. && bash run.sh

# Skip training, run eval + ablation only
bash run.sh --no-train
```

---

## Model Overview

| Component | Detail |
|-----------|--------|
| Architecture | Conditional UNet, base channels 128, channel multipliers (1, 2, 2, 4) |
| Conditioning | Sinusoidal time embedding (512-d) + label MLP (256-d) → AdaGN in each ResBlock |
| Attention | Multi-head self-attention at 32×32, 16×16, 8×8 resolutions |
| Noise schedule | Cosine (`squaredcos_cap_v2`), T=1000 timesteps |
| Training | AdamW lr=2e-4, CosineAnnealingLR, EMA decay=0.9999, 200 epochs, AMP |
| Sampling | DDPM (1000 steps), DDIM (configurable steps), DDIM + classifier guidance |
| Parameters | ~97M |

Full derivations and implementation details are in `LAB6_B11107122_Report.pdf`.
