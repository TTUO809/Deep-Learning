# Lab 6 — Conditional DDPM on iCLEVR

Conditional Denoising Diffusion Probabilistic Model (DDPM) that generates 64×64 RGB
images of 3D objects conditioned on a multi-label (24-class) one-hot vector.
Dataset: iCLEVR (~18k training images).

## Results

| Split    | Sampler                      | Accuracy |
|----------|------------------------------|----------|
| test     | DDIM 100 + guidance (s=1.0)  | **98.61%** |
| new_test | DDIM 100 + guidance (s=1.0)  | **96.43%** |
| test     | DDPM 1000 steps              | 97.22%   |
| new_test | DDPM 1000 steps              | 92.86%   |

## Directory Structure

```
Lab_6/
├── DL_LAB6_B11107122_凃岳霖/
│   ├── src/
│   │   ├── dataset.py         # iCLEVR dataset & dataloader
│   │   ├── model.py           # Conditional UNet (AdaGN ResBlocks + self-attention)
│   │   ├── sampler.py         # DDPM / DDIM / DDIM + classifier guidance
│   │   ├── utils.py           # EMA, seed, denormalize, schedulers
│   │   ├── train.py           # training entry point
│   │   ├── evaluate.py        # image generation + accuracy
│   │   └── denoise_process.py # denoising visualization
│   ├── checkpoints/           # epoch snapshots + last.pt
│   ├── images/                # generated images (test / new_test splits)
│   ├── results/               # image grids and denoising figure
│   ├── file/                  # provided evaluator, data JSONs
│   ├── iclevr/                # training images (not in repo)
│   └── run.sh                 # full pipeline orchestration
├── Materials/                 # course reference material
└── Report/
    └── main.tex
```

## Environment

```bash
conda create -n DL_LAB6 python=3.10 -y
conda activate DL_LAB6
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install diffusers accelerate tqdm pillow wandb
```

## Usage

```bash
cd DL_LAB6_B11107122_凃岳霖/src

# Train (200 epochs, AMP, cosine schedule)
python train.py --epochs 200 --batch_size 64 --amp

# Evaluate — generate images + accuracy
python evaluate.py --ckpt ../checkpoints/last.pt \
    --sampler ddim_guided --steps 100 --guidance_scale 1.0

# Denoising visualization
python denoise_process.py --ckpt ../checkpoints/last.pt \
    --scheduler ddpm --snapshots 8

# Full pipeline (train → eval → denoise → ablation)
cd DL_LAB6_B11107122_凃岳霖 && bash run.sh
```

## Model Overview

- **Architecture**: Conditional UNet, base channels 128, multipliers (1, 2, 2, 4)
- **Conditioning**: sinusoidal time embedding (512-d) + label MLP (256-d) → AdaGN
- **Attention**: multi-head self-attention at 32×32, 16×16, 8×8 resolutions
- **Schedule**: cosine (`squaredcos_cap_v2`), 1000 timesteps
- **Training**: AdamW lr=2e-4, cosine LR decay, EMA decay=0.9999, 200 epochs
- **Sampling**: DDIM (100 steps) + classifier guidance from provided ResNet18 evaluator

Full implementation details are in [DL_LAB6_B11107122_凃岳霖/LAB6_B11107122_Report.pdf](DL_LAB6_B11107122_凃岳霖/LAB6_B11107122_Report.pdf).
