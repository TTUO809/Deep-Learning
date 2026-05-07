# DL Lab 6 — Conditional DDPM on iCLEVR (B11107122 凃岳霖)

Conditional DDPM that generates 64×64 RGB iCLEVR images from a 24-d multi-label
condition. Architecture is a UNet with **AdaGN** ResBlocks, multi-resolution self-
attention, EMA, and DDIM sampling with **classifier guidance** from the provided
ResNet18 evaluator.

## Layout

```
DL_LAB6_B11107122_凃岳霖/
├── file/                  # provided: evaluator.py, checkpoint.pth, *.json
├── iclevr/                # provided: 18,009 training images
├── src/
│   ├── dataset.py         # iCLEVR train / test datasets
│   ├── model.py           # AdaGN UNet
│   ├── sampler.py         # DDPM / DDIM / DDIM + classifier guidance
│   ├── utils.py           # EMA, denormalize, seed
│   ├── train.py           # training entry
│   ├── evaluate.py        # generates images + computes accuracy
│   └── denoise_process.py # report figure
├── checkpoints/
├── images/{test,new_test}/
└── results/               # grids, denoise visualization
```

## Environment

```bash
conda create -n DL_LAB6 python=3.10 -y
conda activate DL_LAB6
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install diffusers accelerate tqdm pillow
```

## Train

```bash
cd src
python train.py --epochs 200 --batch_size 64 --amp
```

Defaults: cosine β-schedule, AdamW lr=2e-4, EMA decay=0.9999, gradient clip=1.0.
Checkpoints land in `../checkpoints/` (`last.pt` always points to the latest).

## Evaluate (generate images + accuracy)

```bash
python evaluate.py --ckpt ../checkpoints/last.pt \
    --sampler ddim_guided --steps 100 --guidance_scale 3.0
```

This writes:
- `../images/test/{0..31}.png`, `../images/new_test/{0..31}.png`
- `../results/test_grid.png`, `../results/new_test_grid.png`
- prints accuracy for both splits (using the provided ResNet18 evaluator).

Other samplers for the report's ablation:

```bash
python evaluate.py --ckpt ../checkpoints/last.pt --sampler ddpm  --tag ddpm
python evaluate.py --ckpt ../checkpoints/last.pt --sampler ddim  --tag ddim
python evaluate.py --ckpt ../checkpoints/last.pt --sampler ddim_guided --guidance_scale 1 --tag g1
python evaluate.py --ckpt ../checkpoints/last.pt --sampler ddim_guided --guidance_scale 3 --tag g3
python evaluate.py --ckpt ../checkpoints/last.pt --sampler ddim_guided --guidance_scale 5 --tag g5
```

## Denoising-process figure

```bash
python denoise_process.py --ckpt ../checkpoints/last.pt \
    --scheduler ddpm --snapshots 8
```

Output: `../results/denoise_process.png` for label set
`["red sphere", "cyan cylinder", "cyan cube"]`.
