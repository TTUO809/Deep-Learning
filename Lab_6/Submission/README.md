# DL Lab 6 — Conditional DDPM on iCLEVR

**Student:** B11107122 凃岳霖  
**Model:** Conditional DDPM with U-Net, EMA, DDIM / classifier-free guidance

---

## Directory structure

```
Submission/
├── train.py            # training script
├── evaluate.py         # inference + accuracy evaluation
├── denoise_process.py  # denoising-process visualization
├── dataset.py          # iCLEVR dataset / dataloader
├── model.py            # Conditional U-Net
├── sampler.py          # DDPM / DDIM / guided samplers + evaluator wrapper
├── utils.py            # EMA, seed, helpers
├── run.sh              # one-shot pipeline (train → eval → visualize)
├── checkpoints/
│   └── last.pt         # trained checkpoint (EMA weights inside)
├── images/             # pre-generated output images
│   ├── test/           # 32 generated images for test.json
│   └── new_test/       # 32 generated images for new_test.json
├── file/               # spec files (see "Data setup" below)
│   ├── objects.json
│   ├── train.json
│   ├── test.json
│   ├── new_test.json
│   ├── evaluator.py
│   └── checkpoint.pth
└── LAB6_B11107122_Report.pdf
```

---

## Environment

```bash
conda create -n DL_LAB6 python=3.10
conda activate DL_LAB6
pip install torch torchvision diffusers tqdm wandb pillow numpy
```

---

## Data setup

Place the following inside `Submission/file/` (provided in the lab spec):

| File | Purpose |
|---|---|
| `objects.json` | 24-class label mapping |
| `train.json` | 18009 training samples |
| `test.json` | 32 test conditions |
| `new_test.json` | 32 new-test conditions |
| `evaluator.py` | official ResNet18 evaluator |
| `checkpoint.pth` | evaluator weights |

For training only, also place the iCLEVR images in `Submission/iclevr/`.

If your data lives in a different directory, set `DATA_ROOT` at the top of `run.sh`:

```bash
DATA_ROOT="/path/to/your/data"   # must contain file/ and iclevr/
```

---

## Quick start

### Eval-only (use the provided checkpoint, no GPU training needed for this step)

```bash
conda activate DL_LAB6
bash run.sh --no-train --no-ablation
```

Outputs:
- `images/test/` and `images/new_test/` — generated images
- `results/test_grid.png`, `results/new_test_grid.png` — 8×4 grid
- `results/denoise_process.png` — denoising timeline
- Accuracy printed to stdout

### Train from scratch then evaluate

```bash
bash run.sh
```

The script auto-detects `checkpoints/last.pt` and resumes if it exists.

### Skip conda activation (if your env is already active)

Comment out the two `conda` lines near the top of `run.sh`:

```bash
# eval "$(conda shell.bash hook)"
# conda activate $CONDA_ENV
```

Then just:

```bash
bash run.sh --no-train
```

---

## Manual commands

```bash
# Evaluate with DDIM + classifier guidance (100 steps, scale=1.0)
python evaluate.py \
    --ckpt checkpoints/last.pt \
    --obj_json      file/objects.json \
    --test_json     file/test.json \
    --new_test_json file/new_test.json \
    --out_dir       images \
    --grid_dir      results \
    --sampler ddim_guided --steps 100 --guidance_scale 1.0

# Denoising-process visualization
python denoise_process.py \
    --ckpt checkpoints/last.pt \
    --obj_json file/objects.json \
    --out results/denoise_process.png \
    --scheduler ddpm --snapshots 8

# Train (with W&B logging; remove --wandb to disable)
python train.py \
    --img_dir    iclevr \
    --train_json file/train.json \
    --obj_json   file/objects.json \
    --ckpt_dir   checkpoints \
    --epochs 200 --batch_size 16 --amp --wandb
```

---

## Results

| Split | Accuracy |
|---|---|
| test | **0.9861** |
| new_test | **0.9405** |

Sampler: DDIM, 100 steps, guidance scale 1.0, EMA weights.
