"""Evaluate a trained checkpoint on test.json / new_test.json.

Outputs:
  images/<split>/<i>.png      — individual generated images (denormalized to [0,1])
  results/<split>_grid.png    — 8x4 grid
  Prints accuracy from the official ResNet18 evaluator.

Example:
  python src/evaluate.py --ckpt checkpoints/last.pt \
      --sampler ddim_guided --guidance_scale 3.0 --steps 100
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from diffusers import DDIMScheduler, DDPMScheduler

from model import ConditionalUNet
from sampler import GuidedEvaluator, ddim_guided_sample, ddim_sample, ddpm_sample
from utils import denormalize, set_seed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--obj_json", default=str(ROOT / "file" / "objects.json"))
    p.add_argument("--test_json", default=str(ROOT / "file" / "test.json"))
    p.add_argument("--new_test_json", default=str(ROOT / "file" / "new_test.json"))
    p.add_argument("--out_dir", default=str(ROOT / "images"))
    p.add_argument("--grid_dir", default=str(ROOT / "results"))
    p.add_argument("--sampler", choices=["ddpm", "ddim", "ddim_guided"],
                   default="ddim_guided")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--guidance_scale", type=float, default=3.0)
    p.add_argument("--beta_schedule", choices=["linear", "squaredcos_cap_v2"],
                   default="squaredcos_cap_v2")
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_ema", dest="use_ema", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=32, help="generate N images per pass")
    p.add_argument("--tag", default="", help="tag appended to output dirs (for ablation runs)")
    return p.parse_args()


def labels_list_to_onehot(label_lists, obj_map, device):
    n_cls = len(obj_map)
    out = torch.zeros(len(label_lists), n_cls, device=device)
    for i, lbs in enumerate(label_lists):
        for lb in lbs:
            out[i, obj_map[lb]] = 1.0
    return out


def make_scheduler(args):
    if args.beta_schedule == "linear":
        return (
            DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear",
                          beta_start=1e-4, beta_end=0.02),
            DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear",
                          beta_start=1e-4, beta_end=0.02),
        )
    return (
        DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"),
        DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"),
    )


def generate_for_split(model, labels_onehot, sampler, args, device, ddim_sched, ddpm_sched,
                       guided=None):
    images = []
    n = labels_onehot.size(0)
    for start in range(0, n, args.batch):
        chunk = labels_onehot[start:start + args.batch]
        if sampler == "ddpm":
            img = ddpm_sample(model, chunk, ddpm_sched, device,
                              num_steps=args.steps if args.steps != 1000 else None)
        elif sampler == "ddim":
            img = ddim_sample(model, chunk, ddim_sched, device, steps=args.steps)
        elif sampler == "ddim_guided":
            assert guided is not None
            img = ddim_guided_sample(model, chunk, ddim_sched, guided, device,
                                     steps=args.steps, guidance_scale=args.guidance_scale)
        else:
            raise ValueError(sampler)
        images.append(img)
    return torch.cat(images, dim=0)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load checkpoint ----
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = ConditionalUNet().to(device)
    if args.use_ema and "ema" in ck:
        model.load_state_dict(ck["ema"])
        print("[ckpt] loaded EMA weights")
    else:
        model.load_state_dict(ck["model"])
        print("[ckpt] loaded raw model weights")
    model.eval()

    ddim_sched, ddpm_sched = make_scheduler(args)

    guided = None
    evaluator = None
    if args.sampler == "ddim_guided":
        guided = GuidedEvaluator()
        evaluator = guided
    else:
        # Still need an evaluator to compute accuracy.
        guided = GuidedEvaluator()
        evaluator = guided

    with open(args.obj_json) as f:
        obj_map = json.load(f)

    splits = [
        ("test", args.test_json),
        ("new_test", args.new_test_json),
    ]

    summary = {}
    for split_name, json_path in splits:
        with open(json_path) as f:
            data = json.load(f)
        labels_onehot = labels_list_to_onehot(data, obj_map, device)
        print(f"[{split_name}] generating {len(data)} images ...")
        images = generate_for_split(
            model, labels_onehot, args.sampler, args, device, ddim_sched, ddpm_sched, guided,
        )

        # ---- Save individual + grid ----
        tag = f"_{args.tag}" if args.tag else ""
        out_dir = os.path.join(args.out_dir, split_name + tag)
        os.makedirs(out_dir, exist_ok=True)
        denorm = denormalize(images)
        for i in range(denorm.size(0)):
            save_image(denorm[i], os.path.join(out_dir, f"{i}.png"))
        os.makedirs(args.grid_dir, exist_ok=True)
        grid_path = os.path.join(args.grid_dir, f"{split_name}{tag}_grid.png")
        save_image(make_grid(denorm, nrow=8), grid_path)
        print(f"[{split_name}] saved -> {out_dir}/, grid -> {grid_path}")

        # ---- Accuracy ----
        # Evaluator expects normalized images in [-1, 1] (as in transforms.Normalize).
        eval_images = images.clamp(-1.0, 1.0)
        # Move to GPU for evaluator (it lives on cuda).
        acc = evaluator.eval(eval_images.cuda(), labels_onehot.cuda())
        print(f"[{split_name}] accuracy = {acc:.4f}")
        summary[split_name] = acc

    print("\n==== SUMMARY ====")
    for k, v in summary.items():
        print(f"  {k:>10s}: {v:.4f}")


if __name__ == "__main__":
    main()
