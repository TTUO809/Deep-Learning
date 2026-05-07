"""Generate the required denoising-process visualization for the report.

Label set (per spec): ["red sphere", "cyan cylinder", "cyan cube"]
Output: a 1-row grid of >=8 snapshots from x_T (noise) -> x_0 (clean).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from diffusers import DDIMScheduler, DDPMScheduler

from model import ConditionalUNet
from sampler import GuidedEvaluator
from utils import denormalize, set_seed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DEFAULT_LABELS = ["red sphere", "cyan cylinder", "cyan cube"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--obj_json", default=str(ROOT / "file" / "objects.json"))
    p.add_argument("--out", default=str(ROOT / "results" / "denoise_process.png"))
    p.add_argument("--steps", type=int, default=1000,
                   help="diffusion steps for the visualization (DDPM uses scheduler default)")
    p.add_argument("--snapshots", type=int, default=8,
                   help="number of denoising snapshots to display (>=8 by spec)")
    p.add_argument("--scheduler", choices=["ddpm", "ddim"], default="ddpm")
    p.add_argument("--beta_schedule", choices=["linear", "squaredcos_cap_v2"],
                   default="squaredcos_cap_v2")
    p.add_argument("--guidance_scale", type=float, default=0.0,
                   help="if >0 and scheduler==ddim, apply classifier guidance")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_ema", action="store_true", default=True)
    return p.parse_args()


def make_scheduler(args):
    kw = (
        dict(beta_schedule="linear", beta_start=1e-4, beta_end=0.02)
        if args.beta_schedule == "linear"
        else dict(beta_schedule="squaredcos_cap_v2")
    )
    if args.scheduler == "ddpm":
        return DDPMScheduler(num_train_timesteps=1000, **kw)
    return DDIMScheduler(num_train_timesteps=1000, **kw)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load model ----
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = ConditionalUNet().to(device)
    if args.use_ema and "ema" in ck:
        model.load_state_dict(ck["ema"])
        print("[ckpt] loaded EMA weights")
    else:
        model.load_state_dict(ck["model"])
    model.eval()

    # ---- Build label one-hot ----
    with open(args.obj_json) as f:
        obj_map = json.load(f)
    labels = torch.zeros(1, len(obj_map), device=device)
    for lb in DEFAULT_LABELS:
        labels[0, obj_map[lb]] = 1.0

    # ---- Set up scheduler ----
    sched = make_scheduler(args)
    sched.set_timesteps(args.steps)
    timesteps = sched.timesteps  # decreasing from T-1 -> 0
    n = len(timesteps)
    # Pick snapshot indices (uniformly spaced over the trajectory).
    snap_idx = set([int(round(i * (n - 1) / max(args.snapshots - 1, 1)))
                    for i in range(args.snapshots)])

    guided = None
    if args.guidance_scale > 0 and args.scheduler == "ddim":
        guided = GuidedEvaluator()
        alphas_cumprod = sched.alphas_cumprod.to(device)

    x = torch.randn(1, 3, 64, 64, device=device)
    snapshots = []
    snapshots.append(x.clone())  # x_T (pure noise)

    for i, t in enumerate(timesteps):
        t_b = torch.full((1,), int(t), device=device, dtype=torch.long)
        if guided is not None:
            with torch.no_grad():
                eps = model(x, t_b, labels)
            grad = guided.get_grad(x, labels)
            eps_hat = eps - args.guidance_scale * (1.0 - alphas_cumprod[int(t)]).sqrt() * grad
            x = sched.step(eps_hat, t, x).prev_sample
        else:
            with torch.no_grad():
                eps = model(x, t_b, labels)
            x = sched.step(eps, t, x).prev_sample
        if i in snap_idx and i != n - 1:
            snapshots.append(x.clone())
    snapshots.append(x.clone())  # final x_0

    # Trim to exactly args.snapshots if we picked too many.
    if len(snapshots) > args.snapshots:
        # uniformly subsample, keeping first and last
        idx = [round(i * (len(snapshots) - 1) / (args.snapshots - 1))
               for i in range(args.snapshots)]
        snapshots = [snapshots[i] for i in idx]

    grid = make_grid(denormalize(torch.cat(snapshots, dim=0)), nrow=len(snapshots))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, args.out)
    print(f"[denoise] saved {len(snapshots)} snapshots -> {args.out}")


if __name__ == "__main__":
    main()
