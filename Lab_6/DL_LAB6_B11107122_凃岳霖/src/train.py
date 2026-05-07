"""Train a conditional DDPM on iCLEVR (Lab 6).

Usage (from repo root):
    python src/train.py --epochs 200 --batch_size 64

Defaults assume the layout:
    DL_LAB6_B11107122_凃岳霖/
        file/{train.json, objects.json}
        iclevr/<images>
        checkpoints/
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from diffusers import DDPMScheduler

from dataset import build_train_loader
from model import ConditionalUNet
from utils import EMA, count_parameters, set_seed

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", default=str(ROOT / "iclevr"))
    p.add_argument("--train_json", default=str(ROOT / "file" / "train.json"))
    p.add_argument("--obj_json", default=str(ROOT / "file" / "objects.json"))
    p.add_argument("--ckpt_dir", default=str(ROOT / "checkpoints"))
    p.add_argument("--resume", default="", help="path to a checkpoint .pt to resume from")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--ema_start_step", type=int, default=2000,
                   help="warmup steps before EMA tracking starts")
    p.add_argument("--num_train_timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", choices=["linear", "squaredcos_cap_v2"],
                   default="squaredcos_cap_v2")
    p.add_argument("--save_every", type=int, default=10, help="save checkpoint every N epochs")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="enable mixed precision")
    # wandb
    p.add_argument("--wandb", action="store_true", help="enable wandb logging")
    p.add_argument("--wandb_project", default="DL_Lab6_DDPM")
    p.add_argument("--wandb_name", default="", help="run name (auto if empty)")
    p.add_argument("--wandb_id", default="", help="existing wandb run id to resume (keeps same charts)")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- wandb init ----
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("[wandb] package not found, skipping logging")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or None,
            id=args.wandb_id or None,
            config=vars(args),
            resume="must" if args.wandb_id else "allow",
            save_code=True,
        )

    # ---- Data ----
    loader = build_train_loader(
        img_dir=args.img_dir,
        json_path=args.train_json,
        obj_json_path=args.obj_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"[data] train samples = {len(loader.dataset)}, steps/epoch = {len(loader)}")

    # ---- Model + Scheduler ----
    model = ConditionalUNet().to(device)
    n_params = count_parameters(model)
    print(f"[model] trainable parameters = {n_params:,}")
    if use_wandb:
        wandb.config.update({"n_params": n_params})

    if args.beta_schedule == "linear":
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            beta_schedule="linear",
            beta_start=1e-4,
            beta_end=0.02,
        )
    else:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
        )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = CosineAnnealingLR(optimizer, T_max=args.epochs)
    ema = EMA(model, decay=args.ema_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        optimizer.load_state_dict(ck["opt"])
        lr_sched.load_state_dict(ck["lr_sched"])
        start_epoch = ck.get("epoch", 0) + 1
        global_step = ck.get("global_step", 0)
        print(f"[resume] from {args.resume} (epoch {start_epoch}, step {global_step})")

    # ---- Train loop ----
    log_path = os.path.join(args.ckpt_dir, "train_log.csv")
    if start_epoch == 0:
        with open(log_path, "w") as f:
            f.write("epoch,step,loss,lr,elapsed\n")
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        n = 0
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            B = imgs.size(0)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,),
                              device=device, dtype=torch.long)
            noise = torch.randn_like(imgs)
            noisy = noise_scheduler.add_noise(imgs, noise, t)

            with torch.amp.autocast("cuda", enabled=args.amp):
                pred = model(noisy, t, labels)
                loss = F.mse_loss(pred, noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            if global_step >= args.ema_start_step:
                ema.update(model)
            step_loss = loss.item()
            running += step_loss * B
            n += B
            pbar.set_postfix(loss=f"{running / n:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            if use_wandb and global_step % 50 == 0:
                wandb.log({"train/step_loss": step_loss,
                           "train/lr": optimizer.param_groups[0]["lr"]},
                          step=global_step)

        lr_sched.step()
        avg_loss = running / max(n, 1)
        elapsed = time.time() - t0
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{global_step},{avg_loss:.6f},"
                    f"{optimizer.param_groups[0]['lr']:.6e},{elapsed:.1f}\n")
        print(f"[epoch {epoch+1}] avg_loss={avg_loss:.4f}  elapsed={elapsed/60:.1f}min")
        if use_wandb:
            wandb.log({"train/epoch_loss": avg_loss,
                       "train/epoch": epoch + 1,
                       "train/elapsed_min": elapsed / 60},
                      step=global_step)

        # ---- Save checkpoint ----
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_epoch{epoch+1:04d}.pt")
            wandb_run_id = wandb.run.id if use_wandb else ""
            ckpt_payload = {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "opt": optimizer.state_dict(),
                "lr_sched": lr_sched.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "args": vars(args),
                "wandb_run_id": wandb_run_id,
            }
            torch.save(ckpt_payload, ckpt_path)
            # also update a "last" pointer for convenience
            last_path = os.path.join(args.ckpt_dir, "last.pt")
            torch.save(ckpt_payload, last_path)
            print(f"[save] {ckpt_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
