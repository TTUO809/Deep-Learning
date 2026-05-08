"""===============================================================
【 Train a conditional DDPM on iCLEVR (Lab 6).

Usage (from repo root):
    python src/train.py --epochs 200 --batch_size 64

Defaults assume the layout:
    DL_LAB6_B11107122_凃岳霖/
        file/{train.json, objects.json}
        iclevr/<images>
        checkpoints/
==============================================================="""

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
    """---------------------------------------------------------------
    【 Argument Parser 】
    
    Parses command-line arguments for training hyperparameters, directory paths, and logging configurations.
    
    Input: Command-line arguments -> Output: Parsed ArgumentNamespace
    ---------------------------------------------------------------"""

    # 1. 建立 ArgumentParser 來管理所有訓練相關的超參數與路徑設定。
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
    """---------------------------------------------------------------
    【 Main Training Routine 】
    
    Executes the full training loop for the Conditional DDPM, including data loading, forward diffusion, noise prediction, optimization with AMP, EMA tracking, and checkpointing.
    
    Input: None -> Output: None (Saves model checkpoints to disk)
    ---------------------------------------------------------------"""
    
    # 1. 系統初始化：讀取參數、設定全域亂數種子以確保實驗可重現性，並建立儲存權重的資料夾。
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # 2. 設備配置：自動偵測並使用 CUDA GPU，若無則降級為 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Weights & Biases (WandB) 實驗追蹤設定。
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

    # 4. 數據加載 (Data Loading)。
    #    回傳的 DataLoader 每次迭代會提供 imgs: (B, 3, H, W) 和 labels: (B, num_classes)。
    loader = build_train_loader(
        img_dir=args.img_dir,
        json_path=args.train_json,
        obj_json_path=args.obj_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"[data] train samples = {len(loader.dataset)}, steps/epoch = {len(loader)}")

    # 5. 建構條件式 U-Net 模型並移至 GPU。
    model = ConditionalUNet().to(device)
    n_params = count_parameters(model)
    print(f"[model] trainable parameters = {n_params:,}")
    if use_wandb:
        wandb.config.update({"n_params": n_params})

    # 6. 建構擴散排程器 (Diffusion Scheduler)。
    #    物理意義：定義了隨時間 t 推進，如何將純淨圖片逐步添加高斯雜訊破壞成純雜訊。
    if args.beta_schedule == "linear":
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            beta_schedule="linear",
            beta_start=1e-4,
            beta_end=0.02,
        )
    else:
        # 使用 Cosine 排程 (squaredcos_cap_v2)，能讓破壞過程更平滑，改善生成品質。
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
        )

    # 7. 優化器與學習率排程。
    #    使用 AdamW 結合權重衰減 (Weight Decay) 進行正則化；使用 CosineAnnealingLR 讓學習率隨 Epoch 平滑下降。
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 8. 指數移動平均 (EMA) 與混合精度訓練 (AMP) 設定。
    #    EMA 物理意義：平滑模型權重，減少訓練過程中的劇烈震盪，通常 EMA 模型生成的圖片會更穩定、品質更好。
    #    GradScaler 物理意義：避免在使用 float16 (AMP) 訓練時發生梯度下溢 (Gradient Underflow)。
    ema = EMA(model, decay=args.ema_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # 9. 斷點續傳 (Checkpoint Resuming) 機制。
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

    # 10. 訓練主迴圈 (Main Training Loop)
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
            # 10.1. 將數據移至 GPU，non_blocking=True 允許數據傳輸與計算重疊，提升效能。
            #       imgs: (B, 3, H, W), labels: (B, num_classes)
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            B = imgs.size(0)
            
            # 10.2. 隨機採樣時間步 t。
            #       為 batch 中的每張圖片隨機選取一個時間步 t，形狀為 (B,)。
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,),
                              device=device, dtype=torch.long)
                              
            # 10.3. 前向擴散 (Forward Diffusion)：製造真實雜訊。
            #       生成與圖片同形狀的純高斯雜訊 noise: (B, 3, H, W)。
            noise = torch.randn_like(imgs)
            
            # 10.4. 根據排程器與時間步 t，將 noise 按照特定比例混入原圖 imgs 中，得到加噪後的圖片 noisy: (B, 3, H, W)。
            noisy = noise_scheduler.add_noise(imgs, noise, t)

            # 10.5. 混合精度前向傳播 (AMP Forward Pass)。
            with torch.amp.autocast("cuda", enabled=args.amp):
                # 10.6. 雜訊預測：模型接收 (破壞後的圖片, 時間步 t, 條件標籤)，預測出被加進去的雜訊。
                #       pred shape: (B, 3, H, W)。
                pred = model(noisy, t, labels)
                
                # 10.7. 損失計算：計算模型預測的雜訊 (pred) 與我們實際加入的真實雜訊 (noise) 之間的均方誤差 (MSE)。
                loss = F.mse_loss(pred, noise)

            # 10.8. 梯度清零與反向傳播。
            #       set_to_none=True 釋放記憶體比設為 0 更有效率。
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            
            # 10.9. 梯度裁剪 (Gradient Clipping)。
            #       先將梯度反縮放回來 (unscale_)，再裁減梯度避免梯度爆炸 (Exploding Gradients)，確保訓練穩定。
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # 10.10. 參數更新與 Scaler 更新。
            scaler.step(optimizer)
            scaler.update()

            # 10.11. 更新 EMA 權重。
            #        只有在達到指定的 warmup 步數 (ema_start_step) 後才開始累積平滑權重。
            global_step += 1
            if global_step >= args.ema_start_step:
                ema.update(model)
                
            # 10.12. 記錄進度與 Loss 狀態。
            step_loss = loss.item()
            running += step_loss * B
            n += B
            pbar.set_postfix(loss=f"{running / n:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            
            if use_wandb and global_step % 50 == 0:
                wandb.log({"train/step_loss": step_loss,
                           "train/lr": optimizer.param_groups[0]["lr"]},
                          step=global_step)

        # 10.13. 每個 Epoch 結束後，推進學習率排程器。
        lr_sched.step()
        
        # 10.14. 結算並寫入當前 Epoch 的平均指標。
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

        # 11. 模型檢查點儲存 (Checkpointing)。
        #     定期將模型、EMA、優化器狀態打包為字典，方便後續推論或斷點續傳。
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
            
            # 同時更新一個名稱固定為 "last.pt" 的檔案，方便快速讀取最新權重。
            last_path = os.path.join(args.ckpt_dir, "last.pt")
            torch.save(ckpt_payload, last_path)
            print(f"[save] {ckpt_path}")

    # 12. 結束 WandB 追蹤進程。
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()