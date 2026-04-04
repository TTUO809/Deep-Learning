import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# LinearLR (Warmup 用), CosineAnnealingLR (主調度器), SequentialLR (組合用)
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from torch.amp import GradScaler, autocast  # 用於混合精度訓練的工具。
from tqdm import tqdm                       # 用於顯示訓練進度條。

from oxford_pet import get_oxford_pet_dataloader
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from utils import set_seed, cal_dice_score, BCEDiceLoss, FocalDiceLoss, detect_optimal_num_workers

def _print_run_config(train_args):
    """
    Args:
        train_args (argparse.Namespace): 包含所有訓練參數。
    Description:
        打印訓練配置，包括模型選擇、訓練參數、優化器設定、學習率調度策略、早停設定以及相關的目錄路徑。
        這有助於在訓練開始前確認所有配置是否正確，並提供清晰的訓練信息輸出。
    """

    print(f"\n=============== Start Train with 【 {train_args.model} 】 ===============")
    print(f"Args:\
          \n - Epochs: {train_args.epochs}\
          \n - Batch Size: {train_args.batch_size}")

    print(f"\nStrategy:")
    if train_args.use_bce:
        print(f"- Loss Function:\
              \n       BCEDiceLoss ({train_args.bce_weight} * BCE + {train_args.dice_weight} * Dice Loss)")
    else:
        print(f"- Loss Function:\
              \n       FocalDiceLoss ({train_args.focal_weight} * Focal Loss + {train_args.dice_weight} * Dice Loss)\
              \n       -> Focal Loss Params: alpha={train_args.focal_alpha}, gamma={train_args.focal_gamma}")

    print(f" - Optimizer:\
          \n       AdamW (LR: {train_args.lr}, WD: {train_args.wd})")
    print(f" - LR Schedule:\
          \n       Warmup ({train_args.warmup_epochs} epochs) + Cosine Annealing (T_max: {train_args.epochs - train_args.warmup_epochs} epochs, eta_min: 1e-6)")
    print(f" - Early Stopping Patience: {train_args.early_stop_patience} epochs")
    print(f" - Mixed Precision (AMP): {not train_args.no_amp}\
          \n - Gradient Clipping: Max Norm = {train_args.grad_clip}")
    
    print(f"\nDirectory:\
            \n - Output Model: {train_args.output_dir}")
    if train_args.ckpt:
        print(f"Resume Checkpoint:\
              \n - Path: {train_args.ckpt}")
    print("="*60)

def _build_dataloaders(train_args):
    """
    Args:
        train_args (argparse.Namespace): 包含所有訓練參數。
    Returns:
        (Tuple[DataLoader, DataLoader]): (train_loader, val_loader)
    Description:
        根據訓練參數中的 batch size 和系統的 CPU 核心數，計算並設置 DataLoader 的 num_workers 以優化數據加載效率。
        使用 get_oxford_pet_dataloader 函數分別創建訓練和驗證的 DataLoader，並返回它們。
    """

    num_workers = detect_optimal_num_workers(train_args.batch_size)['recommended']
    num_workers = 0
    print(f"Loading Train / Validation data...(Batch size: {train_args.batch_size}, num_workers: {num_workers})")
    train_loader = get_oxford_pet_dataloader(
        DATA_DIR=train_args.data_dir,
        split='train',
        batch_size=train_args.batch_size,
        num_workers=num_workers,
    )
    val_loader = get_oxford_pet_dataloader(
        DATA_DIR=train_args.data_dir,
        split='val',
        batch_size=train_args.batch_size,
        num_workers=num_workers,
    )
    return train_loader, val_loader

def _build_model(train_args, device):
    """
    Args:
        train_args (argparse.Namespace): 包含所有訓練參數。
        device (torch.device): 模型將被移動到的設備（GPU 或 CPU）。
    Returns:
        (torch.nn.Module): 已初始化並移動到指定設備的模型。
    Description:
        根據訓練參數中的模型選擇（'unet' 或 'res_unet'），初始化對應的模型類，並將其移動到指定的設備上。
    """
    if train_args.model == 'unet':
        return UNet(in_channels=3, out_channels=1).to(device)
    if train_args.model == 'res_unet':
        return ResNet34UNet(in_channels=3, out_channels=1).to(device)
    raise ValueError(f"Error: Unsupported model type '{train_args.model}', please choose 'unet' or 'res_unet'.")


def _build_loss_function(train_args):
    if train_args.use_bce:
        return BCEDiceLoss(bce_weight=train_args.bce_weight, dice_weight=train_args.dice_weight)
    return FocalDiceLoss(
        focal_weight=train_args.focal_weight,
        dice_weight=train_args.dice_weight,
        alpha=train_args.focal_alpha,
        gamma=train_args.focal_gamma,
    )


def _build_optimizer_scheduler(train_args, model):
    optimizer = optim.AdamW(model.parameters(), lr=train_args.lr, weight_decay=train_args.wd)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=train_args.warmup_epochs)
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_args.epochs - train_args.warmup_epochs,
        eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[train_args.warmup_epochs],
    )
    return optimizer, scheduler


def _load_checkpoint_if_needed(train_args, device, model, optimizer, scheduler, scaler):
    start_epoch = 0
    best_val_dice = 0.0
    best_epoch = 0
    early_stop_counter = 0

    if not train_args.ckpt:
        return start_epoch, best_val_dice, best_epoch, early_stop_counter

    if not os.path.exists(train_args.ckpt):
        print(f"Checkpoint not found: {train_args.ckpt}. Starting from scratch.")
        return start_epoch, best_val_dice, best_epoch, early_stop_counter

    print(f"Resuming training from checkpoint: {train_args.ckpt}")
    ckpt = torch.load(train_args.ckpt, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_val_dice = ckpt.get('best_val_dice', 0.0)
        best_epoch = ckpt.get('best_epoch', 0)
        early_stop_counter = ckpt.get('early_stop_counter', 0)
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch + 1}, best dice so far: {best_val_dice:.4f}")
    else:
        # 相容舊格式（純 model weights）
        model.load_state_dict(ckpt)
        print("Loaded model weights only (old format). Optimizer / scheduler state reset.")

    return start_epoch, best_val_dice, best_epoch, early_stop_counter

def _process_images(images, model_name):
    if model_name == 'unet':
        pad = 92
        images = F.pad(images, (pad, pad, pad, pad), mode='reflect')
    return images

def _run_train_epoch(model, train_loader, optimizer, loss_function, scaler, amp_enabled, device, train_args):
    model.train()
    train_loss = 0.0
    train_dice = 0.0

    train_pbar = tqdm(train_loader, desc="Training")
    for images, masks in train_pbar:
        images, masks = images.to(device), masks.to(device)

        images = _process_images(images, train_args.model)

        optimizer.zero_grad()
        with autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss = loss_function(outputs, masks)
            dice = cal_dice_score(outputs, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_dice += dice
        train_pbar.set_postfix(Loss=f"{loss.item():.4f}", Dice=f"{dice:.4f}")

    avg_train_loss = train_loss / len(train_loader)
    avg_train_dice = train_dice / len(train_loader)
    return avg_train_loss, avg_train_dice


def _run_val_epoch(model, val_loader, amp_enabled, device, train_args):
    model.eval()
    val_dice = 0.0

    val_pbar = tqdm(val_loader, desc="Validation")
    with torch.no_grad():
        for images, masks in val_pbar:
            images, masks = images.to(device), masks.to(device)

            images = _process_images(images, train_args.model)

            with autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(images)
                dice = cal_dice_score(outputs, masks)

            val_dice += dice
            val_pbar.set_postfix(Dice=f"{dice:.4f}")

    return val_dice / len(val_loader)


def _save_best_and_checkpoint(train_args, model, optimizer, scheduler, scaler, epoch, best_val_dice, best_epoch, early_stop_counter):
    best_model_path = os.path.join(train_args.output_dir, f"{train_args.model}_best.pth")
    torch.save(model.state_dict(), best_model_path)

    ckpt_path = os.path.join(train_args.output_dir, f"{train_args.model}_checkpoint.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_dice': best_val_dice,
        'best_epoch': best_epoch,
        'early_stop_counter': early_stop_counter,
    }, ckpt_path)

    print(f"Best model  saved: {best_model_path}  (Dice: {best_val_dice:.4f})")
    print(f"Checkpoint  saved: {ckpt_path}")


def train(train_args):
    """
    Args:
        train_args (argparse.Namespace): 包含所有訓練參數。
    """

    # 啟動隨機種子鎖定，以確保訓練過程的可重現性。
    set_seed(train_args.seed)
    _print_run_config(train_args)

    # 設置設備 (GPU 或 CPU)。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader = _build_dataloaders(train_args)
    model = _build_model(train_args, device)

    # 創建保存模型的目錄。
    os.makedirs(train_args.output_dir, exist_ok=True)

    loss_function = _build_loss_function(train_args)
    optimizer, scheduler = _build_optimizer_scheduler(train_args, model)

    # AMP 混合精度訓練的工具。
    amp_enabled = (not train_args.no_amp) and device.type == 'cuda'
    scaler = GradScaler(device.type, enabled=amp_enabled)

    start_epoch, best_val_dice, best_epoch, early_stop_counter = _load_checkpoint_if_needed(
        train_args,
        device,
        model,
        optimizer,
        scheduler,
        scaler,
    )

    # 訓練循環。
    print("="*60)
    for epoch in range(start_epoch, train_args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch + 1}/{train_args.epochs} (LR: {current_lr:.6e})")

        avg_train_loss, avg_train_dice = _run_train_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            scaler,
            amp_enabled,
            device,
            train_args,
        )
        print(f"-> Train Dice: {avg_train_dice:.4f} | Train Loss: {avg_train_loss:.4f}")

        avg_val_dice = _run_val_epoch(model, val_loader, amp_enabled, device, train_args)
        print(f"-> Val Dice  : {avg_val_dice:.4f}")
        
        scheduler.step() # 更新學習率調度器，根據 epoch 進行調整。
        # scheduler.step(avg_val_dice)    # 更新學習率調度器，根據驗證 Dice Score 進行調整。

        # Early Stopping
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_epoch = epoch + 1
            early_stop_counter = 0
            _save_best_and_checkpoint(
                train_args,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_val_dice,
                best_epoch,
                early_stop_counter,
            )
        else:
            early_stop_counter += 1
            print(f"Early Stopping Counter: {early_stop_counter} / {train_args.early_stop_patience}")
            if early_stop_counter >= train_args.early_stop_patience:
                print(f"\nEarly Stopping !!!")
                break

    # 訓練結束，輸出最佳成績與回合。
    print("\n" + "="*40)
    print(f"Training completed.\n > Best Val Dice Score: {best_val_dice:.4f} at epoch {best_epoch}.")
    print("="*40)


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))                # path-to-/src/
    PROJECT_DIR = os.path.dirname(CURRENT_DIR)                              # path-to-/DL_Lab2_B11107122_凃岳霖/
    DATA_DIR    = os.path.join(PROJECT_DIR, 'dataset', 'oxford-iiit-pet')   # path-to-/DL_Lab2_B11107122_凃岳霖/dataset/oxford-iiit-pet
    OUTPUT_DIR  = os.path.join(PROJECT_DIR, 'saved_models')                 # path-to-/DL_Lab2_B11107122_凃岳霖/saved_models

    # 定義命令行參數。自動將預設值添加到每個參數的說明中，讓使用者在查看幫助信息時能夠清楚地知道每個參數的預設值。
    parser = argparse.ArgumentParser(description="Train on Oxford-IIIT Pet Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 路徑
    parser.add_argument('--data_dir'            , type=str  , default=DATA_DIR  , help='Path to the dataset directory')
    parser.add_argument('--output_dir'          , type=str  , default=OUTPUT_DIR, help='Directory to save trained models')

    # 模型
    parser.add_argument('--model'               , type=str  , default='unet'    , help='Model name for training and saving', choices=['unet', 'res_unet'])
    parser.add_argument('--ckpt'                , type=str  , default=None      , help='Path to a checkpoint to resume training')

    # 可重現性
    parser.add_argument('--seed'                , type=int  , default=42        , help='Random seed for reproducibility')

    # 訓練基本設定
    parser.add_argument('--epochs'              , type=int  , default=20        , help='Number of training epochs')
    parser.add_argument('--batch_size'          , type=int  , default=16        , help='Batch size for training')

    # Optimizer
    parser.add_argument('--lr'                  , type=float, default=5e-4      , help='Learning rate for AdamW optimizer')
    parser.add_argument('--wd'                  , type=float, default=1e-4      , help='Weight decay for AdamW optimizer')

    # Loss function
    parser.add_argument('--use_bce'             , action='store_true'           , help='Use BCEDiceLoss (default: use FocalDiceLoss)')
    parser.add_argument('--bce_weight'          , type=float, default=0.5       , help='Weight for BCE loss (BCEDiceLoss only)')
    parser.add_argument('--focal_weight'        , type=float, default=0.5       , help='Weight for Focal loss (FocalDiceLoss only)')
    parser.add_argument('--dice_weight'         , type=float, default=0.5       , help='Weight for Dice loss')
    parser.add_argument('--focal_alpha'         , type=float, default=0.5       , help='Alpha parameter for Focal Loss')
    parser.add_argument('--focal_gamma'         , type=float, default=2.0       , help='Gamma parameter for Focal Loss')

    # LR Scheduler
    parser.add_argument('--warmup_epochs'       , type=int  , default=5         , help='Number of warmup epochs before Cosine Annealing')

    # 訓練技巧
    parser.add_argument('--no_amp'              , action='store_true'           , help='Disable Automatic Mixed Precision (AMP)')
    parser.add_argument('--grad_clip'           , type=float, default=1.0       , help='Max gradient norm for clipping')
    parser.add_argument('--early_stop_patience' , type=int  , default=5         , help='Patience epochs for early stopping')

    train_args = parser.parse_args()

    '''
    # test dice
    print("Testing Dice Score...")
    dummy_logits = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])  # 模擬模型輸出 (B=1, C=1, H=2, W=2)
    dummy_masks = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])   # 模擬真實遮罩 (B=1, C=1, H=2, W=2)
    dice_score = cal_dice_score(dummy_logits, dummy_masks)
    print(f"Test Dice Score: {dice_score:.4f}")
    '''

    train(train_args)