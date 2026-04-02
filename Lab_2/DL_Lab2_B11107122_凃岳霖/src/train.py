import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torchvision.transforms.functional as TF
# from torch.optim.lr_scheduler import CosineAnnealingLR  # 用於根據 epoch 進行學習率調整的調度器 (餘弦退火)。 
# from torch.optim.lr_scheduler import ReduceLROnPlateau  # 根據驗證損失自動調整學習率的調度器。(過擬合)

# LinearLR (Warmup 用), CosineAnnealingLR (主調度器), SequentialLR (組合用)
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from torch.amp import GradScaler, autocast  # 用於混合精度訓練的工具。
from tqdm import tqdm                       # 用於顯示訓練進度條。

from oxford_pet import get_oxford_pet_dataloader
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from utils import set_seed, cal_dice_score, BCEDiceLoss, FocalDiceLoss

def train(train_args):
    '''
    Args:
        train_args (argparse.Namespace): 包含所有訓練參數。
    '''

    set_seed(train_args.seed) # 啟動隨機種子鎖定

    print(f"\n=============== Start Train with 【 {train_args.model} 】 ===============")
    print(f"Args:\n - Batch Size: {train_args.batch_size}\n - Epochs: {train_args.epochs}")
    
    if train_args.use_focal:
        print(f"\nStrategy:\n - Loss Function: FocalDiceLoss ({train_args.focal_weight} * Focal Loss + {train_args.dice_weight} * Dice Loss)")
        print(f" - Focal Loss Params: alpha={train_args.focal_alpha}, gamma={train_args.focal_gamma}")
    else:
        print(f"\nStrategy:\n - Loss Function: BCEDiceLoss ({train_args.bce_weight} * BCE + {train_args.dice_weight} * Dice Loss)")
    
    print(f" - Optimizer: AdamW (LR: {train_args.learning_rate}, WD: {train_args.weight_decay})")
    print(f" - LR Schedule: Warmup ({train_args.warmup_epochs} epochs) + Cosine Annealing (T_max: {train_args.epochs - train_args.warmup_epochs} epochs, eta_min: 1e-6)")
    print(f" - Early Stopping Patience: {train_args.early_stop_patience} epochs")
    print(f" - Mixed Precision (AMP): {train_args.amp} \n - Gradient Clipping: Max Norm = {train_args.grad_clip}")
    print(f"\nDirectory:\n - Output Model: {train_args.output_model_dir}")
    if train_args.resume:
        print(f"Resume Checkpoint:\n - Path: {train_args.resume}")
    print("="*60)

    # 設置設備 (GPU 或 CPU)。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 獲取 DataLoader。
    print(f"Loading Train / Validation data...(Batch size: {train_args.batch_size}) \n[From {os.path.join(train_args.data_dir, 'train.txt')},\n      {os.path.join(train_args.data_dir, 'val.txt')}]")
    train_loader = get_oxford_pet_dataloader(root_dir=train_args.data_dir, split='train', batch_size=train_args.batch_size)
    val_loader   = get_oxford_pet_dataloader(root_dir=train_args.data_dir, split='val'  , batch_size=train_args.batch_size)

    # 初始化模型並移動到 device。
    if train_args.model == 'unet':
        model = UNet(in_channels=3, out_channels=1).to(device)
    elif train_args.model == 'res_unet':
        model = ResNet34UNet(in_channels=3, out_channels=1).to(device)
    else:
        raise ValueError(f"Error: Unsupported model type '{train_args.model}', please choose 'unet' or 'res_unet'.")

    # 斷點續訓練：如果指定了 resume 參數，嘗試從該路徑加載模型權重。
    if train_args.resume:
        if os.path.exists(train_args.resume):
            print(f"Resuming training from checkpoint: {train_args.resume}")
            model.load_state_dict(torch.load(train_args.resume, map_location=device, weights_only=True))
            print("Checkpoint loaded successfully. Continuing training...")
        else:
            print(f"Checkpoint not found at {train_args.resume}. Starting training from scratch.")

    # 創建保存模型的目錄。
    os.makedirs(train_args.output_model_dir, exist_ok=True)

    # 定義 loss function。
    if train_args.use_focal:
        loss_function = FocalDiceLoss(focal_weight=train_args.focal_weight, dice_weight=train_args.dice_weight, alpha=train_args.focal_alpha, gamma=train_args.focal_gamma)   # 結合 Focal Loss 和 Dice Loss 的自定義損失函數。
    else:
        loss_function = BCEDiceLoss(bce_weight=train_args.bce_weight, dice_weight=train_args.dice_weight)       # 結合 BCE Loss 和 Dice Loss 的自定義損失函數。

    # 定義 optimizer。
    optimizer = optim.AdamW(model.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)  # AdamW 優化器。

    # Warmup + CosineAnnealing 的 LR Scheduler。
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=train_args.warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=train_args.epochs - train_args.warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[train_args.warmup_epochs])
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=train_args.lr_patience, factor=train_args.lr_factor)  # 根據驗證損失自動調整學習率的調度器。

    # AMP 混合精度訓練的工具。
    scaler = GradScaler('cuda')

    # 用來記錄最佳的成績與回合，還有早停的計數器。
    best_val_dice = 0.0 
    best_epoch = 0
    early_stopping_counter = 0

    # 訓練循環。
    print("="*60)
    for epoch in range(train_args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch + 1}/{train_args.epochs} (LR: {current_lr:.6e})")

        # === 訓練階段 ===。
        model.train()       # 設置模型為訓練模式。
        train_loss = 0.0    # 累積訓練 Loss。
        train_dice = 0.0    # 累積訓練 Dice Score。

        # 顯示 訓練 進度條。
        train_pbar = tqdm(train_loader, desc="Training")
        for images, masks in train_pbar:
            images, masks = images.to(device), masks.to(device)

            if train_args.model == 'unet':
                pad = 92
                images = F.pad(images, (pad, pad, pad, pad), mode='reflect') 

            optimizer.zero_grad()   # 清除之前的梯度。

            with autocast('cuda'):  # 混合精度訓練的上下文管理器。
                # 前向傳播。
                outputs = model(images)

                # 補償 UNet 的輸出圖像大小問題，對 GT_masks 進行中心裁剪以匹配輸出圖像的大小。
                # if train_args.model == 'unet':
                #     _, _, H, W = outputs.shape
                #     masks_resized = TF.center_crop(masks, output_size=(H, W))
                # else:
                #     masks_resized = masks
            
                # 計算損失與 Dice Score。
                loss = loss_function(outputs, masks)
                dice = cal_dice_score(outputs, masks)

            # AMP 反向傳播。
            scaler.scale(loss).backward()
            # 在進行梯度裁剪之前，先取消縮放。
            scaler.unscale_(optimizer) 
            # 梯度裁剪，防止梯度爆炸。
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_args.grad_clip)  # 梯度裁剪，防止梯度爆炸。

            # AMP 更新權重參數。
            scaler.step(optimizer)
            scaler.update()

            # 累積 Loss 和 Dice Score。
            train_loss += loss.item()
            train_dice += dice

            # 更新 訓練 進度條。顯示當前的平均 Loss 和 Dice Score。
            train_pbar.set_postfix(Loss=f"{loss.item():.4f}", Dice=f"{dice:.4f}")

        # 計算當前 epoch 的平均 Loss 和 Dice Score。
        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        print(f"-> Train Dice: {avg_train_dice:.4f} | Train Loss: {avg_train_loss:.4f}")

        # === 驗證階段 ===。
        model.eval()    # 設置模型為評估模式。
        val_dice = 0.0  # 累積驗證 Dice Score。

        # 顯示 驗證 進度條。
        val_pbar = tqdm(val_loader, desc="Validation")
        with torch.no_grad():  # 在驗證階段不需要計算梯度。
            for images, masks in val_pbar:
                images, masks = images.to(device), masks.to(device)

                if train_args.model == 'unet':
                    pad = 92
                    images = F.pad(images, (pad, pad, pad, pad), mode='reflect') 

                with autocast('cuda'):  # 混合精度訓練的上下文管理器。
                    # 前向傳播。
                    outputs = model(images)

                    # # 補償 UNet 的輸出圖像大小問題，對 GT_masks 進行中心裁剪以匹配輸出圖像的大小。
                    # if train_args.model == 'unet':
                    #     _, _, H, W = outputs.shape
                    #     masks_resized = TF.center_crop(masks, output_size=(H, W))
                    # else:
                    #     masks_resized = masks

                    # 計算 Dice Score。
                    dice = cal_dice_score(outputs, masks)

                # 累積 Dice Score。
                val_dice += dice

                # 更新 驗證 進度條。顯示當前的平均 Dice Score。
                val_pbar.set_postfix(Dice=f"{dice:.4f}")

        # 計算當前 epoch 的平均 Dice Score。
        avg_val_dice = val_dice / len(val_loader)
        print(f"-> Val Dice  : {avg_val_dice:.4f}")
        
        scheduler.step() # 更新學習率調度器，根據 epoch 進行調整。
        # scheduler.step(avg_val_dice)    # 更新學習率調度器，根據驗證 Dice Score 進行調整。

        # Early Stopping
        if avg_val_dice > best_val_dice:
            # 保存最佳模型。
            best_val_dice = avg_val_dice
            best_epoch = epoch + 1
            early_stop_counter = 0
            save_path = os.path.join(train_args.output_model_dir, f"{train_args.model}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with Dice Score: {best_val_dice:.4f}. Saved at: {save_path}")
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
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))    # src/
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                 # DL_Lab2_B11107122_凃岳霖/

    # 定義命令行參數。
    parser = argparse.ArgumentParser(description="Train on Oxford-IIIT Pet Dataset")

    data_dir_default = os.path.join(PROJECT_ROOT, 'dataset', 'oxford-iiit-pet')
    output_model_dir_default = os.path.join(PROJECT_ROOT, 'saved_models')
    parser.add_argument('--data_dir', type=str, default=data_dir_default, help=f'Path to the dataset directory (default: {data_dir_default})')
    parser.add_argument('--output_model_dir', type=str, default=output_model_dir_default, help=f'Directory to save trained models (default: {output_model_dir_default})')

    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'res_unet'], help='Model name for training and saving (default: unet)')
    
    parser.add_argument('--resume', type=str, default=None, help='Path to a checkpoint to resume training from (default: None)')

    random_seed_default = 42
    parser.add_argument('--seed', type=int, default=random_seed_default, help=f'Random seed for reproducibility (default: {random_seed_default})')

    epochs_default = 20
    batch_size_default = 16

    learning_rate_default = 5e-4
    weight_decay_default = 1e-4
    bce_weight_default = 0.5
    focal_weight_default = 0.5
    dice_weight_default = 0.5
    # lr_patience_default = 5
    # lr_factor_default = 0.5
    early_stop_patience_default = 5
    grad_clip_default = 1.0
    amp_default = True
    focal_default = True
    focal_alpha_default = 0.5
    focal_gamma_default = 2.0
    warmup_epochs_default = 5

    parser.add_argument('--epochs', type=int, default=epochs_default, help=f'Number of training epochs (default: {epochs_default})')
    parser.add_argument('--batch_size', type=int, default=batch_size_default, help=f'Batch size for training (default: {batch_size_default})')
    
    parser.add_argument('--learning_rate', type=float, default=learning_rate_default, help=f'Learning rate for optimizer (default: {learning_rate_default})')
    parser.add_argument('--weight_decay', type=float, default=weight_decay_default, help=f'Weight decay for optimizer (default: {weight_decay_default})')
    parser.add_argument('--bce_weight', type=float, default=bce_weight_default, help=f'Weight for BCE loss (default: {bce_weight_default})')
    parser.add_argument('--focal_weight', type=float, default=focal_weight_default, help=f'Weight for Focal loss (default: {focal_weight_default})')
    parser.add_argument('--dice_weight', type=float, default=dice_weight_default, help=f'Weight for Dice loss (default: {dice_weight_default})')
    # parser.add_argument('--lr_patience', type=int, default=lr_patience_default, help=f'Patience for LR scheduler (default: {lr_patience_default})')
    # parser.add_argument('--lr_factor', type=float, default=lr_factor_default, help=f'Factor for LR scheduler (default: {lr_factor_default})')
    parser.add_argument('--early_stop_patience', type=int, default=early_stop_patience_default, help=f'Patience for early stopping (default: {early_stop_patience_default})')
    parser.add_argument('--grad_clip', type=float, default=grad_clip_default, help=f'Max gradient norm for clipping (default: {grad_clip_default})')
    parser.add_argument('--amp', type=bool, default=amp_default, help=f'Use Automatic Mixed Precision (default: {amp_default})')
    parser.add_argument('--use_focal', default=focal_default, help=f'Use Focal Loss instead of BCE Loss (default: {focal_default})')
    parser.add_argument('--focal_alpha', type=float, default=focal_alpha_default, help=f'Alpha parameter for Focal Loss (default: {focal_alpha_default})')
    parser.add_argument('--focal_gamma', type=float, default=focal_gamma_default, help=f'Gamma parameter for Focal Loss (default: {focal_gamma_default})')
    parser.add_argument('--warmup_epochs', type=int, default=warmup_epochs_default, help=f'Number of warmup epochs for learning rate scheduling (default: {warmup_epochs_default})')

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