import os
import argparse
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
from utils import set_seed, cal_dice_score, BCEDiceLoss, FocalDiceLoss

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
          \n - Batch Size: {train_args.batch_size}\
          \n - Num Workers: {train_args.num_workers}\
          \n - Seed: {train_args.seed}")

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
        根據訓練參數中的資料目錄、批次大小和工作線程數，使用 get_oxford_pet_dataloader 函數分別構建訓練和驗證的 DataLoader。
        這些 DataLoader 將用於後續的訓練和驗證過程中，提供批次化的數據輸入。
    """

    print(f"Loading Train / Validation data...")
    train_loader = get_oxford_pet_dataloader(
        DATA_DIR=train_args.data_dir,
        split='train',
        batch_size=train_args.batch_size,
        num_workers=train_args.num_workers,
    )
    val_loader = get_oxford_pet_dataloader(
        DATA_DIR=train_args.data_dir,
        split='val',
        batch_size=train_args.batch_size,
        num_workers=train_args.num_workers,
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

def _build_criterion(train_args):
    """
    Args:
        train_args (argparse.Namespace): 包含所有訓練參數。
    Returns:
        (torch.nn.Module): 根據訓練參數選擇的損失函數實例。
    Description:
        根據訓練參數中的 use_bce 標誌，選擇使用 BCEDiceLoss 或 FocalDiceLoss 作為損失函數。
    """

    if train_args.use_bce:
        return BCEDiceLoss(bce_weight=train_args.bce_weight, dice_weight=train_args.dice_weight)
    return FocalDiceLoss(
        focal_weight=train_args.focal_weight,
        dice_weight=train_args.dice_weight,
        alpha=train_args.focal_alpha,
        gamma=train_args.focal_gamma,
    )

def _build_optimizer_scheduler(train_args, model):
    """
    Args:
        train_args (argparse.Namespace): 包含所有訓練參數。
        model (torch.nn.Module): 模型參數將被優化器更新。
    Returns:
        (Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]): 優化器和學習率調度器的實例。
    Description:
        使用 AdamW 優化器，並根據訓練參數中的學習率和權重衰減進行配置。
        同時構建一個組合學習率調度器，包含線性暖啟動（ LinearLR ）和餘弦退火（ CosineAnnealingLR ），以實現訓練過程中的學習率調整。
    """

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_args.lr,
        weight_decay=train_args.wd
    )
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        total_iters=train_args.warmup_epochs
    )
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
    """
    Args:
        train_args (argparse.Namespace): 包含所有訓練參數。
        device (torch.device): 模型和優化器狀態將被移動到的設備。
        model (torch.nn.Module): 模型實例，將從 checkpoint 恢復權重。
        optimizer (torch.optim.Optimizer): 優化器實例，將從 checkpoint 恢復狀態。
        scheduler (torch.optim.lr_scheduler._LRScheduler): 學習率調度器實例，將從 checkpoint 恢復狀態。
        scaler (torch.amp.GradScaler): AMP 梯度縮放器實例，將從 checkpoint 恢復狀態。
    Returns:
        (Tuple[int, float, int, int]): (start_epoch, best_val_dice, best_epoch, early_stop_counter)。
    Description:
        如果在訓練參數中指定了 checkpoint 路徑，則嘗試從該 checkpoint 恢復訓練狀態。
        包括模型權重、優化器狀態、學習率調度器狀態、AMP 梯度縮放器狀態，以及訓練回合數、最佳驗證 Dice 分數、最佳回合數和早停計數器。
        如果 checkpoint 不存在或格式不正確，則從頭開始訓練，並返回初始狀態。
    """
    
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
        start_epoch = ckpt.get('epoch', 0)                  # get('epoch') 可能不存在，默認為 0
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
    """
    Args:
        images (torch.Tensor): 原始輸入圖像張量，形狀為 (B, C, H, W)。
        model_name (str): 模型名稱，用於決定是否需要對圖像進行額外的填充處理。
    Returns:
        (torch.Tensor): 處理後的圖像張量，根據模型需求可能已經進行了填充。
    Description:
        根據模型名稱，對輸入圖像進行必要的處理。
        目前的實現是針對 UNet 模型進行反射填充，以確保輸入圖像的尺寸符合模型的要求。
    """

    if model_name == 'unet':
        pad = 92
        images = F.pad(images, (pad, pad, pad, pad), mode='reflect')
    return images

def _run_train_epoch(model, train_loader, optimizer, criterion, scaler, amp_enabled, device, train_args):
    """
    Args:
        model (torch.nn.Module): 要訓練的模型。
        train_loader (DataLoader): 用於訓練的數據加載器。
        optimizer (torch.optim.Optimizer): 用於更新模型權重的優化器。
        criterion (torch.nn.Module): 用於計算損失的損失函數。
        scaler (torch.amp.GradScaler): 用於混合精度訓練的梯度縮放器。
        amp_enabled (bool): 是否啟用自動混合精度。
        device (torch.device): 模型和數據將被移動到的設備。
        train_args (argparse.Namespace): 包含所有訓練參數。
    Returns:
        (Tuple[float, float]): (avg_train_loss, avg_train_dice)，分別為平均訓練損失和平均訓練 Dice 分數。
    Description:
        執行一個訓練回合，對模型進行訓練並計算平均訓練損失和平均訓練 Dice 分數。
        包括前向傳播、損失計算、反向傳播、梯度縮放和優化器更新等步驟。
        同時使用 tqdm 顯示訓練進度條和當前的損失與 Dice 分數。
    """
    
    model.train()
    train_loss = 0.0
    train_dice = 0.0

    train_pbar = tqdm(train_loader, desc="Training")
    for images, masks in train_pbar:
        images, masks = images.to(device), masks.to(device)

        images = _process_images(images, train_args.model)

        optimizer.zero_grad()   # 清除之前的梯度，以免累積。

        # 前向傳播和損失計算在 autocast 上下文中進行，以利用混合精度訓練的性能優勢。
        with autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, masks)
            dice = cal_dice_score(outputs, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_args.grad_clip)   # 梯度裁剪，防止梯度爆炸。
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_dice += dice
        train_pbar.set_postfix(Loss=f"{loss.item():.4f}", Dice=f"{dice:.4f}")

    avg_train_loss = train_loss / len(train_loader)
    avg_train_dice = train_dice / len(train_loader)
    return avg_train_loss, avg_train_dice

def _run_val_epoch(model, val_loader, amp_enabled, device, train_args):
    """
    Args:
        model (torch.nn.Module): 要驗證的模型。
        val_loader (DataLoader): 用於驗證的數據加載器。
        amp_enabled (bool): 是否啟用自動混合精度。
        device (torch.device): 模型和數據將被移動到的設備。
        train_args (argparse.Namespace): 包含所有訓練參數。
    Returns:
        (float): 平均驗證 Dice 分數。
    Description:
        執行一個驗證回合，對模型進行驗證並計算平均驗證 Dice 分數。
        包括前向傳播和 Dice 分數計算等步驟。
        同時使用 tqdm 顯示驗證進度條和當前的 Dice 分數。
    """

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
    """
    Args:
        train_args (argparse.Namespace): 包含所有訓練參數。
        model (torch.nn.Module): 當前的模型實例。
        optimizer (torch.optim.Optimizer): 當前的優化器實例。
        scheduler (torch.optim.lr_scheduler._LRScheduler): 當前的學習率調度器實例。
        scaler (torch.amp.GradScaler): 當前的 AMP 梯度縮放器實例。
        epoch (int): 當前的訓練回合數。
        best_val_dice (float): 當前最佳的驗證 Dice 分數。
        best_epoch (int): 當前最佳的訓練回合數。
        early_stop_counter (int): 當前的早停計數器值。
    Description:
        保存當前的最佳模型權重到指定的路徑。
        並同時保存一個包含模型權重、優化器狀態、學習率調度器狀態、AMP 梯度縮放器狀態以及訓練回合數、最佳驗證 Dice 分數、最佳回合數和早停計數器的 checkpoint。
        這樣可以在訓練過程中保留最佳模型的權重，同時也可以從 checkpoint 恢復訓練狀態。
    """

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

    # 構建數據加載器。
    train_loader, val_loader = _build_dataloaders(train_args)

    # 構建模型，並將其移動到指定設備。
    model = _build_model(train_args, device)

    # 創建保存模型的目錄。
    os.makedirs(train_args.output_dir, exist_ok=True)

    # 構建損失函數。
    criterion = _build_criterion(train_args)

    # 構建優化器和學習率調度器。
    optimizer, scheduler = _build_optimizer_scheduler(train_args, model)

    # AMP 混合精度訓練的工具。
    amp_enabled = (not train_args.no_amp) and device.type == 'cuda'
    scaler = GradScaler(device.type, enabled=amp_enabled)

    # 如果指定了 checkpoint，則從 checkpoint 恢復訓練狀態，包括模型權重、優化器狀態、學習率調度器狀態、AMP 梯度縮放器狀態，以及訓練回合數、最佳驗證 Dice 分數、最佳回合數和早停計數器。
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
            criterion,
            scaler,
            amp_enabled,
            device,
            train_args,
        )
        print(f"-> Train Dice: {avg_train_dice:.4f} | Train Loss: {avg_train_loss:.4f}")

        avg_val_dice = _run_val_epoch(
            model,
            val_loader,
            amp_enabled,
            device,
            train_args
        )
        print(f"-> Val Dice  : {avg_val_dice:.4f}")
        
        scheduler.step() # 更新學習率調度器，根據 epoch 進行調整。

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
    print("\n" + "="*60)
    print(f"Training completed.\n > Best Val Dice Score: {best_val_dice:.4f} at epoch {best_epoch}.")
    print("="*60)

def get_train_args():
    """
    Returns:
        argparse.Namespace: 包含所有訓練參數的命名空間對象。
    Description:
        定義並解析命令行參數，這些參數用於配置訓練過程中的各種選項，包括數據路徑、模型選擇、訓練超參數、優化器設定、學習率調度策略以及訓練技巧等。
    """
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
    parser.add_argument('--num_workers'         , type=int  , default=4         , help='Number of worker processes for data loading')

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
    return train_args

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))                # path-to-/src/
    PROJECT_DIR = os.path.dirname(CURRENT_DIR)                              # path-to-/DL_Lab2_B11107122_凃岳霖/
    DATA_DIR    = os.path.join(PROJECT_DIR, 'dataset', 'oxford-iiit-pet')   # path-to-/DL_Lab2_B11107122_凃岳霖/dataset/oxford-iiit-pet
    OUTPUT_DIR  = os.path.join(PROJECT_DIR, 'saved_models')                 # path-to-/DL_Lab2_B11107122_凃岳霖/saved_models

    train_args = get_train_args()

    train(train_args)