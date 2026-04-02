import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from oxford_pet import get_oxford_pet_dataloader
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from utils import set_seed, cal_dice_score, resolve_model_config

def build_thresholds(start, end, step):
    '''
    Args:
        start (float): 起始 threshold 值。
        end (float): 結束 threshold 值。
        step (float): threshold 之間的步長。
    Returns:
        List[float]: 生成的 threshold 列表。
    '''
    if step <= 0:
        raise ValueError("threshold step must be > 0")
    if start > end:
        raise ValueError("threshold start must be <= end")
    if start < 0 or end > 1:
        raise ValueError("threshold range must be within [0, 1]")

    num_steps = int(round((end - start) / step)) + 1
    thresholds = [round(start + i * step, 6) for i in range(num_steps)]

    # 避免浮點誤差導致最後一個點遺失。
    if thresholds[-1] < end - 1e-9:
        thresholds.append(round(end, 6))

    return thresholds

def evaluate_with_thresholds(model, val_loader, device, model_name, thresholds):
    '''
    Args:
        model (torch.nn.Module): 已加載權重的模型。
        val_loader (torch.utils.data.DataLoader): 驗證集的 DataLoader。
        device (torch.device): 設備 (GPU 或 CPU)。
        model_name (str): 模型名稱，用於決定是否需要 padding。
        thresholds (List[float]): 要評估的 threshold 列表。
    Returns:
        Dice_TH[float, float]: 每個 threshold 對應的平均 Dice Score。
    '''
    # (Dictionary) 累加每個 threshold 的 Dice 分數總和。
    dice_sums = {th: 0.0 for th in thresholds}

    model.eval()
    val_pbar = tqdm(val_loader, desc="Evaluating")
    with torch.no_grad():
        for images, masks in val_pbar:
            images, masks = images.to(device), masks.to(device)

            if model_name == 'unet':
                pad = 92
                images = F.pad(images, (pad, pad, pad, pad), mode='reflect')

            outputs = model(images)

            # 算出每個 threshold 對應的 Dice Score，並累加到 dice_sums 中。
            for th in thresholds:
                dice = cal_dice_score(outputs, masks, threshold=th)
                dice_sums[th] += dice

            # 如果只掃描一個 threshold，就在進度條上顯示當前的 Dice 分數。
            if len(thresholds) == 1:
                val_pbar.set_postfix(Dice=f"{dice:.4f}")

    avg_dice = {th: dice_sums[th] / len(val_loader) for th in thresholds}
    return avg_dice

def evaluate(eval_args):
    '''
    Args:
        eval_args (argparse.Namespace): 包含所有評估參數。
    '''
    # 設定 seed 以確保可重現的評估結果
    set_seed(eval_args.seed)
    
    selected_model, model_path = resolve_model_config(eval_args)

    print(f"=============== Start Evaluate with 【 {selected_model} 】 ===============")
    print(f"Args:\n - Batch Size: {eval_args.batch_size}\n - Threshold: {eval_args.threshold}\n - Seed: {eval_args.seed}")
    print(f"\nDirectory:\n - Model: {model_path}")
    print("="*60)

    # 設置設備 (GPU 或 CPU)。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 獲取 DataLoader。
    print(f"Loading Validation Data...(Batch size: {eval_args.batch_size}) \n[From {os.path.join(eval_args.data_dir, 'val.txt')}]")
    # 使用 num_workers=0 確保評估時的完全可重現性（避免多進程的不確定性）
    val_loader   = get_oxford_pet_dataloader(root_dir=eval_args.data_dir, split='val'  , batch_size=eval_args.batch_size, num_workers=0)

    # 初始化模型並移動到 device。
    if selected_model == 'unet':
        model = UNet(in_channels=3, out_channels=1).to(device)
    elif selected_model == 'res_unet':
        model = ResNet34UNet(in_channels=3, out_channels=1).to(device)
    else:
        raise ValueError(f"Error: Unsupported model type '{selected_model}', please choose 'unet' or 'res_unet'.")

    # 嘗試加載模型權重。
    print(f"Loading model weights from: {model_path}")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Error: Could not find model file {model_path}! Please run train.py to train the model first.")
        return

    # 開始評估模型。
    print("="*60)
    if eval_args.auto_threshold:
        thresholds = build_thresholds(eval_args.threshold_start, eval_args.threshold_end, eval_args.threshold_step)
        print(f"Auto threshold scan: start={eval_args.threshold_start}, end={eval_args.threshold_end}, step={eval_args.threshold_step}")
        print(f"Scanning {len(thresholds)} thresholds: {thresholds}")
    else:
        thresholds = [eval_args.threshold]

    avg_dice_map = evaluate_with_thresholds(model, val_loader, device, selected_model, thresholds)

    print("="*60)
    if eval_args.auto_threshold:
        best_threshold, best_dice = max(avg_dice_map.items(), key=lambda x: x[1])
        print("Threshold Scan Results:")
        for th in thresholds:
            print(f" - threshold={th:.4f}: dice={avg_dice_map[th]:.4f}")
        print("="*60)
        print(f"Best Threshold: {best_threshold:.4f}")
        print(f"Best Val Dice Score: {best_dice:.4f}")
    else:
        avg_val_dice = avg_dice_map[eval_args.threshold]
        print(f"Val Set Dice Score: {avg_val_dice:.4f} (Threshold: {eval_args.threshold})")
    print("="*60)

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

    # 定義命令行參數。
    parser = argparse.ArgumentParser(description="Evaluate on Oxford-IIIT Pet Dataset")
    
    data_dir_default = os.path.join(PROJECT_ROOT, 'dataset', 'oxford-iiit-pet')
    save_model_dir_default = os.path.join(PROJECT_ROOT, 'saved_models')
    parser.add_argument('--data_dir', type=str, default=data_dir_default, help=f'Path to the dataset directory (default: {data_dir_default})')
    parser.add_argument('--save_model_dir', type=str, default=save_model_dir_default, help=f'Directory to load the trained model (default: {save_model_dir_default})')
    parser.add_argument('--model_path', type=str, default='', help='Direct path to the model file (overrides --save_model_dir and --model when provided)')

    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'res_unet'], help='Model name for evaluation (default: unet)')
    
    random_seed_default = 42
    batch_size_default = 16
    threshold_default = 0.5
    parser.add_argument('--seed', type=int, default=random_seed_default, help=f'Random seed for reproducibility (default: {random_seed_default})')
    parser.add_argument('--batch_size', type=int, default=batch_size_default, help=f'Batch size for evaluation (default: {batch_size_default})')
    parser.add_argument('--threshold', type=float, default=threshold_default, help=f'Threshold for converting probabilities to binary masks (default: {threshold_default})')
    parser.add_argument('--auto_threshold', action='store_true', help='Auto scan thresholds and report the best one on val set')
    parser.add_argument('--threshold_start', type=float, default=0.3, help='Start threshold for auto scan (default: 0.3)')
    parser.add_argument('--threshold_end', type=float, default=0.7, help='End threshold for auto scan (default: 0.7)')
    parser.add_argument('--threshold_step', type=float, default=0.05, help='Step size for auto threshold scan (default: 0.05)')
    
    eval_args = parser.parse_args()
    
    evaluate(eval_args)