import os
import argparse
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

from oxford_pet import get_oxford_pet_dataloader
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from utils import cal_dice_score

def evaluate(eval_args):
    '''
    Args:
        eval_args (argparse.Namespace): 包含所有評估參數。
    '''
    print(f"=============== Start Evaluate with 【 {eval_args.model} 】 ===============")
    print(f"Args:\n - Batch Size: {eval_args.batch_size}\n - Threshold: {eval_args.threshold}")
    print(f"Directory:\n - Model: {eval_args.save_model_dir}")
    print("="*60)

    # 設置設備 (GPU 或 CPU)。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 獲取 DataLoader。
    print(f"Loading Validation Data...(Batch size: {eval_args.batch_size}) \n[From {os.path.join(eval_args.data_dir, 'val.txt')}]")
    val_loader   = get_oxford_pet_dataloader(root_dir=eval_args.data_dir, split='val'  , batch_size=eval_args.batch_size)

    # 初始化模型並移動到 device。
    if eval_args.model == 'unet':
        model = UNet(in_channels=3, out_channels=1).to(device)
    elif eval_args.model == 'res_unet':
        model = ResNet34UNet(in_channels=3, out_channels=1).to(device)
    else:
        raise ValueError(f"Error: Unsupported model type '{eval_args.model}', please choose 'unet' or 'res_unet'.")

    # 自動組裝模型路徑
    model_path = os.path.join(eval_args.save_model_dir, f"{eval_args.model}_best.pth")

    # 嘗試加載模型權重。
    print(f"Loading model weights from: {model_path}")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Error: Could not find model file {model_path}! Please run train.py to train the model first.")
        return

    # 開始 評估 模型。
    model.eval()            # 切換到評估模式。
    val_dice = 0.0          # 累積 Dice Score。

    # 顯示 評估 進度條。
    print("="*60)
    val_pbar = tqdm(val_loader, desc="Evaluating")
    with torch.no_grad():  # 在評估階段不需要計算梯度。
        for images, masks in val_pbar:
            images, masks = images.to(device), masks.to(device)

            # 前向傳播。
            outputs = model(images)

            # 補償 UNet 的輸出圖像大小問題，對 GT_masks 進行中心裁剪以匹配輸出圖像的大小。
            if eval_args.model == 'unet':
                _, _, H, W = outputs.shape
                masks_resized = TF.center_crop(masks, output_size=(H, W))
            else:
                masks_resized = masks

            # 計算 Dice Score。
            dice = cal_dice_score(outputs, masks_resized, threshold=eval_args.threshold)

            # 累積 Dice Score。
            val_dice += dice

            # 更新 評估 進度條。顯示當前的平均 Dice Score。
            val_pbar.set_postfix(Dice=f"{dice:.4f}")
    
    # 計算當前 epoch 的平均 Dice Score。
    avg_val_dice = val_dice / len(val_loader)
    print("\n" + "="*60)
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

    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'res_unet'], help='Model name for evaluation (default: unet)')
    
    batch_size_default = 16
    threshold_default = 0.5
    parser.add_argument('--batch_size', type=int, default=batch_size_default, help=f'Batch size for evaluation (default: {batch_size_default})')
    parser.add_argument('--threshold', type=float, default=threshold_default, help=f'Threshold for converting probabilities to binary masks (default: {threshold_default})')
    
    eval_args = parser.parse_args()
    
    evaluate(eval_args)