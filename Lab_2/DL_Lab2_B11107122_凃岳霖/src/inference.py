import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

from oxford_pet import get_oxford_pet_dataloader
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet

def rle_encode(mask):
    '''
    Args:
        mask (numpy.ndarray): 二值化的遮罩圖像，形狀為 (H, W)，值為 0 或 1。
    Returns:
        (str): RLE 編碼的字符串，格式為 "start length start length ..."。
    '''
    pixels = mask.flatten(order='F')  # 按列優先（Fortran 順序 = 先由上到下、再由左到右）展平圖像，以符合 RLE 的要求。

    # 1. 在前後各補一個 0，方便精準捕捉從 0 變 1，或從 1 變 0 的瞬間
    pixels = np.concatenate([[0], pixels, [0]])
    
    # 2. 讓陣列錯位 1 格做比較，不一樣的地方就是 邊界 (Run 的起點或終點)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    
    # 3. 終點減去起點，瞬間算出所有長度 (length)
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)

def inference(infer_args):
    '''
    Args:
        infer_args (argparse.Namespace): 包含所有推理參數。
    '''

    print(f"=============== Start Inference with 【 {infer_args.model} 】 ===============")
    print(f"Args:\n - Batch Size: {infer_args.batch_size}\n - TTA: {infer_args.tta}")
    print(f"Directory:\n - Model: {infer_args.save_model_dir}\n - Output CSV: {infer_args.output_csv_dir}")
    print("="*60)
    
    # 設置設備 (GPU 或 CPU)。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 獲取 DataLoader。
    print(f"Loading Test Data...(Batch size: {infer_args.batch_size}) \n[From {os.path.join(infer_args.data_dir, f'test_{infer_args.model}.txt')}]")
    test_loader  = get_oxford_pet_dataloader(root_dir=infer_args.data_dir, split=f'test_{infer_args.model}', batch_size=infer_args.batch_size)
    all_image_names = test_loader.dataset.image_names   # 獲取測試集的所有圖像名稱列表，For image_id。
    image_idx = 0                                       # 追蹤當前處理的圖像。

    # 初始化模型並移動到 device。
    if infer_args.model == 'unet':
        model = UNet(in_channels=3, out_channels=1).to(device)
    elif infer_args.model == 'res_unet':
        model = ResNet34UNet(in_channels=3, out_channels=1).to(device)
    else:
        raise ValueError(f"Unsupported model: {infer_args.model}")

    # 自動組裝模型路徑
    model_path = os.path.join(infer_args.save_model_dir, f"{infer_args.model}_best.pth")

    # 嘗試加載模型權重。
    print(f"Loading model weights from: {model_path}")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Error: Could not find model file {model_path}! Please run train.py to train the model first.")
        return

    # 開始 推理 模型。
    model.eval()    # 切換到評估模式。
    results = []    # 用來存儲推理結果的列表，每個元素是一個字典，包含 'image_id' 和 'rle_mask' 兩個鍵。

    # 定義一個內部函數來處理 UNet 的前向傳播，因為 UNet 的輸出圖像大小與輸入圖像大小不匹配，需要進行特殊處理。
    def _forward_unet(images):
        if infer_args.model == 'unet':
            imgs_padded = F.pad(images, (94, 94, 94, 94), mode='reflect') # UNet 的輸出圖像比輸入圖像小 94 像素（每邊），因此需要對輸入圖像進行反射填充以補償這個尺寸差異。
            outputs = model(imgs_padded)
            return outputs[:, :, 2:-2, 2:-2]  # 將 UNet 的輸出圖像裁剪回與原始輸入圖像相同的大小。
        else:
            return model(images)

    # 顯示 推理 進度條。
    print("="*60)
    test_pbar = tqdm(test_loader, desc="Inferencing")
    with torch.no_grad():
        for images, _ in test_pbar:
            images = images.to(device)

            if infer_args.tta:
                # TTA (Test Time Augmentation) - 讓模型看四種不同的角度，然後取平均機率值。
                p0 = _forward_unet(images)
                p1 = torch.flip(_forward_unet(torch.flip(images, [3])), [3])       # 水平翻轉
                p2 = torch.flip(_forward_unet(torch.flip(images, [2])), [2])       # 垂直翻轉
                p3 = torch.flip(_forward_unet(torch.flip(images, [2, 3])), [2, 3]) # 雙向翻轉
                outputs = (p0 + p1 + p2 + p3) / 4.0
            else:
                outputs = _forward_unet(images)

            probs = torch.sigmoid(outputs)                  # 將模型輸出轉換為概率值。
            preds = (probs > infer_args.threshold).float()  # 將概率值轉換為二值化的預測遮罩。
            preds_np = preds.cpu().numpy()                  # 將預測遮罩從 PyTorch 張量轉換為 NumPy 陣列，以便進行 RLE 編碼。

            for i in range(len(images)):
                image_id = all_image_names[image_idx]  # 獲取當前圖像的 ID。
                mask_np = np.squeeze(preds_np[i]).astype(np.uint8)  # 現在是 uint8 (0 或 1)的二值化遮罩圖像，形狀為 (1, H, W)，需要去掉通道維度。
                original_img_path = os.path.join(infer_args.data_dir, 'images', f"{image_id}.jpg")  # 原始圖像的路徑。
                if os.path.exists(original_img_path):
                    with Image.open(original_img_path) as orig_pil:
                        original_width, original_height = orig_pil.size  # 獲取原始圖像的寬度和高度。

                    mask_pil_pred = Image.fromarray(mask_np, mode='L')  # 將二值化遮罩圖像轉換為 PIL 圖像對象，使用 'L' 模式表示單通道灰度圖像。
                    mask_pil_orig = mask_pil_pred.resize((original_width, original_height), resample=Image.NEAREST)  # 將遮罩圖像調整回原始圖像的大小，使用 NEAREST 插值以保持二值化。
                    mask_np = np.array(mask_pil_orig) # 將調整大小後的遮罩圖像轉換回 NumPy 陣列，以便進行 RLE 編碼。
                else:
                    print(f"Warning: Original image not found for {image_id}, using resized mask for RLE encoding.")

                rle_string = rle_encode(mask_np)  # 對二值化遮罩圖像進行 RLE 編碼，得到 RLE 字符串。
                results.append({'image_id': image_id, 'encoded_mask': rle_string})  # 將當前圖像的 ID 和對應的 RLE 編碼結果添加到 results 列表中。
                image_idx += 1  # 更新圖像索引以處理下一個圖像。

    # 將推理結果保存到 CSV 文件中。
    os.makedirs(infer_args.output_csv_dir, exist_ok=True)  # 確保輸出目錄存在。
    tta_suffix = "_tta" if infer_args.tta else ""
    thresh_suffix = f"_th{int(infer_args.threshold*100)}" if infer_args.threshold != 0.5 else ""
    output_csv = os.path.join(infer_args.output_csv_dir, f'submission_{infer_args.model}{tta_suffix}{thresh_suffix}.csv')
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print("\n" + "="*60)
    print(f"Inference completed! Kaggle submission file saved to: {output_csv}")
    print("="*60)

if __name__ == '__main__':
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

    # 定義命令行參數。
    parser = argparse.ArgumentParser(description="Inference on Oxford-IIIT Pet Dataset")

    data_dir_default = os.path.join(PROJECT_ROOT, 'dataset', 'oxford-iiit-pet')
    save_model_dir_default = os.path.join(PROJECT_ROOT, 'saved_models')
    output_csv_dir_default = os.path.join(PROJECT_ROOT, 'submission')
    parser.add_argument('--data_dir', type=str, default=data_dir_default, help=f'Path to the dataset directory (default: {data_dir_default})')
    parser.add_argument('--save_model_dir', type=str, default=save_model_dir_default, help=f'Directory to load the trained model (default: {save_model_dir_default})')
    parser.add_argument('--output_csv_dir', type=str, default=output_csv_dir_default, help=f'Directory to save the output CSV file for Kaggle submission (default: {output_csv_dir_default})')

    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'res_unet'], help='Model name for inference (default: unet)')
    
    batch_size_default = 16
    threshold_default = 0.5
    parser.add_argument('--batch_size', type=int, default=batch_size_default, help=f'Batch size for inference (default: {batch_size_default})')     
    parser.add_argument('--threshold', type=float, default=threshold_default, help=f'Threshold for converting probabilities to binary masks (default: {threshold_default})')
    
    parser.add_argument('--tta', action='store_true', help='Enable Test-Time Augmentation (TTA) for improved performance') 
    
    args = parser.parse_args()

    inference(args)