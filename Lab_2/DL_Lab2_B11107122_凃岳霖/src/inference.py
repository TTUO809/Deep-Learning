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
from utils import set_seed, resolve_model_config
from train import build_model, process_images
from evaluate import load_model_weights

def _print_run_config(infer_args, selected_model, model_path):
    """
    Args:
        infer_args (argparse.Namespace): 包含所有推理參數的命名空間對象。
        selected_model (str): 根據解析後的模型類型（如 "unet" 或 "res_unet"）。
        model_path (str): 模型權重的路徑。
    Description:
        打印推理配置，包括模型名稱、批次大小、是否使用 TTA、閾值、隨機種子，以及模型權重和輸出 CSV 的路徑。
    """

    print(f"=============== Start Inference with 【 {selected_model} 】 ===============")
    print(f"Args:\
         \n - Batch Size: {infer_args.batch_size}\
         \n - Num Workers: {infer_args.num_workers}\
         \n - Seed: {infer_args.seed}\
         \n - Threshold: {infer_args.threshold}\
         \n - TTA: {infer_args.tta}")

    print(f"\nDirectory:\
            \n - Model: {model_path}\
            \n - Output CSV: {infer_args.output_dir}")
    print("=" * 60)

def _build_test_dataloader(infer_args, selected_model):
    """
    Args:
        infer_args (argparse.Namespace): 包含所有推理參數的命名空間對象。
        selected_model (str): 根據解析後的模型類型（如 "unet" 或 "res_unet"）。
    Returns:
        (torch.utils.data.DataLoader): 用於推理的測試集 DataLoader。
        (List[str]): 測試集中所有圖像的名稱列表。
    Description:
        根據推理參數構建測試集的 DataLoader 和所有圖像名稱列表。
        1. 使用 get_oxford_pet_dataloader 函數從指定的數據目錄和分割（如 "test_unet" 或 "test_res_unet"）中加載測試數據，並設置批次大小和工作進程數。
        2. 獲取測試集的所有圖像名稱列表，這些名稱將用於生成提交文件中的 image_id。
    """

    print(f"Loading Test {selected_model} Data...")
    test_loader = get_oxford_pet_dataloader(
        DATA_DIR=infer_args.data_dir,
        split=f'test_{selected_model}',
        batch_size=infer_args.batch_size,
        num_workers=infer_args.num_workers,
    )
    all_image_names = test_loader.dataset.image_names
    return test_loader, all_image_names

def _forward_with_tta(model, images,use_tta):
    """
    Args:
        model (torch.nn.Module): 已經加載權重的模型實例。
        images (torch.Tensor): 輸入圖像張量，形狀為 (B, C, H, W)。
        use_tta (bool): 是否啟用 Test-Time Augmentation (TTA)。
    Returns:
        (torch.Tensor): 模型的輸出張量，形狀為 (B, 1, H, W)，包含每個像素屬於前景的概率。
    Description:
        根據 use_tta 參數決定是否對輸入圖像進行 TTA (Test-Time Augmentation)。
        1. 如果不使用 TTA ，則直接將圖像輸入模型進行前向傳播，返回預測結果。
        2. 如果使用 TTA ，則對輸入圖像進行四種不同的翻轉（原始、水平翻轉、垂直翻轉、雙向翻轉），將每種翻轉後的圖像輸入模型進行前向傳播。
           然後將四種翻轉的預測結果進行反向翻轉以恢復原始方向，最後取四種預測結果的平均值作為最終的預測結果，這樣可以提高模型在推理過程中的穩定性和性能。
    """

    if not use_tta:
        return model(images)

    # TTA (Test Time Augmentation) - 讓模型看四種不同的角度，然後取平均機率值。
    p0 = model(images)
    p1 = torch.flip(model(torch.flip(images, [3])), [3])       # 水平翻轉
    p2 = torch.flip(model(torch.flip(images, [2])), [2])       # 垂直翻轉
    p3 = torch.flip(model(torch.flip(images, [2, 3])), [2, 3]) # 雙向翻轉
    return (p0 + p1 + p2 + p3) / 4.0

def _restore_mask_to_original_size(mask_np, image_id, data_dir):
    """
    Args:
        mask_np (numpy.ndarray): 二值化的遮罩圖像，形狀為 (H, W)，值為 0 或 1。
        image_id (str): 圖像的 ID ，用於構建原始圖像的路徑。
        data_dir (str): 數據集的根目錄，用於構建原始圖像的路徑。
    Returns:
        (numpy.ndarray): 恢復到原始大小的遮罩圖像，形狀為 (H_original, W_original)，值為 0 或 1。
    Description:
        將二值化的遮罩圖像恢復到原始大小，以便進行 RLE 編碼。
        1. 根據 image_id 和 data_dir 構建原始圖像的路徑，嘗試打開原始圖像以獲取其原始尺寸。
        2. 如果原始圖像不存在，則打印警告信息並返回當前大小的遮罩圖像。
        3. 如果原始圖像存在，則使用 PIL 庫將二值化的遮罩圖像轉換為 PIL 圖像，然後使用 resize 方法將遮罩圖像調整到原始圖像的尺寸，最後將調整後的遮罩圖像轉換回 NumPy 陣列並返回。
    """

    original_img_path = os.path.join(data_dir, 'images', f"{image_id}.jpg")
    if not os.path.exists(original_img_path):
        print(f"Warning: Original image not found for {image_id}, using resized mask for RLE encoding.")
        return mask_np

    with Image.open(original_img_path) as orig_pil:
        original_width, original_height = orig_pil.size

    mask_pil_pred = Image.fromarray(mask_np, mode='L')
    mask_pil_orig = mask_pil_pred.resize((original_width, original_height), resample=Image.NEAREST)
    return np.array(mask_pil_orig)

def rle_encode(mask):
    """
    Args:
        mask (numpy.ndarray): 二值化的遮罩圖像，形狀為 (H, W)，值為 0 或 1。
    Returns:
        (str): RLE 編碼的字符串，格式為 "start length start length ..."。
    Description:
        將二值化的遮罩圖像進行 RLE (Run-Length Encoding) 編碼，以便生成提交文件中的 encoded_mask。
        1. 首先將遮罩圖像展平為一維陣列，並按照列優先的順序展平，以符合 RLE 的要求。
        2. 在展平的陣列前後各補一個 0 ，以便精準捕捉從 0 變 1 ，或從 1 變 0 的瞬間，這些瞬間就是 RLE 編碼中的起點和終點。
        3. 使用 numpy 的 where 函數找到展平陣列中值變化的位置，這些位置就是 RLE 編碼中的起點和終點，並將其索引加 1 以符合 RLE 的 1-based 索引要求。
        4. 對找到的起點和終點進行處理，將終點索引減去起點索引，得到每個 run 的長度，最後將起點和長度交替組合成 RLE 編碼的字符串並返回。
    """

    pixels = mask.flatten(order='F')  # 按列優先（Fortran 順序 = 先由上到下、再由左到右）展平圖像，以符合 RLE 的要求。

    # 1. 在前後各補一個 0，方便精準捕捉從 0 變 1，或從 1 變 0 的瞬間
    pixels = np.concatenate([[0], pixels, [0]])
    
    # 2. 讓陣列錯位 1 格做比較，不一樣的地方就是 邊界 (Run 的起點或終點)
    #    np.where(pixels[1:] != pixels[:-1]) 會返回一個包含所有變化位置索引的陣列，這些位置就是 RLE 編碼中的起點和終點。
    #    由於我們在前面補了一個 0，所以索引需要加 1 才能對應到原始圖像的正確位置。
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    
    # 3. 終點減去起點，瞬間算出所有長度 (length)
    #  由於 runs 中的索引是交替的（起點、終點、起點、終點...），我們可以通過將偶數索引的值減去奇數索引的值來計算每個 run 的長度。
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)

def _save_submission(results, infer_args, selected_model):
    """
    Args:
        results (List[Dict[str, str]]): 包含推理結果的列表，每個元素是一個字典，包含 'image_id' 和 'encoded_mask' 兩個鍵。
        infer_args (argparse.Namespace): 包含所有推理參數的命名空間對象。
        selected_model (str): 根據解析後的模型類型（如 "unet" 或 "res_unet"）。
    Description:
        將推理結果保存為 CSV 文件，以便提交到 Kaggle。
        1. 首先確保輸出目錄存在，如果不存在則創建。
        2. 根據推理參數生成輸出 CSV 的文件名稱，包含模型名稱、閾值和是否使用 TTA 的信息，以便區分不同配置的提交文件。
        3. 使用 pandas 將推理結果列表轉換為 DataFrame(是一種表格數據結構，類似於 Excel 表格) ，然後將其保存為 CSV 文件，最後打印完成信息和 CSV 文件的路徑。
    """

    os.makedirs(infer_args.output_dir, exist_ok=True)
    tta_suffix = "_tta" if infer_args.tta else ""
    thresh_suffix = f"_th{int(infer_args.threshold * 100)}" if infer_args.threshold != 0.5 else ""
    output_csv = os.path.join(infer_args.output_dir, f'submission_{selected_model}{thresh_suffix}{tta_suffix}.csv')

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print("\n" + "=" * 60)
    print(f"Inference completed! Kaggle submission file saved to: {output_csv}")
    print("=" * 60)

def inference(infer_args):
    """
    Args:
        infer_args (argparse.Namespace): 包含所有推理參數。
    Description:
        執行推理過程，從加載模型權重、構建測試集 DataLoader、對測試圖像進行推理、將預測遮罩恢復到原始大小、進行 RLE 編碼。
        最後將結果保存為 CSV 文件以供提交到 Kaggle。
    """

    # 啟動隨機種子鎖定，以確保推論過程的可重現性。
    set_seed(infer_args.seed)

    # 根據使用者提供的參數解析出要使用的模型類型和對應的模型權重路徑。
    selected_model, model_path = resolve_model_config(infer_args)

    # 打印推理配置，包括模型名稱、批次大小、是否使用 TTA、閾值、隨機種子，以及模型權重和輸出 CSV 的路徑。
    _print_run_config(infer_args, selected_model, model_path)
    
    # 設置設備 (GPU 或 CPU)。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 構建測試集的 DataLoader 和所有圖像名稱列表。
    test_loader, all_image_names = _build_test_dataloader(infer_args, selected_model)
    image_idx = 0   # 追蹤當前處理的圖像。

    # 初始化模型並加載權重。
    model = build_model(infer_args, device)
    if not load_model_weights(model, model_path, device):
        return

    # 開始 推理 模型。
    model.eval()    # 設置模型為評估模式，禁用 dropout 和 batch normalization 等訓練特定的行為，確保推理過程的穩定性和一致性。
    results = []    # 用來存儲推理結果的列表，每個元素是一個字典，包含 'image_id' 和 'rle_mask' 兩個鍵。

    print("=" * 60)
    test_pbar = tqdm(test_loader, desc="Inferencing")
    with torch.no_grad():
        for images, _ in test_pbar:
            images = images.to(device)
            images = process_images(images, selected_model)
            outputs = _forward_with_tta(model, images, infer_args.tta)

            probs = torch.sigmoid(outputs)                  # 將模型輸出轉換為概率值。
            preds = (probs > infer_args.threshold).float()  # 將概率值轉換為二值化的預測遮罩。
            preds_np = preds.cpu().numpy()                  # 將預測遮罩從 PyTorch 張量轉換為 NumPy 陣列，以便進行 RLE 編碼。

            # 對當前批次中的每個圖像進行處理，將二值化遮罩圖像恢復到原始大小，然後對其進行 RLE 編碼，最後將結果添加到 results 列表中。
            for i in range(len(images)):
                image_id = all_image_names[image_idx]  # 獲取當前圖像的 ID。
                mask_np = np.squeeze(preds_np[i]).astype(np.uint8)  # 現在是 uint8 (0 或 1)的二值化遮罩圖像，形狀為 (1, H, W)，需要去掉通道維度。
                mask_np = _restore_mask_to_original_size(mask_np, image_id, infer_args.data_dir)

                rle_string = rle_encode(mask_np)  # 對二值化遮罩圖像進行 RLE 編碼，得到 RLE 字符串。
                results.append({'image_id': image_id, 'encoded_mask': rle_string})  # 將當前圖像的 ID 和對應的 RLE 編碼結果添加到 results 列表中。
                image_idx += 1  # 更新圖像索引以處理下一個圖像。

    _save_submission(results, infer_args, selected_model)

def get_inference_args():
    """
    Returns:
        argparse.Namespace: 包含所有推理參數的命名空間對象。
    Description:
        定義並解析命令行參數，用於推理過程中指定模型類型、模型權重路徑、輸出目錄、批次大小、閾值、隨機種子等配置。
    """

    parser = argparse.ArgumentParser(description="Inference on Oxford-IIIT Pet Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 路徑
    parser.add_argument('--data_dir'        , type=str  , default=DATA_DIR        , help='Path to the dataset directory')
    parser.add_argument('--save_model_dir'  , type=str  , default=SAVE_MODEL_DIR  , help='Directory to load the trained model')
    parser.add_argument('--output_dir'      , type=str  , default=OUTPUT_DIR      , help='Directory to save the output CSV file for Kaggle submission')
    
    # 模型
    parser.add_argument('--model'           , type=str  , default='unet'          , help='Model name for inference', choices=['unet', 'res_unet'])
    parser.add_argument('--model_path'      , type=str  , default=''              , help='Direct path to the model file (overrides --save_model_dir and --model when provided)')
    
    # 可重現性
    parser.add_argument('--seed'            , type=int  , default=42              , help='Random seed for reproducibility')

    # 推論基本設定
    parser.add_argument('--batch_size'      , type=int  , default=16              , help='Batch size for inference')   
    parser.add_argument('--num_workers'     , type=int  , default=4               , help='Number of worker processes for data loading')

    # 推論增強選項
    parser.add_argument('--threshold'       , type=float, default=0.5             , help='Threshold for converting probabilities to binary masks')
    parser.add_argument('--tta'             , action='store_true'                 , help='Enable Test-Time Augmentation (TTA) for improved performance') 

    return parser.parse_args()

if __name__ == '__main__':
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))            # path-to-/src/
    PROJECT_DIR = os.path.dirname(CURRENT_DIR)                          # path-to-/DL_Lab2_B11107122_凃岳霖/
    DATA_DIR = os.path.join(PROJECT_DIR, 'dataset', 'oxford-iiit-pet')  # path-to-/DL_Lab2_B11107122_凃岳霖/dataset/oxford-iiit-pet/
    SAVE_MODEL_DIR = os.path.join(PROJECT_DIR, 'saved_models')          # path-to-/DL_Lab2_B11107122_凃岳霖/saved_models/
    OUTPUT_DIR = os.path.join(PROJECT_DIR, 'submission')                # path-to-/DL_Lab2_B11107122_凃岳霖/submission/

    infer_args = get_inference_args()

    inference(infer_args)