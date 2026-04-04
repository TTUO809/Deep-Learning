import os
import argparse
import torch
from tqdm import tqdm

from oxford_pet import get_oxford_pet_dataloader
from utils import set_seed, cal_dice_score, resolve_model_config
from train import build_model, process_images

def _print_run_config(eval_args, selected_model, model_path):
    """
    Args:
        eval_args (argparse.Namespace): 包含所有評估參數的命名空間對象。
        selected_model (str): 根據解析後的模型類型（如 "unet" 或 "res_unet"）。
        model_path (str): 模型權重的路徑。
    Description:
        打印評估配置，包括模型名稱、批次大小、 threshold 、隨機種子，以及模型權重路徑。
    """

    print(f"=============== Start Evaluate with 【 {selected_model} 】 ===============")
    print(f"Args:\
         \n - Batch Size: {eval_args.batch_size}\
         \n - Num Workers: {eval_args.num_workers}\
         \n - Threshold: {eval_args.threshold}\
         \n - Seed: {eval_args.seed}\
         \n - Auto Threshold: {eval_args.auto_threshold} [{eval_args.threshold_start}, {eval_args.threshold_end}, step={eval_args.threshold_step}]")
    
    print(f"\nDirectory:\
            \n - Model: {model_path}")
    print("=" * 60)

def _build_val_dataloader(eval_args):
    """
    Args:
        eval_args (argparse.Namespace): 包含所有評估參數的命名空間對象。
    Returns:
        (torch.utils.data.DataLoader): 驗證集 DataLoader。
    Description:
        構建驗證集 DataLoader。
    """

    print(f"Loading Validation Data...")
    return get_oxford_pet_dataloader(
        DATA_DIR=eval_args.data_dir,
        split='val',
        batch_size=eval_args.batch_size,
        num_workers=eval_args.num_workers,
    )

def load_model_weights(model, model_path, device):
    """
    Args:
        model (torch.nn.Module): 已經初始化的模型實例。
        model_path (str): 模型權重的路徑。
        device (torch.device): 設備 (GPU 或 CPU)。
    Returns:
        (bool): 是否成功加載模型權重。
    Description:
        嘗試從指定的模型權重路徑加載權重到模型中。
        1. 使用 torch.load 函數加載模型權重，並將其映射到當前設備（ GPU 或 CPU ）。
        2. 如果成功加載權重，則返回 True ；如果模型文件未找到，則捕獲 FileNotFoundError ，打印錯誤信息並返回 False。
    """

    print(f"Loading model weights from: {model_path}")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))   # 只加載權重，不加載其他訓練狀態（如優化器狀態、epoch 等），確保推理過程的簡潔性和可重現性。
        return True
    except FileNotFoundError:
        print(f"Error: Could not find model file {model_path}! Please run train.py to train the model first.")
        return False

def _build_thresholds(start, end, step):
    """
    Args:
        start (float): 起始 threshold 值。
        end (float): 結束 threshold 值。
        step (float): threshold 之間的步長。
    Returns:
       ( List[float]): 生成的 threshold 列表。
    Description:
        根據指定的起始值、結束值和步長生成一系列 threshold 值，用於在評估過程中掃描不同的 threshold 以找到最佳的 threshold。
        1. 首先檢查輸入參數的有效性，確保 step 大於 0 ， start 小於或等於 end ，且 start 和 end 都在 [0, 1] 範圍內。
        2. 使用列表推導式生成從 start 到 end （包含 end ）之間以 step 為間隔的 threshold 值，並將其四捨五入到小數點後 6 位以避免浮點數精度問題。
        3. 最後檢查生成的 threshold 列表的最後一個值是否因為浮點數誤差而遺失了 end ，如果是，則將 end 添加到列表中。
    """

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

def _evaluate_with_thresholds(model, val_loader, device, model_name, thresholds):
    """
    Args:
        model (torch.nn.Module): 已加載權重的模型。
        val_loader (torch.utils.data.DataLoader): 驗證集的 DataLoader。
        device (torch.device): 設備 (GPU 或 CPU)。
        model_name (str): 模型名稱，用於決定是否需要 padding。
        thresholds (List[float]): 要評估的 threshold 列表。
    Returns:
        (Dict[float, float]): 每個 threshold 對應的平均 Dice Score。
    Description:
        使用指定的模型和驗證集 DataLoader 評估模型在不同 threshold 下的表現，計算並返回每個 threshold 對應的平均 Dice Score。
        1. 初始化一個字典 dice_sums 用於累加每個 threshold 的 Dice 分數總和。
        2. 將模型設置為評估模式，並使用 tqdm 包裝驗證集 DataLoader 以顯示進度條。
        3. 在不計算梯度的上下文中，對驗證集中的每個批次進行迭代：
            a. 將圖像和遮罩移動到指定設備。
            b. 根據模型名稱對圖像進行必要的處理（如 padding）。
            c. 對處理後的圖像進行前向傳播，獲得模型輸出。
            d. 對於每個 threshold，計算當前批次的 Dice Score 並累加到 dice_sums 中。
            e. 如果只掃描一個 threshold，則在進度條上顯示當前的 Dice 分數。
        4. 最後，計算每個 threshold 的平均 Dice Score 並返回一個字典 avg_dice，其中鍵為 threshold，值為對應的平均 Dice Score。
    """

    # (Dictionary) 累加每個 threshold 的 Dice 分數總和。
    dice_sums = {th: 0.0 for th in thresholds}

    model.eval()
    val_pbar = tqdm(val_loader, desc="Evaluating")
    with torch.no_grad():
        for images, masks in val_pbar:
            images, masks = images.to(device), masks.to(device)

            images = process_images(images, model_name)

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
    """
    Args:
        eval_args (argparse.Namespace): 包含所有評估參數。
    Description:
        執行評估過程，從加載模型權重、構建驗證集 DataLoader、對驗證集圖像進行評估、計算不同 threshold 下的平均 Dice 分數，最後打印評估結果。
    """

    # 啟動隨機種子鎖定，以確保推論過程的可重現性。
    set_seed(eval_args.seed)
    
    # 根據使用者提供的參數解析出要使用的模型類型和對應的模型權重路徑。
    selected_model, model_path = resolve_model_config(eval_args)

    # 打印評估配置，包括模型名稱、批次大小、threshold、隨機種子，以及模型權重路徑。
    _print_run_config(eval_args, selected_model, model_path)

    # 設置設備 (GPU 或 CPU)。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 構建驗證集 DataLoader。
    val_loader = _build_val_dataloader(eval_args)

    # 根據解析出的模型類型構建模型，並將其移動到指定設備上。
    eval_args.model = selected_model
    model = build_model(eval_args, device)
    if not load_model_weights(model, model_path, device):
        return

    # 開始評估模型。
    print("="*60)
    if eval_args.auto_threshold:
        thresholds = _build_thresholds(eval_args.threshold_start, eval_args.threshold_end, eval_args.threshold_step)
        print(f"Auto threshold scan: start={eval_args.threshold_start}, end={eval_args.threshold_end}, step={eval_args.threshold_step}")
        print(f"Scanning {len(thresholds)} thresholds: {thresholds}")
    else:
        thresholds = [eval_args.threshold]

    avg_dice_map = _evaluate_with_thresholds(model, val_loader, device, selected_model, thresholds)

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

def get_eval_args():
    """
    Returns:
        argparse.Namespace: 包含所有評估參數的命名空間對象。
    Description:
        定義並解析命令行參數，用於評估過程中指定模型類型、模型權重路徑、批次大小、閾值與自動 threshold 掃描等配置。
    """

    parser = argparse.ArgumentParser(description="Evaluate on Oxford-IIIT Pet Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 路徑
    parser.add_argument('--data_dir'        , type=str, default=DATA_DIR        , help=f'Path to the dataset directory')
    parser.add_argument('--save_model_dir'  , type=str, default=SAVE_MODEL_DIR  , help=f'Directory to load the trained model')
    
    # 模型
    parser.add_argument('--model'           , type=str, default='unet'          , help='Model name for evaluation', choices=['unet', 'res_unet'])
    parser.add_argument('--model_path'      , type=str, default=''              , help='Direct path to the model file (overrides --save_model_dir and --model when provided)')

    # 可重現性
    parser.add_argument('--seed'            , type=int, default=42              , help='Random seed for reproducibility')

    # 驗證基本設定
    parser.add_argument('--batch_size'      , type=int, default=16              , help='Batch size for evaluation')
    parser.add_argument('--num_workers'     , type=int, default=4               , help='Number of worker processes for data loading')

    # Threshold 掃描設定
    parser.add_argument('--threshold'       , type=float, default=0.5           , help='Threshold for converting probabilities to binary masks')
    parser.add_argument('--auto_threshold'  , action='store_true'               , help='Auto scan thresholds and report the best one on val set')
    parser.add_argument('--threshold_start' , type=float, default=0.3           , help='Start threshold for auto scan')
    parser.add_argument('--threshold_end'   , type=float, default=0.7           , help='End threshold for auto scan')
    parser.add_argument('--threshold_step'  , type=float, default=0.05          , help='Step size for auto threshold scan')

    return parser.parse_args()

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))            # path-to-/src/
    PROJECT_DIR = os.path.dirname(CURRENT_DIR)                          # path-to-/Lab_2/DL_Lab2_B11107122_凃岳霖/
    DATA_DIR = os.path.join(PROJECT_DIR, 'dataset', 'oxford-iiit-pet')  # path-to-/Lab_2/DL_Lab2_B11107122_凃岳霖/dataset/oxford-iiit-pet
    SAVE_MODEL_DIR = os.path.join(PROJECT_DIR, 'saved_models')          # path-to-/Lab_2/DL_Lab2_B11107122_凃岳霖/saved_models

    eval_args = get_eval_args()

    evaluate(eval_args)