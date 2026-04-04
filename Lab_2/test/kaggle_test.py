import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

def rle_encode(mask):
    '''
    製作過程：將二值化遮罩編碼為 Kaggle 要求的 RLE 字串格式。
    '''
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape):
    '''
    反向工程：將 Kaggle 的 RLE 字串解碼回 2D numpy 陣列。
    Args:
        mask_rle (str): RLE 格式字串 (例如 "1 3 10 5")
        shape (tuple): 圖片的真實 (高度, 寬度)
    Returns:
        numpy.ndarray: 0 與 1 的二值化 2D 陣列
    '''
    # 處理模型預測為全黑 (空字串或 NaN) 的防呆機制
    if pd.isna(mask_rle) or mask_rle == '':
        return np.zeros(shape, dtype=np.uint8)
    
    s = str(mask_rle).split()
    # 提取起點與長度 (RLE 起點是 1-based，所以要減 1 轉回程式的 0-based)
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1 
    ends = starts + lengths
    
    # 建立一個全黑的一維陣列
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    
    # 把有前景的地方塗白 (設為 1)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
        
    # Kaggle 要求是 Fortran order (按欄優先) 展平的，所以變回 2D 時也要用 order='F'
    return img.reshape(shape, order='F')

def generate_gt_csv(list_file, gt_dir, output_csv):
    '''
    製作過程：讀取名單，去 trimaps 抓出解答，轉成 RLE 並存成標準答案 CSV。
    '''
    print(f"⚠️  找不到標準答案 CSV，正在從 {list_file} 製作 Ground Truth 答案卷...")
    with open(list_file, 'r') as f:
        lines = f.readlines()
        image_names = [line.strip().split()[0] for line in lines if line.strip() and not line.startswith('#')]

    results = []
    for image_id in tqdm(image_names, desc="Generating GT RLE"):
        gt_path = os.path.join(gt_dir, f"{image_id}.png")
        if not os.path.exists(gt_path):
            print(f"⚠️ 警告: 找不到真實標籤 {gt_path}，跳過此圖片。")
            continue

        with Image.open(gt_path) as img:
            gt_img = np.array(img)

        # 依照 oxford_pet.py 的規則：標籤 1 是前景，2 和 3 是背景
        gt_binary = np.where(gt_img == 1, 1, 0).astype(np.uint8)
        
        # 轉換成 RLE
        rle_string = rle_encode(gt_binary)
        results.append({'image_id': image_id, 'encoded_mask': rle_string})

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ 標準答案 CSV 已成功製作並儲存至: {output_csv}\n (共 {len(results)} 筆資料)")
    print("="*60)

def evaluate_submission(eval_args):
    print(f"\n=== 啟動本地 Kaggle 評估中心 ===")
    
    # 💡 智慧導航：自動定位當前 test/ 資料夾的位置，無論你在哪裡執行指令，gt_*.csv 都會乖乖存進 test/ 裡面
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    gt_csv_path = os.path.join(current_script_dir, f"gt_test_{eval_args.model}.csv")
    
    list_file = os.path.join(eval_args.data_dir, 'annotations', f"test_{eval_args.model}.txt")
    
    # 1. 確認 / 製作 標準答案 CSV
    if not os.path.exists(gt_csv_path):
        if not os.path.exists(list_file):
            print(f"❌ 錯誤: 找不到測試名單 {list_file}，無法製作答案！")
            return
        generate_gt_csv(list_file, eval_args.gt_dir, gt_csv_path)
    else:
        print(f"✅ 找到現成的標準答案 CSV: {gt_csv_path}")

    # 2. 確認 推論提交檔 是否存在
    # 優先使用使用者明確指定的 CSV 路徑；若未提供，才使用既有規則自動組路徑。
    if eval_args.csv_path:
        csv_path = eval_args.csv_path
    else:
        tta_suffix = "_tta" if eval_args.tta else ""
        csv_path = os.path.join(eval_args.csv_dir, f'submission_{eval_args.model}{tta_suffix}.csv')
    if not os.path.exists(csv_path):
        print(f"❌ 錯誤: 找不到推論 CSV 檔案 {csv_path}！請先執行 inference.py。")
        return
        
    print(f"📈 讀取推論檔: {csv_path}")
    pred_df = pd.read_csv(csv_path)
    
    # 將 pred_df 轉換為字典方便快速搜尋
    pred_dict = dict(zip(pred_df['image_id'], pred_df['encoded_mask']))
    
    # 讀取測試清單以確保評估順序
    with open(list_file, 'r') as f:
        image_names = [line.strip().split()[0] for line in f.readlines() if line.strip() and not line.startswith('#')]

    dice_scores = []
    
    print("="*60)
    print("開始比對預測結果與真實標籤...")
    for image_id in tqdm(image_names, desc="Comparing Predictions"):
        
        # 尋找原始的 Ground Truth 標籤 (為了獲取真實的長寬 shape 和驗證二值化遮罩)
        gt_path = os.path.join(eval_args.gt_dir, f"{image_id}.png")
        if not os.path.exists(gt_path):
            print(f"\n⚠️ 警告: 找不到真實標籤 {gt_path}，跳過此圖片。")
            continue
            
        with Image.open(gt_path) as img:
            gt_img = np.array(img)
            
        # 標籤轉換
        gt_binary = np.where(gt_img == 1, 1, 0).astype(np.uint8)
        shape = gt_binary.shape
        
        # 從提交檔中獲取該圖片預測的 RLE
        rle_mask = pred_dict.get(image_id, "")
        
        # 將 CSV 中的 RLE 字串解碼回跟真實圖片一樣大的 2D 陣列
        pred_binary = rle_decode(rle_mask, shape)
        
        # 計算 Dice Score = 2 * (A ∩ B) / (A + B)
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = pred_binary.sum() + gt_binary.sum()
        
        # 防呆：如果兩張圖都是全黑的，算作完美預測 1.0
        if union == 0:
            dice = 1.0 
        else:
            dice = (2.0 * intersection) / union + 1e-7  # 加一個小常數避免除以零
            
        dice_scores.append(dice)
        
    final_dice = np.mean(dice_scores)
    print("="*60)
    print(f"🏆 Local Kaggle Test ({eval_args.model}) Dice Score: {final_dice:.5f}")
    print("="*60+ "\n")

def get_kaggle_args():
    parser = argparse.ArgumentParser(description="Local Kaggle Evaluation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_dir'    , type=str, default=DATA_DIR    , help='Path to the dataset directory')
    parser.add_argument('--gt_dir'      , type=str, default=GT_DIR      , help='Path to the ground truth directory')
    parser.add_argument('--csv_dir'     , type=str, default=CSV_DIR     , help='Path to your inference CSV file')
    parser.add_argument('--csv_path'    , type=str, default=''          , help='Direct path to submission CSV (overrides --csv_dir/--model/--tta when provided)')

    parser.add_argument('--model'       , type=str, default='unet'      , help='Model name for evaluation', choices=['unet', 'res_unet'])

    parser.add_argument('--tta'         , action='store_true'           , help='Whether to evaluate with Test Time Augmentation (TTA)')
    
    return parser.parse_args()

if __name__ == '__main__':
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))            # path-to-/test/
    Lab_2_DIR = os.path.dirname(CURRENT_DIR)                            # path-to-/Lab_2/
    PROJECT_DIR = os.path.join(Lab_2_DIR, 'DL_Lab2_B11107122_凃岳霖')   # path-to-/Lab_2/DL_Lab2_B11107122_凃岳霖/
    DATA_DIR = os.path.join(PROJECT_DIR, 'dataset', 'oxford-iiit-pet')  # path-to-/Lab_2/DL_Lab2_B11107122_凃岳霖/dataset/oxford-iiit-pet
    GT_DIR = os.path.join(DATA_DIR, 'annotations', 'trimaps')           # path-to-/Lab_2/DL_Lab2_B11107122_凃岳霖/dataset/oxford-iiit-pet/annotations/trimaps
    CSV_DIR = os.path.join(PROJECT_DIR, 'submission')                   # path-to-/Lab_2/DL_Lab2_B11107122_凃岳霖/submission
    
    kaggle_args = get_kaggle_args()

    evaluate_submission(kaggle_args)