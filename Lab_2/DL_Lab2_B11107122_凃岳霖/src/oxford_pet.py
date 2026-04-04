import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from utils import set_seed, detect_optimal_num_workers

class OxfordPetDataset(Dataset):

    def __init__(self, DATA_DIR, split='train', val_ratio=0.2, seed=42, image_size=388, verbose=False):
        """
        Args:
            DATA_DIR (str)           : Image 和 Mask 所在的資料夾根目錄路徑。
            split (str)              : 決定要使用 【'train' 或 'val' 或 'test'】 的資料分割 (預設為 'train')。
            val_ratio (float)        : Validation Set 佔 Dataset 的比例 (預設為 0.2 也就是 train:val = 8:2)。
            seed (int)               : 可重現的結果用 (預設為 42)。
            image_size (int or tuple): 圖片調整的大小 (預設調整為 (388, 388))。
            verbose (bool)           : 是否啟用詳細的 debug 輸出 (預設為 False)。
        Description:
            負責讀取 Oxford Pet Dataset 中的圖片和遮罩，並根據 Kaggle 提供的資料分割進行初始化。如果不存在，則根據 trainval.txt 進行動態切分。
            提供必要的前處理和數據增強，並確保在多進程 DataLoader 中的隨機性可重現。
        """

        self.DATA_DIR = DATA_DIR
        self.split = split
        self.verbose = verbose
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size) # 確保 image_size 是 tuple 格式 (H, W)。
        
        IMAGE_DIR = os.path.join(DATA_DIR, 'images')
        MASK_DIR = os.path.join(DATA_DIR, 'annotations', 'trimaps')
        LIST_PATH = os.path.join(DATA_DIR, 'annotations', f'{split}.txt') # 優先讀取 Kaggle 提供的明確切分名單 (train.txt, val.txt, test_unet.txt, test_res_unet.txt)。

        if os.path.exists(LIST_PATH):
            # 如果 Kaggle 給的指定清單存在，直接使用。
            print(f"[Init] Reading 【{split}】 list file: {LIST_PATH}")

            with open(LIST_PATH, 'r') as f:
                lines = f.readlines()
                self.image_names = [line.strip().split()[0] for line in lines if line.strip() and not line.startswith('#')] # 忽略空行和註解行，提取第一個欄位作為 Image Name。
        else:
            # 如果指定的清單檔案不存在，則回退到自動切分。
            if split in ['train', 'val']:
                TRAINVAL_PATH = os.path.join(DATA_DIR, 'annotations', 'trainval.txt')
                if not os.path.exists(TRAINVAL_PATH):
                    raise FileNotFoundError(f"Split file {split}.txt not found, and fallback file also does not exist: {TRAINVAL_PATH}")

                print(f"[Init] Split file {split}.txt not found. Falling back to {TRAINVAL_PATH} with dynamic split (Val Ratio: {val_ratio}, Random Seed: {seed})")

                with open(TRAINVAL_PATH, 'r') as f:
                    lines = f.readlines()
                    all_image_names = [line.strip().split()[0] for line in lines if line.strip() and not line.startswith('#')]

                # 統一使用 NumPy 全域 RNG，讓切分行為和 set_seed() 一致。
                np.random.seed(seed)
                np.random.shuffle(all_image_names)
                val_size = int(len(all_image_names) * val_ratio) # 全部圖片數量 * Val set 比例，得到 Val set 的圖片數量。
                if split == 'train':
                    self.image_names = all_image_names[val_size:] # 取 [val_ratio ~ len(all_image_names)-1] 的圖片名稱作為 Train set。
                else:
                    self.image_names = all_image_names[:val_size] # 取 [0 ~ val_ratio-1]                    的圖片名稱作為 Val set。
            else:
                raise FileNotFoundError(f"Split file not found: {split}.txt. Please verify the filename or provide a valid split list file.")

        print(f"[Init] {split} mode: loaded {len(self.image_names)} image entries.")

        # 最後組合出完整的 Image 和 Mask Paths。
        self.IMAGE_PATHS = [os.path.join(IMAGE_DIR, name + '.jpg') for name in self.image_names]
        self.MASK_PATHS  = [os.path.join(MASK_DIR,  name + '.png') for name in self.image_names]
        
        if len(self.IMAGE_PATHS) > 0:
            print(f"[Init] First image path: {self.IMAGE_PATHS[0]}")
            print(f"[Init] First mask path: {self.MASK_PATHS[0]}")

    def __len__(self):
        """
        Returns:
            (int): 整個 Dataset 中圖片的總數量。
        Description:
            讓 DataLoader 知道整個 Dataset 的大小，從而能夠正確地迭代和分批次讀取數據。
        """

        return len(self.image_names)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): 要取得的圖片索引。
        Returns:
            image (torch.Tensor): 經過前處理後的圖片。
            mask (torch.Tensor) : 經過前處理後的遮罩。
        Description:
            負責根據給定的索引讀取對應的圖片和遮罩，進行必要的前處理（如數據增強、調整大小、轉換為 Tensor 等），並返回處理後的結果。
        """

        if self.verbose:
            print(f"\n[Verbose/getitem] Reading index: {idx} (image name: {self.image_names[idx]})")
        
        # 讀取 Image。
        image = Image.open(self.IMAGE_PATHS[idx]).convert('RGB')        # 【PIL Object (W, H)-RGB】

        # 讀取 Mask。
        if os.path.exists(self.MASK_PATHS[idx]):
            # 如果有 Mask 的情況下，讀取並轉換成二值遮罩【標籤轉換：1(前景) -> 1(前景), 2(背景)&3(邊界) -> 0(背景)】。
            mask = Image.open(self.MASK_PATHS[idx]).convert('L')        # 【PIL Object (W, H)-L】
            mask_np = np.array(mask)                                    # 【-> np.uint8 (H, W)】轉成 NumPy 陣列方便轉換。
            binary_mask = np.where(mask_np == 1, 1, 0).astype(np.uint8) # 【np.uint8 (H, W)】轉成二值遮罩。
            mask = Image.fromarray(binary_mask)                         # 【-> PIL Object (W, H)-L】轉回 PIL Image 格式，方便後續的 transform 處理。 

            if self.verbose:
                print(f"[Verbose/getitem {idx}(1)] Original mask pixel values: {np.unique(mask_np)}")
                print(f"[Verbose/getitem {idx}(2)] Converted mask pixel values: {np.unique(binary_mask)}")

        else:
            # 如果沒有 Mask 的情況下，建立一個全黑的遮罩。 (PIL Image 的 size 是 (W, H)，而 np.array 是 (H, W, C)，所以要反轉。並確保 Mask 是 8-bit 的黑白圖)

            if self.verbose:
                print(f"[Verbose/getitem {idx}] Mask not found ({self.MASK_PATHS[idx]}), creating an all-zero mask for inference.")
            
            mask = Image.fromarray(np.zeros((image.size[1], image.size[0]), dtype=np.uint8)) # 【PIL Object (W, H)-L】

        if self.verbose:
            print(f"[Verbose/getitem {idx}(3)] Original  image size: {image.size} (W, H), mode: {image.mode}, value range: [{np.array(image).min()}, {np.array(image).max()}]")
            print(f"[Verbose/getitem {idx}(4)] Converted  mask size: {mask.size} (W, H), mode: {mask.mode},   value range: [{np.array(mask).min()}, {np.array(mask).max()}]")

        # 前處理 Image 和 Mask。
        if self.split == 'train':
            
            # 1. 隨機水平翻轉。 (50% 機率)
            if torch.rand(1).item() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # 2. 隨機垂直翻轉。 (50% 機率)
            if torch.rand(1).item() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # 3. 隨機仿射增強 (旋轉 ±15度、平移 ±10%、縮放 90%~110%)。 (50% 機率)
            if torch.rand(1).item() > 0.5:
                affine_params = T.RandomAffine.get_params(
                    degrees=[-15, 15], 
                    translate=[0.1, 0.1], 
                    scale_ranges=[0.9, 1.1], 
                    shears=None, img_size=image.size    # sgears=None 表示不進行剪切變換。
                )
                # 對 Image 使用雙線性插值，對 Mask 使用最近鄰插值，以保持 Mask 的二值化特性。
                image = TF.affine(image, *affine_params, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.affine(mask, *affine_params, interpolation=TF.InterpolationMode.NEAREST)

            # 4. 隨機調整 亮度(20%)、對比度(20%)、飽和度(20%)、色調(5%)。 (50% 機率)
            if torch.rand(1).item() > 0.5:
                color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
                image = color_jitter(image)

        # 統一調整大小。
        image = TF.resize(image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)

        # 將 Image 轉換為 Tensor 並 Normalize，Mask 轉換為 Tensor。
        image = TF.to_tensor(image) # 【torch.float32 (C, H, W)】。
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 【torch.float32 (C, H, W)，值域經過 Normalize 處理，平均值為 0，標準差為 1，但實際數值會根據原始圖像的顏色分布而有所不同。】
        mask = TF.to_tensor(mask)   # 【torch.float32 (1, H, W)]】。
        mask = (mask > 0).float()
        
        if self.verbose:
            print(f"[Verbose/getitem {idx}(5)] Transformed image shape: {image.shape} (C, H, W), value range: [{image.min()}, {image.max()}]")
            print(f"[Verbose/getitem {idx}(6)] Transformed mask shape: {mask.shape} (1, H, W), value range: [{mask.min()}, {mask.max()}]")

        return image, mask

def _worker_init_fn(worker_id):
    """
    Args:
        worker_id (int): DataLoader 中子進程的 ID。
    Description:
        為了確保在使用多進程 DataLoader 時，每個子進程的隨機性都是可重現的，這個函數會根據全域種子和 worker_id 來設置每個子進程的獨立 RNG 種子。
        這樣可以確保在不同的運行中，即使使用多進程載入數據，數據增強的隨機行為也是一致的，從而提高實驗的可重現性。
    """

    worker_seed = torch.initial_seed() % (2 ** 32)  # 獲取全域種子並確保它在 32 位整數範圍內，這是因為某些 RNG 實現可能要求種子必須是 32 位的。
    set_seed(worker_seed, deterministic=False)

def get_oxford_pet_dataloader(DATA_DIR, split='train', batch_size=16, num_workers=4, image_size=388, verbose=False):
    """
    Args:
        DATA_DIR (str)           : Image 和 Mask 所在的資料夾根目錄路徑。
        split (str)              : 決定要使用 【'train' 或 'val' 或 'test'】 的資料分割 (預設為 'train')。
        batch_size (int)         : 每個 Batch 的圖片數量 (預設為 16)。
        num_workers (int)        : 用於資料載入的子進程數量 (預設為 4)。
        image_size (int or tuple): 圖片調整的大小 (預設調整為 (388, 388))。
        verbose (bool)           : 是否啟用詳細的 debug 輸出 (預設為 False)。
    Returns:
        (DataLoader): 包含指定資料分割的 DataLoader，已經套用必要的前處理轉換。
    Description:
        這個函數負責創建並返回一個 DataLoader，該 DataLoader 使用 OxfordPetDataset 來讀取指定分割的數據，並根據提供的參數進行批次大小、子進程數量和圖像大小的配置。
    """
    
    dataset = OxfordPetDataset(DATA_DIR=DATA_DIR, split=split, image_size=image_size, verbose=verbose)
    
    # 建立獨立的 Generator，與全域 RNG 隔離，確保 shuffle 順序不受 augmentation 的 RNG 消耗影響。
    # torch.initial_seed() 讀取由 set_seed() 設定的種子，不需要額外傳入 seed 參數。
    g = torch.Generator()
    g.manual_seed(torch.initial_seed() % (2 ** 32))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
        generator=g,
    )

    return dataloader

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))                # path-to-/src/
    PROJECT_DIR = os.path.dirname(CURRENT_DIR)                              # path-to-/DL_Lab2_B11107122_凃岳霖/
    DATA_DIR    = os.path.join(PROJECT_DIR, 'dataset', 'oxford-iiit-pet')   # path-to-/DL_Lab2_B11107122_凃岳霖/dataset/oxford-iiit-pet/

    # 為了在 headless 環境（如某些遠程伺服器）中也能正常運行，檢測是否存在 DISPLAY 環境變量，如果不存在【headless 環境（無圖形界面）】則使用 "Agg" 後端來避免圖形界面相關的錯誤。
    import matplotlib
    is_headless = not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if is_headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 啟動隨機種子鎖定，以確保訓練過程的可重現性。
    set_seed(42, deterministic=False)

    print("=== Start Oxford Pet Dataset Test ===")

    image_size = 388

    print(f"--- [Dataset first sample test] ---")
    dataset = OxfordPetDataset(DATA_DIR=DATA_DIR, split='train', seed=42, image_size=image_size, verbose=True)

    image, mask = dataset[0]

    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask value range: [{mask.min()}, {mask.max()}]")

    print(f"\n--- [DataLoader batch test] ---")
    batch_size = 2
    
    # 根據系統資源檢測並建議 DataLoader 的 num_workers 設定，提供詳細的系統資訊和建議，以幫助使用者選擇合適的 num_workers 值。
    worker_info = detect_optimal_num_workers(batch_size=batch_size)
    print(f"\n[System Info]")
    print(f"  CPU Cores: {worker_info['cpu_count']}")
    print(f"  Recommended num_workers: {worker_info['recommended']}")
    print(f"  Safe range: {worker_info['safe_range'][0]} ~ {worker_info['safe_range'][1]}")
    print(f"  Advice: {worker_info['advice']}")
    
    # 為了確保測試的穩定性，這裡直接使用推薦的 num_workers 值，並在輸出中明確指出這一點。
    num_workers = worker_info['recommended']
    print(f"\nUsing num_workers={num_workers} for stable testing.")

    print("\nLoading Kaggle train split (train) ...")
    train_loader = get_oxford_pet_dataloader(DATA_DIR=DATA_DIR, batch_size=batch_size, split='train', num_workers=num_workers, image_size=image_size, verbose=True)

    print("\nLoading Kaggle test split (test_unet) ...")
    test_loader = get_oxford_pet_dataloader(DATA_DIR=DATA_DIR, batch_size=batch_size, split='test_unet', num_workers=num_workers, image_size=image_size, verbose=True)

    try:
        print("\nFetching one batch from train loader for inspection...")

        # 從 Train Loader 抓出一組 Batch。
        images_batch, masks_batch = next(iter(train_loader))

        print(f"\nTrain batch image shape: {images_batch.shape} (expected: [{batch_size}, 3, {image_size}, {image_size}])")
        print(f"Train batch mask shape: {masks_batch.shape} (expected: [{batch_size}, 1, {image_size}, {image_size}])")

        # 顯示這個 Batch 的圖片和遮罩，幫助確認前處理和增強是否正常工作。
        plt.figure(figsize=(12, 6))
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        for i in range(batch_size):

            # 顯示 Image (分上下兩行，每行 batch_size 個圖，目前在上行)。
            plt.subplot(2, batch_size, i + 1)
            image_np = images_batch[i].permute(1, 2, 0).numpy()
            image_np = np.clip(image_np * std + mean, 0.0, 1.0) # 反 Normalize 後再顯示，確保圖像看起來正常。
            plt.imshow(image_np)
            plt.title(f"Train Image {i}")
            plt.axis('off')

            # 顯示 Mask (目前在下行)。
            plt.subplot(2, batch_size, i + batch_size + 1)
            plt.imshow(masks_batch[i].squeeze(), cmap='gray') # 【torch.float32 (1, H, W) -(squeeze)-> np.array (H, W)】採灰階顯示。
            plt.title(f"Train Mask {i}")
            plt.axis('off')

        plt.tight_layout()

        # 在 headless 環境中，無法使用 plt.show() 顯示圖像，因此將圖像保存到本地文件中，並提供保存路徑的輸出提示。
        if is_headless:
            PREVIEW_PATH = os.path.join(PROJECT_DIR, "debug_batch_preview.png")
            plt.savefig(PREVIEW_PATH, dpi=150)
            print(f"Headless environment detected. Saved preview to: {PREVIEW_PATH}")
        else:
            plt.show()

    except Exception as e:
        print(f"\nDataLoader test failed. Error: {e}")