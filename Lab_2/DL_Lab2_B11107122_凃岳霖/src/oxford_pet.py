import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


class OxfordPetDataset(Dataset):
    # 初始化用。負責讀取資料、切分資料集、建立 Image 和 Mask 的 Paths 列表。
    def __init__(self, root_dir, split='train', val_ratio=0.2, random_seed=42, verbose=False):
        '''
        Args:
            root_dir (str)   : Image 和 Mask 所在的資料夾根目錄路徑。
            split (str)      : 決定要使用 【'train' 或 'val' 或 'test'】 的資料分割 (預設為 'train')。
            transform        : Image 和 Mask 的前處理函數 (預設不更改任何轉換與加工)。
            val_ratio (float): Validation Set 佔 Dataset 的比例 (預設為 0.2 也就是 train:val = 8:2)。
            random_seed (int): 可重現的結果用 (預設為 42)。
        '''

        self.root_dir = root_dir
        self.split = split
        self.verbose = verbose
        
        image_dir = os.path.join(root_dir, 'images')
        mask_dir = os.path.join(root_dir, 'annotations', 'trimaps')

        # 優先讀取 Kaggle 提供的明確切分名單 (train.txt, val.txt, test_unet.txt)。
        list_file = os.path.join(root_dir, 'annotations', f'{split}.txt')

        # 讀取檔案 與 遇意外時的自動切分邏輯。
        if os.path.exists(list_file):
            # 如果 Kaggle 給的指定清單存在，直接使用。
            if self.verbose:
                print(f"[Verbose/init] 正在讀取清單檔案: {list_file}")

            with open(list_file, 'r') as f:
                lines = f.readlines()
                # 忽略空行和註解行，提取第一個欄位作為 Image Name。
                self.image_names = [line.strip().split()[0] for line in lines if line.strip() and not line.startswith('#')]
        else:
            # 如果指定的清單檔案不存在，則回退到自動切分。
            if split in ['train', 'val']:
                trainval_file = os.path.join(root_dir, 'annotations', 'trainval.txt')
                if not os.path.exists(trainval_file):
                    raise FileNotFoundError(f"找不到 清單檔案: {split}.txt ，且 備用清單檔案: {trainval_file} 也不存在！")
                
                if self.verbose:
                    print(f"[Verbose/init] 找不到 清單檔案: {split}.txt ，改為讀取 備用清單檔案: {trainval_file} 並進行動態切分 (Val Ratio: {val_ratio}, Random Seed: {random_seed})")

                with open(trainval_file, 'r') as f:
                    lines = f.readlines()
                    all_image_names = [line.strip().split()[0] for line in lines if line.strip() and not line.startswith('#')]

                random.seed(random_seed)    # 【確保每次切分、打亂結果一致】。
                random.shuffle(all_image_names) # 打亂順序。
                val_size = int(len(all_image_names) * val_ratio) # 全部圖片數量 * Val set 比例，得到 Val set 的圖片數量。
                if split == 'train':
                    self.image_names = all_image_names[val_size:] # 取 [val_ratio ~ len(all_image_names)-1] 的圖片名稱作為 Train set。
                else:
                    self.image_names = all_image_names[:val_size] # 取 [0 ~ val_ratio-1]                    的圖片名稱作為 Val set。
            else:
                # list_file = os.path.join(root_dir, 'annotations', 'test.txt')
                # 如果所有名單都找不到，則報錯。
                raise FileNotFoundError(f"找不到 清單檔案: {split}.txt。請確認檔案名稱是否正確，或提供正確的切分清單檔案！")

        if self.verbose:
            print(f"[Verbose/init] {split} 模式: 共載入 {len(self.image_names)} 張圖片名單。")

        # 最後組合出完整的 Image 和 Mask Paths。
        self.image_paths = [os.path.join(image_dir, name + '.jpg') for name in self.image_names]
        self.mask_paths  = [os.path.join(mask_dir,  name + '.png') for name in self.image_names]
        
        if self.verbose and len(self.image_paths) > 0:
            print(f"[Verbose/init] 首張 Image 路徑: {self.image_paths[0]}")
            print(f"[Verbose/init] 首張 Mask 路徑: {self.mask_paths[0]}")

    # 回傳總共有多少圖片用。
    def __len__(self):
        '''
        Returns:
            (int): 整個 Dataset 中圖片的總數量。
        '''

        return len(self.image_names)

    # 取得指定圖片並執行必要【前處理】用。
    def __getitem__(self, idx):
        '''
        Args:
            idx (int): 要取得的圖片索引。
        Returns:
            image (torch.Tensor): 經過前處理後的圖片。
            mask (torch.Tensor): 經過前處理後的遮罩。
        '''

        if self.verbose:
            print(f"\n[Verbose/getitem] 正在讀取 Index: {idx} (圖片名稱: {self.image_names[idx]})")
        
        # 讀取 Image。
        image = Image.open(self.image_paths[idx]).convert('RGB')        # 【PIL Object (W, H)-RGB】

        # 讀取 Mask。
        if os.path.exists(self.mask_paths[idx]):
            # 如果有 Mask 的情況下，讀取並轉換成二值遮罩【標籤轉換：1(前景) -> 1(前景), 2(背景)&3(邊界) -> 0(背景)】。
            mask = Image.open(self.mask_paths[idx]).convert('L')        # 【PIL Object (W, H)-L】

            mask_np = np.array(mask)                                    # 【-> np.uint8 (H, W)】轉成 NumPy 陣列方便轉換。
            binary_mask = np.where(mask_np == 1, 1, 0).astype(np.uint8) # 【np.uint8 (H, W)】轉成二值遮罩。
            mask = Image.fromarray(binary_mask)                         # 【-> PIL Object (W, H)-L】轉回 PIL Image 格式，方便後續的 transform 處理。 

            if self.verbose:
                print(f"[Verbose/getitem {idx}(1)] 原始的 Mask 像素值內容: {np.unique(mask_np)}")
                print(f"[Verbose/getitem {idx}(2)] 轉換後 Mask 像素值內容: {np.unique(binary_mask)}")

        else:
            # 如果沒有 Mask 的情況下，建立一個全黑的遮罩 (PIL Image 的 size 是 (W, H)，而 np.array 是 (H, W, C)，所以要反轉。並確保 Mask 是 8-bit 的黑白圖)。

            if self.verbose:
                print(f"[Verbose/getitem {idx}] 找不到 Mask ({self.mask_paths[idx]})，產生全 0 遮罩供推論使用。")
            
            mask = Image.fromarray(np.zeros((image.size[1], image.size[0]), dtype=np.uint8)) # 【PIL Object (W, H)-L】

        if self.verbose:
            print(f"[Verbose/getitem {idx}(3)] 原始的 Image Size: {image.size} (W, H), Mode: {image.mode}, Value Range: [{np.array(image).min()}, {np.array(image).max()}]")
            print(f"[Verbose/getitem {idx}(4)] 原始的 Mask Size: {mask.size} (W, H), Mode: {mask.mode}, Value Range: [{np.array(mask).min()}, {np.array(mask).max()}]")

        # 前處理 Image 和 Mask。
        if self.split == 'train':
            # 1. 隨機水平翻轉 (50% 機率)。
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # 2. 隨機垂直翻轉 (50% 機率)。
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # 3. 隨機旋轉 (-15度 到 15度)、隨機平移、隨機縮放。
            affine_params = T.RandomAffine.get_params(
                degrees=[-15, 15], 
                translate=[0.1, 0.1], 
                scale_ranges=[0.9, 1.1], 
                shears=None, img_size=image.size
            )
            image = TF.affine(image, *affine_params, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.affine(mask, *affine_params, interpolation=TF.InterpolationMode.NEAREST)

            # 4. 隨機調整 亮度(20%)、對比度(20%)、飽和度(20%)、色調(5%)。
            if random.random() > 0.5:
                color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
                image = color_jitter(image)

        # 統一調整大小。
        image_size = (388, 388)
        image = TF.resize(image, image_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, image_size, interpolation=TF.InterpolationMode.NEAREST)

        # 將 Image 轉換為 Tensor 並 Normalize，Mask 轉換為 Tensor。
        image = TF.to_tensor(image) # 【torch.float32 (C, H, W)，值域 [0.0, 1.0]】
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 【torch.float32 (C, H, W)，值域經過 Normalize 處理，平均值為 0，標準差為 1，但實際數值會根據原始圖像的顏色分布而有所不同。】
        mask = TF.to_tensor(mask)   # 【torch.float32 (1, H, W)，值域 [0.0, 1.0]】。
        mask = (mask > 0).float()
        
        if self.verbose:
            print(f"[Verbose/getitem {idx}(5)] Transform 後 Image Shape: {image.shape} (C, H, W), Value Range: [{image.min()}, {image.max()}]")
            print(f"[Verbose/getitem {idx}(6)] Transform 後 Mask Shape: {mask.shape} (1, H, W), Value Range: [{mask.min()}, {mask.max()}]")

        return image, mask


def get_oxford_pet_dataloader(root_dir, split='train', batch_size=16, num_workers=4, verbose=False):
    '''
    Args:
        root_dir (str)   : Image 和 Mask 所在的資料夾根目錄路徑。
        split (str)      : 決定要使用 【'train' 或 'val' 或 'test'】 的資料分割 (預設為 'train')。
        batch_size (int) : 每個 Batch 的圖片數量 (預設為 16)。
        num_workers (int): 用於資料載入的子進程數量 (預設為 4)。
    Returns:
        DataLoader: 包含指定資料分割的 DataLoader，已經套用必要的前處理轉換。
    '''

    dataset = OxfordPetDataset(root_dir=root_dir, split=split, verbose=verbose)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'), num_workers=num_workers)

    return dataloader

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    ROOT_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'oxford-iiit-pet')
    import matplotlib.pyplot as plt

    print("=== 開始 Verbose 驗證測試 ===")

    print(f"--- 【Dataset 首張測試】 ---")
    dataset = OxfordPetDataset(root_dir=ROOT_DIR, split='train', verbose=True)

    image, mask = dataset[0]

    print(f"Image 形狀: {image.shape}")
    print(f"Mask 形狀: {mask.shape}")
    print(f"Mask 數值範圍: [{mask.min()}, {mask.max()}]")

    print(f"\n--- 【DataLoader 批次測試】 ---")
    batch_size = 4

    print("\n 載入 Kaggle Train Set 測試 ...")
    train_loader = get_oxford_pet_dataloader(root_dir=ROOT_DIR, batch_size=batch_size, split='train', verbose=True)

    print("\n 載入 Kaggle Test Set (test_unet) 測試 ...")
    test_loader = get_oxford_pet_dataloader(root_dir=ROOT_DIR, batch_size=batch_size, split='test_unet', verbose=True)

    try:
        print("\n從 Train Loader 抓取一組 Batch 進行檢查...")

        # 從 Train Loader 抓出一組 Batch，並顯示 Image 和 Mask 的形狀與內容。
        images_batch, masks_batch = next(iter(train_loader))

        print(f"\nTrain Batch 影像形狀: {images_batch.shape} (預期: [{batch_size}, 3, 256, 256])")
        print(f"Train Batch 遮罩形狀: {masks_batch.shape} (預期: [{batch_size}, 1, 256, 256])")

        plt.figure(figsize=(12, 6))
        for i in range(batch_size):
            # 顯示 Image (分上下兩行，每行 batch_size 個圖，目前在上行)。
            plt.subplot(2, batch_size, i + 1)
            plt.imshow(images_batch[i].permute(1, 2, 0).numpy()) # 【torch.float32 (C, H, W) -> np.array (H, W, C)】(因為經過 Normalize，顏色會偏掉是正常的)
            plt.title(f"Train Image {i}")
            plt.axis('off')

            # 顯示 Mask (目前在下行)。
            plt.subplot(2, batch_size, i + batch_size + 1)
            plt.imshow(masks_batch[i].squeeze(), cmap='gray') # 【torch.float32 (1, H, W) -(squeeze)-> np.array (H, W)】採灰階顯示。
            plt.title(f"Train Mask {i}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\nDataLoader 測試失敗。錯誤訊息: {e}")