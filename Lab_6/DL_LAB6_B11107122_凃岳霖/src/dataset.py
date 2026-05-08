"""===============================================================
【 PyTorch Dataset and DataLoader implementations for the iCLEVR dataset. 】

Handles image loading, preprocessing, and multi-label one-hot encoding
for conditional diffusion models.
==============================================================="""

import json
import os
from typing import List

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


class ICLEVRTrainDataset(Dataset):
    """---------------------------------------------------------------
    【 Dataset class for loading iCLEVR training images and conditions. 】
    
    Reads `train.json` (dict: filename -> labels) and returns (image, onehot).
    ---------------------------------------------------------------"""

    def __init__(self, img_dir: str, json_path: str, obj_json_path: str, img_size: int = 64):
        # 1. 讀取標籤與物件對應表：將 JSON 內容轉為字典。
        with open(json_path) as f:
            data = json.load(f)
        with open(obj_json_path) as f:
            self.obj_map = json.load(f)

        self.num_cls = len(self.obj_map)
        self.img_dir = img_dir
        self.samples = list(data.items())  # [(filename, [labels])]

        # 2. 定義影像前處理：調整尺寸 -> 轉為 Tensor ，（0~255）縮放到 [0, 1]  -> 標準化至 [-1, 1]。
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.samples)

    def labels_to_onehot(self, label_list: List[str]) -> torch.Tensor:
        """Converts a list of string labels into a multi-label one-hot tensor."""
        # 1. 根據類別總數建立全零張量。
        oh = torch.zeros(self.num_cls, dtype=torch.float32)
        # 2. 遍歷標籤，根據 obj_map 找到對應索引並將值設為 1.0。
        for lb in label_list:
            oh[self.obj_map[lb]] = 1.0
        return oh

    def __getitem__(self, idx):
        # 1. 讀取圖片並轉為 RGB 模式（避免 PNG 包含 Alpha 通道）。
        fname, labels = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        # 2. 執行影像增強 / 轉換並將標籤轉為 One-hot。
        img = self.transform(img)
        return img, self.labels_to_onehot(labels)


class ICLEVRTestDataset(Dataset):
    """---------------------------------------------------------------
    【 Dataset class for loading iCLEVR evaluation conditions. 】
    
    Reads `test.json` / `new_test.json` (list of label-lists) and returns onehot only.
    ---------------------------------------------------------------"""

    def __init__(self, json_path: str, obj_json_path: str):
        with open(json_path) as f:
            data = json.load(f)
        with open(obj_json_path) as f:
            self.obj_map = json.load(f)
        self.num_cls = len(self.obj_map)
        self.samples = data  # list of label-lists

    def __len__(self):
        return len(self.samples)

    def labels_to_onehot(self, label_list: List[str]) -> torch.Tensor:
        oh = torch.zeros(self.num_cls, dtype=torch.float32)
        for lb in label_list:
            oh[self.obj_map[lb]] = 1.0
        return oh

    def __getitem__(self, idx):
        # 僅回傳條件 One-hot，用於推論模型生成影像。
        return self.labels_to_onehot(self.samples[idx])


def build_train_loader(img_dir: str, json_path: str, obj_json_path: str,
                        batch_size: int = 64, num_workers: int = 4,
                        img_size: int = 64) -> DataLoader:
    """---------------------------------------------------------------
    【 Builds and returns the training DataLoader with optimized data fetching. 】
    ---------------------------------------------------------------"""
    
    ds = ICLEVRTrainDataset(img_dir, json_path, obj_json_path, img_size=img_size)

    # 1. pin_memory=True：將資料鎖定在分頁記憶體中，加速傳送到 GPU。
    # 2. drop_last=True：捨棄最後不足一個 Batch 的資料，確保訓練步數穩定且 shape 一致。
    # 3. persistent_workers：訓練結束後保持 Worker 進程不釋放，減少每個 Epoch 切換時的等待時間。
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, drop_last=True,
                      persistent_workers=num_workers > 0)
