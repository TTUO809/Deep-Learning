"""iCLEVR dataset loaders for conditional DDPM (Lab 6)."""
import json
import os
from typing import List

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


class ICLEVRTrainDataset(Dataset):
    """Reads `train.json` (dict: filename -> labels) and returns (image, onehot)."""

    def __init__(self, img_dir: str, json_path: str, obj_json_path: str, img_size: int = 64):
        with open(json_path) as f:
            data = json.load(f)
        with open(obj_json_path) as f:
            self.obj_map = json.load(f)

        self.num_cls = len(self.obj_map)
        self.img_dir = img_dir
        self.samples = list(data.items())  # [(filename, [labels])]

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.samples)

    def labels_to_onehot(self, label_list: List[str]) -> torch.Tensor:
        oh = torch.zeros(self.num_cls, dtype=torch.float32)
        for lb in label_list:
            oh[self.obj_map[lb]] = 1.0
        return oh

    def __getitem__(self, idx):
        fname, labels = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        img = self.transform(img)
        return img, self.labels_to_onehot(labels)


class ICLEVRTestDataset(Dataset):
    """Reads `test.json` / `new_test.json` (list of label-lists) and returns onehot only."""

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
        return self.labels_to_onehot(self.samples[idx])


def build_train_loader(img_dir: str, json_path: str, obj_json_path: str,
                        batch_size: int = 64, num_workers: int = 4,
                        img_size: int = 64) -> DataLoader:
    ds = ICLEVRTrainDataset(img_dir, json_path, obj_json_path, img_size=img_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, drop_last=True,
                      persistent_workers=num_workers > 0)
