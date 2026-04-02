# Lab 2 — Binary Semantic Segmentation (Oxford-IIIT Pet)

使用 UNet 與 ResNet34-UNet 對 Oxford-IIIT Pet 資料集進行寵物前景分割。

---

## Table of Contents

- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Dataset Setup](#dataset-setup)
- [Training](#training)
- [Validation](#validation)
- [Inference (Generate Kaggle Submission CSV)](#inference-generate-kaggle-submission-csv)
- [Local Kaggle Evaluation](#local-kaggle-evaluation)
- [Quick Start Commands](#quick-start-commands)
- [Technical Details](TECHNICAL.md)

---

## Project Structure

```
Lab_2/
├── README.md                          ← 本文件
├── test/
│   └── kaggle_test.py                 ← 本地 Kaggle 評估工具
└── DL_Lab2_B11107122_凃岳霖/
    ├── requirements.txt
    ├── dataset/
    │   ├── nycu-2026-spring-dl-lab2-unet.zip
    │   ├── nycu-2026-spring-dl-lab2-res-net-34-unet.zip
    │   └── oxford-iiit-pet/           ← 資料集（執行下載腳本後產生）
    │       ├── images/
    │       └── annotations/
    │           └── trimaps/
    ├── saved_models/                  ← 訓練完成的模型權重（自動產生）
    ├── submission/                    ← 推理輸出的 CSV（自動產生）
    └── src/
        ├── download_dataset.py        ← 資料集下載與解壓
        ├── train.py                   ← 訓練
        ├── evaluate.py                ← 驗證集評估
        ├── inference.py               ← 推理 並 產生 Kaggle 提交 CSV
        ├── oxford_pet.py              ← Dataset / DataLoader
        ├── utils.py                   ← Dice Score、Loss Functions
        └── models/
            ├── unet.py
            └── resnet34_unet.py
```

---

## Environment Setup

> 建議使用 Python 3.9+，並在虛擬環境內安裝。

```bash
cd DL_Lab2_B11107122_凃岳霖
pip install -r requirements.txt
```

---

## Dataset Setup

本專案已包含 Kaggle 切分 zip，請先確認 `dataset/` 內有以下檔案：

| 檔名 |
|------|
| `nycu-2026-spring-dl-lab2-unet.zip` |
| `nycu-2026-spring-dl-lab2-res-net-34-unet.zip` |

### Run Download Script

腳本會自動下載官方 Oxford-IIIT Pet 圖片與標注，並把你放在 `dataset/` 內的 Kaggle zip 解壓成 `.txt` 切分名單到 `annotations/`。

```bash
# 工作目錄：Lab_2/
python DL_Lab2_B11107122_凃岳霖/src/download_dataset.py
```

完成後 `dataset/oxford-iiit-pet/` 應包含：

```
oxford-iiit-pet/
├── images/          ← ~7,400 張 .jpg
└── annotations/
    ├── trimaps/     ← Ground Truth .png
    ├── train.txt
    ├── val.txt
    ├── test_unet.txt
    └── test_res_unet.txt
```

若上述檔案存在，即可直接進入訓練與評估流程。

---

## Training

```bash
# 工作目錄：Lab_2/

# UNet（預設參數）
python DL_Lab2_B11107122_凃岳霖/src/train.py --model unet

# ResNet34-UNet（預設參數）
python DL_Lab2_B11107122_凃岳霖/src/train.py --model res_unet

# 常用參數
python DL_Lab2_B11107122_凃岳霖/src/train.py \
    --model unet \
    --epochs 200 \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --early_stop_patience 20 \
    --resume saved_models/unet_best.pth   # 斷點續訓（可選）
```

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--model` | `unet` | `unet` / `res_unet` |
| `--epochs` | `20` | 訓練總 epoch 數 |
| `--batch_size` | `16` | Batch size |
| `--learning_rate` | `5e-4` | AdamW 學習率 |
| `--weight_decay` | `1e-4` | AdamW weight decay |
| `--warmup_epochs` | `5` | LR Warmup epoch 數 |
| `--early_stop_patience` | `5` | 驗證 Dice 不提升的容忍 epoch 數 |
| `--grad_clip` | `1.0` | 梯度裁剪最大 norm |
| `--use_focal` | `True` | 使用 FocalDiceLoss（否則用 BCEDiceLoss）|
| `--resume` | `None` | 斷點模型路徑 |

訓練結束後最佳模型自動儲存至 `saved_models/{model}_best.pth`。

---

## Validation

### Single Threshold

```bash
python DL_Lab2_B11107122_凃岳霖/src/evaluate.py --model unet --threshold 0.5
```

### Auto Threshold Scan

加上 `--auto_threshold` 旗標，腳本會在一次前向傳播中掃描指定範圍內的所有 threshold，最後列出每個結果並標出最佳值。

```bash
python DL_Lab2_B11107122_凃岳霖/src/evaluate.py \
    --model unet \
    --auto_threshold \
    --threshold_start 0.3 \
    --threshold_end 0.7 \
    --threshold_step 0.05
```

### Evaluate with Custom Model Path

若要評估特定的模型權重檔（而非自動組合 `saved_models/{model}_best.pth`），使用 `--model_path`：

```bash
python DL_Lab2_B11107122_凃岳霖/src/evaluate.py \
    --model_path ./DL_Lab2_B11107122_凃岳霖/saved_models/unet_best_145e.pth \
    --auto_threshold
```

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--model` | `unet` | `unet` / `res_unet` |
| `--model_path` | — | 直接指定模型權重檔路徑（優先於自動組路徑；模型類型會從檔名推斷）|
| `--threshold` | `0.5` | 單一 threshold 模式下使用 |
| `--auto_threshold` | — | 啟用自動掃描模式 |
| `--threshold_start` | `0.3` | 掃描起始值 |
| `--threshold_end` | `0.7` | 掃描結束值 |
| `--threshold_step` | `0.05` | 掃描步長 |
| `--batch_size` | `16` | Batch size |

---

## Inference (Generate Kaggle Submission CSV)

### Basic Inference

```bash
# 基本推理
python DL_Lab2_B11107122_凃岳霖/src/inference.py --model unet

# 啟用 TTA（Test Time Augmentation）
python DL_Lab2_B11107122_凃岳霖/src/inference.py --model unet --tta

# 指定自訂 threshold
python DL_Lab2_B11107122_凃岳霖/src/inference.py --model res_unet --tta --threshold 0.45
```

### Inference with Custom Model Path

使用 `--model_path` 指定特定的模型權重檔：

```bash
python DL_Lab2_B11107122_凃岳霖/src/inference.py \
    --model_path ./DL_Lab2_B11107122_凃岳霖/saved_models/res_unet_best_168e.pth \
    --tta \
    --threshold 0.45
```

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--model` | `unet` | `unet` / `res_unet` |
| `--model_path` | — | 直接指定模型權重檔路徑（優先於自動組路徑；模型類型會從檔名推斷）|
| `--threshold` | `0.5` | 二值化閾值 |
| `--tta` | — | 啟用 TTA（水平、垂直、雙向翻轉取平均）|
| `--batch_size` | `16` | Batch size |

輸出 CSV 儲存至 `submission/`，命名規則：

```
submission_{model}[_tta][_th{threshold*100}].csv
# 範例：submission_unet_tta.csv、submission_res_unet_tta_th45.csv
```

---

## Local Kaggle Evaluation

用來在本地計算與 Kaggle 相同的 Dice Score，驗證提交結果。

### Basic Evaluation

```bash
# 工作目錄：Lab_2/

# 評估 UNet（一般推理）
python test/kaggle_test.py --model unet

# 評估 UNet（TTA）
python test/kaggle_test.py --model unet --tta

# 評估 ResNet34-UNet（TTA）
python test/kaggle_test.py --model res_unet --tta
```

### Evaluate with Custom CSV Path

優先使用明確指定的 CSV 檔案路徑：

```bash
python test/kaggle_test.py \
    --model unet \
    --csv_path ./DL_Lab2_B11107122_凃岳霖/submission/submission_unet_tta.csv
```

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--model` | `unet` | `unet` / `res_unet` |
| `--tta` | — | 對應 TTA 推理結果的 CSV |
| `--csv_path` | — | 直接指定 CSV 路徑（優先於自動組路徑）|

首次執行會自動從 `trimaps/` 產生 Ground Truth CSV（`test/gt_test_{model}.csv`）並快取，之後直接讀取。

---

## Quick Start Commands

基於實驗結果的建議執行流程（200 epoch、early stopping patience 20、TTA）。

### Environment Setup (First Time)

```bash
# 安裝套件
pip install -r DL_Lab2_B11107122_凃岳霖/requirements.txt
```

### Dataset Setup (First Time)

```bash
# 建立/補齊資料集
python DL_Lab2_B11107122_凃岳霖/src/download_dataset.py
```

### UNet Full Pipeline

```bash
# 訓練（約 7.5 小時）
python DL_Lab2_B11107122_凃岳霖/src/train.py --model unet --epochs 200 --early_stop_patience 20 | tee train_unet.log

# 驗證集評估並自動掃描最佳 threshold
python DL_Lab2_B11107122_凃岳霖/src/evaluate.py --model unet --auto_threshold | tee eval_unet.log

# 推理（使用 TTA 提升準確度）
python DL_Lab2_B11107122_凃岳霖/src/inference.py --model unet --tta | tee infer_unet.log

# 本地 Kaggle 評估
python test/kaggle_test.py --model unet --tta | tee kaggle_unet.log
```

**預期結果**：Best Val Dice ~0.9080 | Kaggle Dice ~0.91442

### ResNet34-UNet Full Pipeline

```bash
# 訓練
python DL_Lab2_B11107122_凃岳霖/src/train.py --model res_unet --epochs 200 --early_stop_patience 20 | tee train_res_unet.log

# 驗證集評估並自動掃描最佳 threshold
python DL_Lab2_B11107122_凃岳霖/src/evaluate.py --model res_unet --auto_threshold | tee eval_res_unet.log

# 推理（使用自訂 threshold 0.45 與 TTA）
python DL_Lab2_B11107122_凃岳霖/src/inference.py --model res_unet --threshold 0.45 --tta | tee infer_res_unet.log

# 本地 Kaggle 評估
python test/kaggle_test.py --model res_unet --tta | tee kaggle_res_unet.log
```

**預期結果**：Best Val Dice ~0.9267 | Kaggle Dice ~0.92985