# Lab 2 — Binary Semantic Segmentation (Oxford-IIIT Pet)

使用 UNet 與 ResNet34-UNet 對 Oxford-IIIT Pet 資料集進行寵物前景分割。

---

## 專案結構

```
Lab_2/
├── README.md                          ← 本文件
├── test/
│   └── kaggle_test.py                 ← 本地 Kaggle 評估工具
└── DL_Lab2_B11107122_凃岳霖/
    ├── requirements.txt
    ├── dataset/                       ← 資料集（執行下載腳本後產生）
    │   └── oxford-iiit-pet/
    │       ├── images/
    │       └── annotations/
    │           └── trimaps/
    ├── saved_models/                  ← 訓練完成的模型權重（自動產生）
    ├── submission/                    ← 推理輸出的 CSV（自動產生）
    └── src/
        ├── download_dataset.py        ← 資料集下載與解壓
        ├── train.py                   ← 訓練
        ├── evaluate.py                ← 驗證集評估 / 自動掃描 threshold
        ├── inference.py               ← 推理並產生 Kaggle 提交 CSV
        ├── oxford_pet.py              ← Dataset / DataLoader
        ├── utils.py                   ← Dice Score、Loss Functions
        └── models/
            ├── unet.py
            └── resnet34_unet.py
```

---

## 環境安裝

> 建議使用 Python 3.9+，並在虛擬環境內安裝。

```bash
cd DL_Lab2_B11107122_凃岳霖
pip install -r requirements.txt
```

---

## 資料集建立

初始只有 `src/` 與 `requirements.txt`，需要執行以下步驟建立完整資料集。

### Step 1 — 將 Kaggle 提供的 zip 檔放入 dataset

你會自行提供 Kaggle 下載的競賽切分檔，請先把以下兩個 zip 放到 `dataset/` 目錄下：

| 檔名 |
|------|
| `nycu-2026-spring-dl-lab2-unet.zip` |
| `binary-semantic-segmentation-res-net-34-u-net.zip` |

放置後的資料夾結構：

```
DL_Lab2_B11107122_凃岳霖/
└── dataset/
    ├── nycu-2026-spring-dl-lab2-unet.zip
    └── binary-semantic-segmentation-res-net-34-u-net.zip
```

### Step 2 — 執行下載腳本

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

---

## 訓練

```bash
# 工作目錄：Lab_2/

# UNet（預設參數）
python DL_Lab2_B11107122_凃岳霖/src/train.py --model unet

# ResNet34-UNet
python DL_Lab2_B11107122_凃岳霖/src/train.py --model res_unet

# 常用參數
python DL_Lab2_B11107122_凃岳霖/src/train.py \
    --model unet \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --early_stop_patience 10 \
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

## 驗證集評估

### 單一 threshold

```bash
python DL_Lab2_B11107122_凃岳霖/src/evaluate.py --model unet --threshold 0.5
```

### 自動掃描最佳 threshold

加上 `--auto_threshold` 旗標，腳本會在一次前向傳播中掃描指定範圍內的所有 threshold，最後列出每個結果並標出最佳值。

```bash
python DL_Lab2_B11107122_凃岳霖/src/evaluate.py \
    --model unet \
    --auto_threshold \
    --threshold_start 0.3 \
    --threshold_end 0.7 \
    --threshold_step 0.05
```

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--model` | `unet` | `unet` / `res_unet` |
| `--threshold` | `0.5` | 單一 threshold 模式下使用 |
| `--auto_threshold` | — | 啟用自動掃描模式 |
| `--threshold_start` | `0.3` | 掃描起始值 |
| `--threshold_end` | `0.7` | 掃描結束值 |
| `--threshold_step` | `0.05` | 掃描步長 |
| `--batch_size` | `16` | Batch size |

---

## 推理（產生 Kaggle 提交 CSV）

```bash
# 基本推理
python DL_Lab2_B11107122_凃岳霖/src/inference.py --model unet

# 啟用 TTA（Test Time Augmentation）
python DL_Lab2_B11107122_凃岳霖/src/inference.py --model unet --tta

# 指定自訂 threshold
python DL_Lab2_B11107122_凃岳霖/src/inference.py --model res_unet --tta --threshold 0.45
```

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--model` | `unet` | `unet` / `res_unet` |
| `--threshold` | `0.5` | 二值化閾值 |
| `--tta` | — | 啟用 TTA（水平、垂直、雙向翻轉取平均）|
| `--batch_size` | `16` | Batch size |

輸出 CSV 儲存至 `submission/`，命名規則：

```
submission_{model}[_tta][_th{threshold*100}].csv
# 範例：submission_unet_tta.csv、submission_res_unet_tta_th45.csv
```

---

## 本地 Kaggle 評估

用來在本地計算與 Kaggle 相同的 Dice Score，驗證提交結果。

```bash
# 工作目錄：Lab_2/

# 評估 UNet（一般推理）
python test/kaggle_test.py --model unet

# 評估 UNet（TTA）
python test/kaggle_test.py --model unet --tta

# 評估 ResNet34-UNet
python test/kaggle_test.py --model res_unet --tta

# 直接指定 CSV 路徑（優先於自動組路徑）
python test/kaggle_test.py --model unet --csv_path /path/to/your/submission.csv
```

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--model` | `unet` | `unet` / `res_unet` |
| `--tta` | — | 對應 TTA 推理結果的 CSV |
| `--csv_path` | — | 直接指定 CSV 路徑（覆蓋自動組路徑）|

首次執行會自動從 `trimaps/` 產生 Ground Truth CSV（`test/gt_test_{model}.csv`）並快取，之後直接讀取。

---

## 完整執行流程總覽

```bash
# 1. 安裝環境
pip install -r DL_Lab2_B11107122_凃岳霖/requirements.txt

# 2. 建立資料集（先把 Kaggle zip 放進 dataset/）
python DL_Lab2_B11107122_凃岳霖/src/download_dataset.py

# 3. 訓練
python DL_Lab2_B11107122_凃岳霖/src/train.py --model unet --epochs 100
python DL_Lab2_B11107122_凃岳霖/src/train.py --model res_unet --epochs 100

# 4. 找最佳 threshold
python DL_Lab2_B11107122_凃岳霖/src/evaluate.py --model unet --auto_threshold

# 5. 推理
python DL_Lab2_B11107122_凃岳霖/src/inference.py --model unet --tta --threshold 0.45

# 6. 本地評估
python test/kaggle_test.py --model unet --tta
```
