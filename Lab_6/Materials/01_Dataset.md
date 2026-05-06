# Dataset - iCLEVR

## 提供檔案

|檔案|說明|
|---|---|
|`readme.txt`|Dataset 詳細說明|
|`train.json`|訓練資料標籤|
|`test.json`|測試資料 1|
|`new_test.json`|測試資料 2|
|`object.json`|物件類別字典（24 classes）|
|`evaluator.py`|Pretrained evaluator 腳本|
|`checkpoint.pth`|Evaluator 權重|
|`iclevr.zip`|圖像資料集（從 Google Drive 下載）|

> `file.zip` 從 e3 / NTU Cool 下載；`iclevr.zip` 從開源 [Google Drive](https://drive.google.com/file/d/1Y-N1O0qltVtYMq95CAzJ1s-y-NM0qd_O/view) 下載

---

## 資料格式

### object.json

- 24 個物件 = **3 種形狀 × 8 種顏色**
- Label → index 對照表

### train.json

- **18,009** 筆訓練資料
- 格式：dict，key 為檔名，value 為 label list
- 每張圖包含 **1 到 3 個物件**

```json
{
  "CLEVR_train_001032_0.png": ["yellow sphere"],
  "CLEVR_train_001032_1.png": ["yellow sphere", "gray cylinder"],
  "CLEVR_train_001032_2.png": ["yellow sphere", "gray cylinder", "purple cube"]
}
```

### test.json / new_test.json

- 各 **32** 筆測試資料
- 格式：**list**（非 dict），每個 element 為 label list

```json
[
  ["gray cube"],
  ["red cube"],
  ["blue cube", "green cube"]
]
```

> ⚠️ test.json 格式與 train.json **不同**（list vs dict），DataLoader 需分開處理

---

## Data Loader 實作要點

- [ ] 讀取 `train.json`，對應圖像路徑與 multi-label
- [ ] 讀取 `test.json` / `new_test.json`（list 格式，無圖像路徑）
- [ ] 將 labels 轉換為 **one-hot vector**（長度 = 24）
- [ ] 圖像 normalization：

```python
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
```

- [ ] 定義 **multi-label condition embedding** 方法（e.g., linear projection of one-hot）

---

## 注意事項

- 除提供的檔案外，**不可使用其他訓練資料**（e.g., background image）
- Evaluator 輸入解析度固定為 **64×64**（原始圖為 32×32 upsample 而來）
- 若自訂輸出解析度，需在評估前 resize