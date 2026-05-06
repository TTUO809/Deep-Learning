# Evaluation & Output Requirements

## Pretrained Evaluator 使用規則

- 基於 **ResNet18**，檔案：`evaluator.py` + `checkpoint.pth`
- **不可修改** evaluator 的 class 與權重
- 若需額外功能，請**繼承 class**

### 呼叫方式

```python
accuracy = evaluator.eval(images, labels)
# images: (N, 3, 64, 64), normalized with Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
# labels: one-hot tensor, shape (N, 24)，e.g. [[1,1,0,...],[0,1,0,...],...]
```

### Accuracy 計算邏輯（`compute_acc`）

```
對每張圖：
  k = 該圖的 label 數量（one-hot sum）
  取 model output 的 top-k index
  與 ground truth 的 top-k index 比對
  命中數 / 總 label 數 = accuracy
```

> ⚠️ 這是 **per-label accuracy**，不是 per-image accuracy。只要有一個 label 沒生成就會扣分。

### Checkpoint 路徑設定

evaluator 預設讀取 `'./checkpoint.pth'`，若路徑不同需自行修改 line 40：

```python
checkpoint = torch.load('./checkpoint.pth')  # 修改成你的路徑
```

> **不可修改 class 與 model 權重**，但可繼承 class 加功能（如 classifier guidance）

### 圖像規格

- Normalization：`transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`
- 解析度：**64×64**（若生成其他尺寸需 resize）
- 存檔格式：**PNG**

---

## 需要生成的輸出

### 1. Synthetic Image Grid（×2 個測試集）

- 測試資料：`test.json`、`new_test.json`
- 使用 `torchvision.utils.make_grid(images)`
- 格式：**8 images/row，4 rows**（共 32 張）

```python
from torchvision.utils import make_grid, save_image
grid = make_grid(images, nrow=8)
save_image(grid, "test_grid.png")
```

### 2. Denoising Process Grid

- Label set：`["red sphere", "cyan cylinder", "cyan cube"]`
- 格式：**至少 8 張**（1 row，從 noisy → clear）
- 從 $x_T$（純 noise）到 $x_0$（生成結果）均勻取樣幾個時間點

```python
# 均勻取 8 個時間點的 snapshot
snapshots = []
for t in reversed(range(0, T, T // 8)):
    # ... denoising step ...
    snapshots.append(x_t)
grid = make_grid(snapshots, nrow=8)
```

### 3. 個別圖像存檔

```
images/
├── test/
│   ├── 0.png   ← 對應 test.json 第 0 筆
│   ├── 1.png
│   └── ...
└── new_test/
    ├── 0.png   ← 對應 new_test.json 第 0 筆
    ├── 1.png
    └── ...
```

---

## Denormalization（存圖前）

```python
def denormalize(tensor):
    # [-1, 1] → [0, 1]
    return (tensor + 1) / 2
```