# Lab 5 — value-Based Reinforcement Learning

## 從零開始 (訓練 ➔ 評估 ➔ 錄影 ➔ 打包)

### 直接執行即可

```bash
    bash run_task1.sh
    bash run_task2.sh
    bash run_task3.sh
```

### 跳過訓練，直接對現有模型「評估 + 錄影」

```bash
    # Task 1 & 2：傳入具體的 .pt 檔案路徑
    bash run_task1.sh ../LAB5_B11107122_task1.pt
    bash run_task2.sh ../LAB5_B11107122_task2.pt

    # Task 3：傳入整個資料夾路徑 (因為要同時處理 best 跟 milestone)
    bash run_task3.sh ../LAB5_B11107122_task3_best.pt
```

### 只要看分數，不要浪費時間錄影

```bash
    bash run_task1.sh ../LAB5_B11107122_task1.pt --eval-only
    bash run_task2.sh ../LAB5_B11107122_task2.pt --eval-only
    bash run_task3.sh ../LAB5_B11107122_task3_best.pt --eval-only
```

### 純錄影模式 (只要生影片，不跑 20 seeds 評估)

```bash
    bash run_task1.sh ../LAB5_B11107122_task1.pt --record-only
    bash run_task2.sh ../LAB5_B11107122_task2.pt --record-only
    bash run_task3.sh ../ --record-only
```

### 指定測驗特定的 Milestone (僅限 Task 3)

```bash
    bash run_task3.sh ../ --eval-only --milestone 600000
    bash run_task3.sh ../ --eval-only --milestone 1000000
    bash run_task3.sh ../ --eval-only --milestone 1500000
    bash run_task3.sh ../ --eval-only --milestone 2000000
    bash run_task3.sh ../ --eval-only --milestone 2500000
    bash run_task3.sh ../LAB5_B11107122_task3_600000.pt --eval-only
    bash run_task3.sh ../LAB5_B11107122_task3_1000000.pt --eval-only
    bash run_task3.sh ../LAB5_B11107122_task3_1500000.pt --eval-only
    bash run_task3.sh ../LAB5_B11107122_task3_2000000.pt --eval-only
    bash run_task3.sh ../LAB5_B11107122_task3_2500000.pt --eval-only
```