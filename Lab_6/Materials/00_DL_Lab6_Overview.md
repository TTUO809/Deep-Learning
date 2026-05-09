# Lab6 - Generative Models

> **Course:** NYCU 2026 Spring Deep Learning  
> **TA:** 楊成偉 (Cheng-Wei Yang)  
> **Deadline:** 2026-05-12 23:59

---

## 目標

實作一個 **Conditional DDPM (Denoising Diffusion Probabilistic Model) 條件性去噪擴散概率模型**，依照 multi-label conditions 生成合成圖像。

- 輸入：一組 label 條件（e.g., "red sphere", "yellow cube", "gray cylinder"）
- 輸出：包含對應物件的合成圖像
- 評估：將生成圖像餵入 pretrained ResNet18 evaluator 計算 accuracy


<p align="center"> <img src="./Images/Pasted_image_20260506182943.png " /> </p>

---

## 繳交格式

- [x] 實驗報告 `.pdf`
- [x] Source code `.py`
- [x] Images 資料夾（結構如下）

```
images/
├── test/
│   ├── 0.png
│   ├── 1.png
│   └── ...
└── new_test/
    ├── 0.png
    ├── 1.png
    └── ...
```

打包成：`DL_LAB6_YourStudentID_YourName.zip`  
⚠️ **不要上傳 dataset**  
⚠️ 格式不符 **扣 5%**

---

## 相關筆記索引

- [[01_Dataset]](./01_Dataset.md)
- [[02_Model_Architecture]](./02_Model_Architecture.md)
- [[03_Training_Details]](./03_Training_Details.md)
- [[04_Evaluation]](./04_Evaluation.md)
- [[05_Scoring_Criteria]](./05_Scoring_Criteria.md)
- [[06_Implementation_Checklist]](./06_Implementation_Checklist.md)