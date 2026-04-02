Lab2 Quick Guide
================================================================================
Project: Binary Semantic Segmentation (Oxford-IIIT Pet)
Models: UNet, ResNet34_UNet
Python: 3.9+
================================================================================

[1] Environment Setup (required)

cd path-to-/DL_Lab2_B11107122_凃岳霖
pip install -r requirements.txt

================================================================================
[2] Dataset Setup (required, run once)

This project already includes Kaggle split zip files in dataset/:
- nycu-2026-spring-dl-lab2-unet.zip
- nycu-2026-spring-dl-lab2-res-net-34-unet.zip

Run:
python src/download_dataset.py

After running, confirm:
- dataset/oxford-iiit-pet/images/
- dataset/oxford-iiit-pet/annotations/trimaps/
- train.txt, val.txt, test_unet.txt, test_res_unet.txt

================================================================================
[3] Training

Help:
python src/train.py --help

UNet:
python src/train.py --model unet --epochs 200 --early_stop_patience 20

ResNet34_UNet:
python src/train.py --model res_unet --epochs 200 --early_stop_patience 20

Output Best checkpoints:
- saved_models/unet_best.pth
- saved_models/res_unet_best.pth

================================================================================
[4] Validation (Auto threshold scan)

Help:
python src/evaluate.py --help

UNet:
python src/evaluate.py --model unet --auto_threshold

ResNet34_UNet:
python src/evaluate.py --model res_unet --auto_threshold

================================================================================
[5] Inference (TTA)

Help:
python src/inference.py --help

UNet:
python src/inference.py --model unet --threshold 0.5 --tta

ResNet34_UNet:
python src/inference.py --model res_unet --threshold 0.45 --tta

Output CSV:
submission/submission_*.csv

================================================================================
