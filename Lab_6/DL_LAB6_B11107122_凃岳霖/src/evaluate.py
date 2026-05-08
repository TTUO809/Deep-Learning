"""===============================================================
【 Evaluate a trained checkpoint on test.json / new_test.json. 】

Outputs:
  images/<split>/<i>.png      — individual generated images (denormalized to [0,1])
  results/<split>_grid.png    — 8x4 grid
  Prints accuracy from the official ResNet18 evaluator.
==============================================================="""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from model import ConditionalUNet
from sampler import GuidedEvaluator, ddim_guided_sample, ddim_sample, ddpm_sample
from utils import denormalize, make_schedulers, set_seed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def parse_args():
    """---------------------------------------------------------------
    【 Argument Parser for Evaluation 】
    ---------------------------------------------------------------"""

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--obj_json"     , default=str(ROOT / "file" / "objects.json"))
    p.add_argument("--test_json"    , default=str(ROOT / "file" / "test.json"))
    p.add_argument("--new_test_json", default=str(ROOT / "file" / "new_test.json"))
    p.add_argument("--out_dir"      , default=str(ROOT / "images"))
    p.add_argument("--grid_dir"     , default=str(ROOT / "results"))
    p.add_argument("--sampler", choices=["ddpm", "ddim", "ddim_guided"],
                   default="ddim_guided")
    p.add_argument("--steps"            , type=int  , default=100)
    p.add_argument("--guidance_scale"   , type=float, default=3.0)
    p.add_argument("--beta_schedule", choices=["linear", "squaredcos_cap_v2"],
                   default="squaredcos_cap_v2")
    p.add_argument("--use_ema"  , action="store_true", default=True)
    p.add_argument("--no_ema"   , dest="use_ema", action="store_false")
    p.add_argument("--seed"     , type=int  , default=42)
    p.add_argument("--batch"    , type=int  , default=32, help="generate N images per pass")
    p.add_argument("--tag"                  , default="", help="tag appended to output dirs (for ablation runs)")
    return p.parse_args()


def labels_list_to_onehot(label_lists: list[list[str]], obj_map: dict[str, int], device: torch.device) -> torch.Tensor:
    """---------------------------------------------------------------
    【 Labels to One-Hot Encoding 】
    
    Converts a list of string labels into a batched multi-hot / one-hot tensor representation for neural network conditioning.
    
    Input shape: 
        label_lists: List of length N (batch size), where each element is a List of string classes.
    Output shape: (N, num_classes)
    ---------------------------------------------------------------"""

    n_cls = len(obj_map)
    n_label = len(label_lists)
    
    # 1. 初始化全零張量，形狀為 (N, 總類別數)，準備作為神經網路的條件輸入。
    out = torch.zeros(n_label, n_cls, device=device)
    
    # 2. 雙層迴圈遍歷：將樣本中出現的類別索引位置設為 1.0 (Multi-hot 編碼的物理意義：代表該圖片包含這些特定的物件)。
    for i, lbs in enumerate(label_lists):
        for lb in lbs:
            out[i, obj_map[lb]] = 1.0
            
    return out


def generate_for_split(model, labels_onehot: torch.Tensor, sampler: str, args, device: torch.device, 
                       ddim_sched, ddpm_sched, guided=None) -> torch.Tensor:
    """---------------------------------------------------------------
    【 Batched Image Generation Loop 】
    
    Generates images in chunks to prevent Out-Of-Memory (OOM) errors, routing the process to the specified sampling algorithm.
    
    Input shape: 
        labels_onehot: (N_total, num_classes)
    Output shape: (N_total, 3, H, W)
    ---------------------------------------------------------------"""

    images = []
    n = labels_onehot.size(0)
    
    # 1. 批次處理：依據 args.batch 切割總樣本數 N，避免 GPU 記憶體一次性塞滿 (OOM)。
    for start in range(0, n, args.batch):
        chunk = labels_onehot[start:start + args.batch] # chunk shape: (B, num_classes)
        
        # 2. 根據指定的採樣策略呼叫對應的生成函數。
        if sampler == "ddpm":
            img = ddpm_sample(model, chunk, ddpm_sched, device,
                              num_steps=args.steps if args.steps != 1000 else None)
        elif sampler == "ddim":
            img = ddim_sample(model, chunk, ddim_sched, device, steps=args.steps)
        elif sampler == "ddim_guided":
            # 確保引導評估器 (GuidedEvaluator) 已被實例化。
            assert guided is not None
            img = ddim_guided_sample(model, chunk, ddim_sched, guided, device,
                                     steps=args.steps, guidance_scale=args.guidance_scale)
        else:
            raise ValueError(sampler)
            
        images.append(img)
        
    # 3. 將所有小批次的生成圖片沿著 Batch 維度 (dim=0) 拼接起來，恢復成 (N_total, 3, H, W)。
    return torch.cat(images, dim=0)


def main():
    """---------------------------------------------------------------
    【 Main Evaluation Pipeline 】
    
    Orchestrates the model loading, multi-split dataset processing, image generation, metric calculation (accuracy via ResNet18), and grid visualization.
    ---------------------------------------------------------------"""

    # 1. 系統初始化。
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 權重加載 (Checkpoint Loading)。
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = ConditionalUNet().to(device)
    
    # 3. EMA 權重優先：若訓練時有保留指數移動平均 (EMA) 的權重，強烈建議使用，因為它的生成效果通常更平滑且品質更高。
    if args.use_ema and "ema" in ck:
        model.load_state_dict(ck["ema"])
        print("[ckpt] loaded EMA weights")
    else:
        model.load_state_dict(ck["model"])
        print("[ckpt] loaded raw model weights")
    model.eval() # 關閉 Dropout/BatchNorm 的隨機性。

    # 4. 準備排程器與評估器。
    ddim_sched, ddpm_sched = make_schedulers(args.beta_schedule)
    
    # 不論是否使用 ddim_guided 採樣，我們最終都需要 GuidedEvaluator 裡面的 ResNet18 來計算 Accuracy。
    guided = GuidedEvaluator()
    evaluator = guided

    # 5. 讀取類別對應表 (Object Map)，將字串轉換為類別索引。
    with open(args.obj_json) as f:
        obj_map = json.load(f)

    splits = [
        ("test", args.test_json),
        ("new_test", args.new_test_json),
    ]

    summary = {}
    
    # 6. 依序對不同資料分割 (split) 進行評估。
    for split_name, json_path in splits:
        with open(json_path) as f:
            data = json.load(f)
            
        # 6.1. 將文字標籤轉為模型看得懂的 Multi-hot Tensor。
        labels_onehot = labels_list_to_onehot(data, obj_map, device)
        print(f"[{split_name}] generating {len(data)} images ...")
        
        # 6.2. 執行生成。images shape: (N_total, 3, H, W)，此時的值域約在 [-1, 1]。
        images = generate_for_split(
            model, labels_onehot, args.sampler, args, device, ddim_sched, ddpm_sched, guided,
        )

        # 7. 儲存單張圖片與視覺化網格 (Grid)。
        tag = f"_{args.tag}" if args.tag else ""
        out_dir = os.path.join(args.out_dir, split_name + tag)
        os.makedirs(out_dir, exist_ok=True)
        
        # 7.1. 反正規化：將圖片從 [-1, 1] 的數學空間拉回 [0, 1] 的物理顯示空間。
        denorm = denormalize(images)
        
        for i in range(denorm.size(0)):
            save_image(denorm[i], os.path.join(out_dir, f"{i}.png"))
            
        os.makedirs(args.grid_dir, exist_ok=True)
        grid_path = os.path.join(args.grid_dir, f"{split_name}{tag}_grid.png")
        
        # 7.2. 將 N 張圖片拼成 8 行 (nrow=8) 的預覽網格。
        save_image(make_grid(denorm, nrow=8), grid_path)
        print(f"[{split_name}] saved -> {out_dir}/ , grid -> {grid_path}")

        # 8. 計算分類準確率 (Accuracy)。
        # Evaluator 期望的輸入域是標準的 [-1, 1] (等同於 transforms.Normalize(0.5, 0.5))。
        # clamp 確保生成過程中微小的數值溢出被物理限縮。
        eval_images = images.clamp(-1.0, 1.0)
        
        # 8.1. 將圖片送入預先訓練好的 ResNet18 檢驗是否成功生成了指定的物件。
        acc = evaluator.eval(eval_images.cuda(), labels_onehot.cuda())
        print(f"[{split_name}] accuracy = {acc:.4f}")
        summary[split_name] = acc

    # 9. 印出最終評估報表。
    print("\n====== SUMMARY ======")
    for k, v in summary.items():
        print(f"  {k:>10s}: {v:.4f}")
    print("="*21 + "\n")


if __name__ == "__main__":
    main()