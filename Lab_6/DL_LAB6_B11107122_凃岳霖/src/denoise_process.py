"""===============================================================
【 Generate the required denoising-process visualization for the report. 】

Label set (per spec): ["red sphere", "cyan cylinder", "cyan cube"]
Output: a 1-row grid of >=8 snapshots from x_T (noise) -> x_0 (clean).
==============================================================="""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from model import ConditionalUNet
from sampler import GuidedEvaluator
from utils import denormalize, make_schedulers, set_seed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# 預設指定的生成物件標籤，用於展示去噪過程
DEFAULT_LABELS = ["red sphere", "cyan cylinder", "cyan cube"]


def parse_args():
    """---------------------------------------------------------------
    【 Argument Parser for Denoise Visualization 】
    
    Parses command-line arguments specific to generating a visual timeline of the diffusion denoising process.

    ---------------------------------------------------------------"""

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt"             , required=True)
    p.add_argument("--obj_json"         , default=str(ROOT / "file" / "objects.json"))
    p.add_argument("--out"              , default=str(ROOT / "results" / "denoise_process.png"))
    p.add_argument("--steps"            , type=int, default=1000,
                   help="diffusion steps for the visualization (DDPM uses scheduler default)")
    p.add_argument("--snapshots"        , type=int, default=8,
                   help="number of denoising snapshots to display (>=8 by spec)")
    p.add_argument("--scheduler"        , choices=["ddpm", "ddim"], default="ddpm")
    p.add_argument("--beta_schedule"    , choices=["linear", "squaredcos_cap_v2"],
                   default="squaredcos_cap_v2")
    p.add_argument("--guidance_scale"   , type=float, default=0.0,
                   help="if >0 and scheduler==ddim, apply classifier guidance")
    p.add_argument("--seed"         , type=int, default=42)
    p.add_argument("--use_ema"      , action="store_true", default=True)
    return p.parse_args()



def main():
    """---------------------------------------------------------------
    【 Denoising Process Visualization Pipeline 】
    
    Generates an image from pure noise to a clean output, saving specific intermediate snapshots along the Markov chain to visualize the model's generative process.
    
    Input: None -> Output: None (Saves a grid image to disk)
    ---------------------------------------------------------------"""
    
    # 1. 系統初始化與設定設備。
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 權重加載 (Checkpoint Loading)。
    #    優先嘗試加載 EMA (指數移動平均) 權重，因為 EMA 能提供更平滑、高品質的生成結果。
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = ConditionalUNet().to(device)
    if args.use_ema and "ema" in ck:
        model.load_state_dict(ck["ema"])
        print("[ckpt] loaded EMA weights")
    else:
        model.load_state_dict(ck["model"])
    model.eval()

    # 3. 準備條件標籤 (Label One-Hot Encoding)。
    #    讀取字典映射表，將目標 `DEFAULT_LABELS` 轉換為 Multi-hot 張量。
    #    labels shape: (1, num_classes)
    with open(args.obj_json) as f:
        obj_map = json.load(f)
    labels = torch.zeros(1, len(obj_map), device=device)
    for lb in DEFAULT_LABELS:
        labels[0, obj_map[lb]] = 1.0

    # 4. 初始化排程器並設置總時間步數。
    #    timesteps 會是一個從大到小的陣列 (例如從 999 遞減至 0)。
    ddim_sched, ddpm_sched = make_schedulers(args.beta_schedule)
    sched = ddpm_sched if args.scheduler == "ddpm" else ddim_sched
    sched.set_timesteps(args.steps)
    timesteps = sched.timesteps  
    n = len(timesteps)
    
    # 5. 計算「快照」索引 (Snapshot Indices)。
    #    為了在畫面上均勻展示去噪過程，依據總步數與使用者要求的快照數 (args.snapshots)，
    #    等比例切分出需要記錄圖片狀態的時間步索引。
    snap_idx = set([int(round(i * (n - 1) / max(args.snapshots - 1, 1)))
                    for i in range(args.snapshots)])

    # 6. 配置分類器引導 (Classifier Guidance)。
    #    若啟用引導，載入輔助的 ResNet18 評估器，並提取 alphas_cumprod 備用。
    guided = None
    if args.guidance_scale > 0 and args.scheduler == "ddim":
        guided = GuidedEvaluator()
        alphas_cumprod = sched.alphas_cumprod.to(device)

    # 7. 產生初始的高斯雜訊 x_T。
    #    物理意義：擴散模型生成的起點，完全無意義的電視雪花畫面。
    #    x shape: (1, 3, 64, 64)
    x = torch.randn(1, 3, 64, 64, device=device)
    snapshots = []
    snapshots.append(x.clone())  # 保存第一張純雜訊快照

    # 8. 開始逐步去噪迴圈 (Denoising Loop)。
    for i, t in enumerate(timesteps):
        # 建立時間嵌入所需的一維 Tensor。
        t_b = torch.full((1,), int(t), device=device, dtype=torch.long)
        
        if guided is not None:
            # 8.1 有分類器引導的路線。
            with torch.no_grad():
                eps = model(x, t_b, labels)
            
            # 使用評估器計算目前狀態 x 相對目標標籤的 BCE 梯度。
            grad = guided.get_grad(x, labels)
            
            # 梯度注入：根據公式修改預測出來的雜訊 eps。
            # 物理意義：將模型原本預測的「一般去噪方向」，強行往「更符合分類器認知」的方向推動。
            eps_hat = eps - args.guidance_scale * (1.0 - alphas_cumprod[int(t)]).sqrt() * grad
            
            # 透過修改過的 eps_hat 反推上一個時間步的圖片狀態。
            x = sched.step(eps_hat, t, x).prev_sample
        else:
            # 8.2 標準的去噪路線 (無外加梯度干涉)。
            with torch.no_grad():
                eps = model(x, t_b, labels)
            x = sched.step(eps, t, x).prev_sample
            
        # 9. 捕捉並儲存中繼狀態。
        #    如果目前的迴圈進度落在我們算好的 snap_idx 中 (且不是最後一步，因為最後一步會另外加)，則複製並記錄。
        if i in snap_idx and i != n - 1:
            snapshots.append(x.clone())
            
    # 儲存最後生成的乾淨原圖 x_0。
    snapshots.append(x.clone())

    # 10. 確保快照數量精確等於 args.snapshots。
    #     如果因為進位問題多記錄了，進行均勻的抽樣過濾，同時強制保留頭 (純雜訊) 與尾 (最終成品)。
    if len(snapshots) > args.snapshots:
        idx = [round(i * (len(snapshots) - 1) / (args.snapshots - 1))
               for i in range(args.snapshots)]
        snapshots = [snapshots[i] for i in idx]

    # 11. 影像後處理與儲存。
    #     將多個張量 (N, 3, 64, 64) 沿著第 0 維度拼接。
    #     使用 denormalize 將數值拉回 [0, 1] 顯示範圍，然後透過 make_grid 拼成單列長條圖 (nrow=快照總數)。
    grid = make_grid(denormalize(torch.cat(snapshots, dim=0)), nrow=len(snapshots))
    
    #     確保目標資料夾存在後，寫入硬碟。
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, args.out)
    print(f"[denoise] saved {len(snapshots)} snapshots -> {args.out}")


if __name__ == "__main__":
    main()