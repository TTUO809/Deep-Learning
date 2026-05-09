"""===============================================================
【 Sampling routines: DDPM, DDIM, DDIM + Classifier Guidance. 】

The classifier-guidance path uses the provided ResNet18 evaluator (without modifying its
weights) — we subclass `evaluation_model` and add a `get_grad` method that backpropagates a
multi-label BCE through the (frozen) ResNet18 only with respect to the *input image*.
==============================================================="""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# Allow `from evaluator import evaluation_model` even when running from src/.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "file"))


def _make_evaluator(checkpoint_path: Optional[str] = None):
    """---------------------------------------------------------------
    【 Dynamic Evaluator Loader 】
    
    Dynamically imports and instantiates the evaluator model by temporarily changing the working directory, avoiding hard-coded relative path issues.
    
    Input: checkpoint_path (Optional[str]) -> Output: Tuple of (evaluator_instance, imported_module)
    ---------------------------------------------------------------"""

    from importlib import import_module

    # 1. 記錄當前的工作目錄，以便在匯入完成後能安全地恢復。
    cwd = os.getcwd()

    # 2. 決定目標路徑並切換目錄，確保模型內部的 torch.load 能找到正確的相對路徑檔案
    target_dir = checkpoint_path or str(_ROOT / "file")
    os.chdir(target_dir)

    try:
        # 3. 動態匯入 evaluator 模組並實例化評估模型。
        ev_mod = import_module("evaluator")
        evaluator = ev_mod.evaluation_model()
    finally:
        # 4. 無論匯入成功或發生例外，都強制將工作目錄切換回原處，防止影響系統其他運作。
        os.chdir(cwd)
    
    return evaluator, ev_mod


class GuidedEvaluator:
    """---------------------------------------------------------------
    【 Guided Evaluator Wrapper 】
    
    Wraps the provided evaluation model to expose accuracy metrics and calculate the gradient of the input image with respect to the multi-label BCE loss for classifier guidance.
    
    Input shape: 
        x: (B, 3, H, W)
        labels: (B, num_classes)
    Output shape:
        grad: (B, 3, H, W)
    ---------------------------------------------------------------"""

    def __init__(self, checkpoint_path: Optional[str] = None):
        # 1. 實例化內部評估模型。
        self._inner, _ = _make_evaluator(checkpoint_path)
        self.resnet18 = self._inner.resnet18  # already on cuda + eval()

        # 2. 凍結權重：分類器引導 (Classifier Guidance) 只需要對「輸入圖片」求梯度，
        #    因此我們將模型所有參數的 requires_grad 設為 False，藉此大幅節省記憶體並加速反向傳播。
        for p in self.resnet18.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def eval(self, images: torch.Tensor, labels: torch.Tensor) -> float:
        # 1. 代理呼叫原本模型的評估函數，並確保在此過程中不計算梯度。
        return self._inner.eval(images, labels)

    def get_grad(self, x: torch.Tensor, labels_onehot: torch.Tensor,
                 img_size: int = 64) -> torch.Tensor:
        """
        【 Gradient Calculator for Guidance 】
        
        Computes the gradient of the Binary Cross Entropy (BCE) loss with respect to the input image, steering the generation process towards the target labels.
        
        Input shape: 
            x: (B, 3, H_in, W_in)
            labels_onehot: (B, num_classes)
        Output shape: (B, 3, H_in, W_in)
        """

        # 1. 從計算圖中分離輸入 x，並宣告對其追蹤梯度 (requires_grad_(True))，這是因為 x 是我們要優化的對象。
        x_in = x.detach().requires_grad_(True)

        # 2. 空間維度對齊：ResNet18 期望固定尺寸的輸入 (如 64x64)。若輸入尺寸不符，則使用雙線性插值 (bilinear) 調整。
        if x_in.shape[-1] != img_size:
            x_eval = F.interpolate(x_in, size=img_size, mode="bilinear", align_corners=False)
        else:
            x_eval = x_in

        # 3. 設備對齊：確保輸入 Tensor 與 ResNet18 模型處於同一個運算設備 (如 CUDA) 上。
        device = next(self.resnet18.parameters()).device
        x_eval = x_eval.to(device)

        # 4. 前向傳播：將圖片送入 ResNet18，輸出各類別的機率值 (已經過 Sigmoid)。
        out = self.resnet18(x_eval)

        # 5. 計算引導損失：使用 BCE Loss 來衡量模型預測與目標標籤 (labels_onehot) 的差距。使用 reduction="sum" 以放大梯度訊號。
        loss = F.binary_cross_entropy(out, labels_onehot.to(device).float(), reduction="sum")

        # 6. 反向傳播並提取梯度：計算 Loss 對輸入圖片 x_in 的梯度，引導圖片在像素空間中往「Loss 變小的方向」移動。
        grad = torch.autograd.grad(loss, x_in)[0]

        # 7. 回傳分離後的梯度，避免外部優化器意外串聯計算圖導致記憶體洩漏。
        return grad.detach()


@torch.no_grad()
def ddpm_sample(model, labels: torch.Tensor, scheduler, device,
                img_size: int = 64, num_steps: Optional[int] = None) -> torch.Tensor:
    """---------------------------------------------------------------
    【 Standard DDPM Sampling Loop 】
    
    Generates images from pure noise using the Denoising Diffusion Probabilistic Models (DDPM) standard Markov chain.
    
    Input shape:
        labels: (B, num_classes)
    Output shape: (B, 3, img_size, img_size)
    ---------------------------------------------------------------"""

    # 1. 設定模型為評估模式，關閉 Dropout 與 BatchNorm 的更新。
    model.eval()
    B = labels.size(0)

    # 2. 初始化潛在空間：生成完全隨機的高斯雜訊 x_T。
    x = torch.randn(B, 3, img_size, img_size, device=device)

    # 3. 初始化排程器步數設定。
    if num_steps is None:
        scheduler.set_timesteps(scheduler.config.num_train_timesteps)
    else:
        scheduler.set_timesteps(num_steps)

    # 4. 逐步去噪迴圈：沿著時間步 t 從 T 倒退回 0。
    for t in scheduler.timesteps:
        # 5. 擴張時間步 t 以匹配 batch size，確保模型能接收正確形狀的時間嵌入 (Time Embedding)。
        t_b = torch.full((B,), int(t), device=device, dtype=torch.long)
        
        # 6. 模型預測：給定當前雜訊圖片 x、時間 t 與條件 labels，預測出被加入的雜訊 eps。
        eps = model(x, t_b, labels)
        
        # 7. 更新步驟：透過 scheduler 依照 DDPM 公式計算上一個時間步 (t-1) 的圖片狀態。
        x = scheduler.step(eps, t, x).prev_sample

    return x


@torch.no_grad()
def ddim_sample(model, labels: torch.Tensor, ddim_scheduler, device,
                steps: int = 100, img_size: int = 64) -> torch.Tensor:
    """---------------------------------------------------------------
    【 Standard DDIM Sampling Loop 】
    
    Generates images using the Denoising Diffusion Implicit Models (DDIM) approach for accelerated, deterministic sampling.
    
    Input shape:
        labels: (B, num_classes)
    Output shape: (B, 3, img_size, img_size)
    ---------------------------------------------------------------"""

    # 1. 設置模型為評估模式。
    model.eval()
    B = labels.size(0)
    
    # 2. 建立初始高斯雜訊 (物理意義：擴散過程最終點的純雜訊狀態)。
    x = torch.randn(B, 3, img_size, img_size, device=device)
    
    # 3. 設置 DDIM 壓縮步數，允許跳步採樣以加速生成。
    ddim_scheduler.set_timesteps(steps)
    
    # 4. 進行確定性 (Deterministic) 去噪迴圈。
    for t in ddim_scheduler.timesteps:
        # 5. 建構批次化的時間張量。
        t_b = torch.full((B,), int(t), device=device, dtype=torch.long)
        
        # 6. 預測雜訊。
        eps = model(x, t_b, labels)
        
        # 7. 使用 DDIM 演算法更新至 t-1 步。由於 DDIM 的非馬可夫性質，此步驟無須加入隨機雜訊 (eta=0)。
        x = ddim_scheduler.step(eps, t, x).prev_sample
        
    return x


def ddim_guided_sample(model, labels: torch.Tensor, ddim_scheduler,
                       guided: GuidedEvaluator, device,
                       steps: int = 100, guidance_scale: float = 3.0,
                       img_size: int = 64) -> torch.Tensor:
    """---------------------------------------------------------------
    【 Universal Guided DDIM Sampling 】
    
    Performs DDIM sampling while using a classifier to inject gradients into the predicted clean image (x0-space), guiding the generation towards specific labels.
    
    Input shape:
        labels: (B, num_classes)
    Output shape: (B, 3, img_size, img_size)
    ---------------------------------------------------------------"""
    # 1. 確保生成模型進入評估模式。
    model.eval()
    B = labels.size(0)
    
    # 2. 產生初始純粹高斯雜訊 x_T。
    x = torch.randn(B, 3, img_size, img_size, device=device)
    
    # 3. 初始化 DDIM 的時間步距陣列，並將累積乘積的 alpha 常數表提取至計算設備上。
    ddim_scheduler.set_timesteps(steps)
    alphas_cumprod = ddim_scheduler.alphas_cumprod.to(device)

    # 4. 啟動跳步降噪的 DDIM 迴圈。
    for t in ddim_scheduler.timesteps:
        # 5. 擴展當前時間步 t 至整個 Batch。
        t_b = torch.full((B,), int(t), device=device, dtype=torch.long)
        
        # 6. 阻斷梯度：讓模型預測當前的雜訊 eps。此處不需要模型權重的梯度，僅作推論。
        with torch.no_grad():
            eps = model(x, t_b, labels)
        
        # 7. 提取當前 timestep 對應的動力學參數 (alpha_t 及其平方根)。
        alpha_t = alphas_cumprod[int(t)]
        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_1m = (1.0 - alpha_t).sqrt()

        # 8. 物理意義轉換：預測乾淨的 x0_hat。透過當下的 x 去除預測出的雜訊 eps 來估計原圖。
        #    利用 clamp 限制在 [-1.0, 1.0] 範圍內，符合 Evaluator 期望的真實圖片物理極限。
        x0_hat = ((x - sqrt_1m * eps) / sqrt_alpha_t).clamp(-1.0, 1.0)
        
        # 9. 梯度引導：讓外部 Evaluator (如 ResNet18) 觀察這張預測出的「乾淨原圖」x0_hat，
        #    並針對目標標籤計算出 BCE 損失的梯度。
        grad = guided.get_grad(x0_hat, labels, img_size=img_size)

        # 10. 對每張圖片進行梯度 Norm Clipping：多物體圖片的 BCE(sum) 梯度比單物體大 N 倍，
        #     若不限制會過度推動 x0_guided 飽和至 ±1 造成雪花屏。
        #     只在 norm > 1.0 時才縮放（保留小梯度的自然強度，避免放大雜訊）。
        B_cur = grad.shape[0]
        grad_norm = grad.view(B_cur, -1).norm(dim=1).view(B_cur, 1, 1, 1).clamp(min=1e-8)
        grad = grad * (1.0 / grad_norm).clamp(max=1.0)  # clip to max norm=1.0, never amplify

        # 11. 直接在 x0 像素空間進行引導！
        #     目標是讓 Loss (BCE) 變小，所以是「減去」梯度方向乘以 guidance_scale (引導強度)。
        x0_guided = x0_hat - guidance_scale * grad
        
        # 12. 【終極防護網】確保引導後的像素值絕對不會超出 [-1, 1] 的物理極限。
        x0_guided = x0_guided.clamp(-1.0, 1.0)

        # 13. 逆向推導：將引導完畢的乾淨圖片，安全地反推回當前步驟需要的等效雜訊 eps_hat。
        #     這使得後續的 DDIM 更新能吃到被修正過的方向。
        eps_hat = (x - sqrt_alpha_t * x0_guided) / sqrt_1m
        
        # 14. 正常往下走：使用被梯度干涉過的 eps_hat 執行標準 DDIM 演算法更新至 t-1 狀態。
        x = ddim_scheduler.step(eps_hat, t, x).prev_sample

    return x
