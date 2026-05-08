"""===============================================================
【 Utility functions for training and evaluation. 】

Includes random seed initialization for reproducibility, Exponential Moving Average (EMA) 
for model weights, and image denormalization helpers.
==============================================================="""

from __future__ import annotations

import copy
import os
import random
from typing import Iterable

import numpy as np
import torch
from diffusers import DDIMScheduler, DDPMScheduler


def set_seed(seed: int = 42) -> None:
    """---------------------------------------------------------------
    【 Sets the random seed for deterministic execution. 】
    
    Ensures reproducibility across Python, NumPy, and PyTorch. Also configures
    cuDNN to deterministic mode to prevent precision errors during backward pass.
    
    Args:
        seed (int): The seed value to use. Defaults to 42.
    ---------------------------------------------------------------"""
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EMA:
    """---------------------------------------------------------------
    【 Exponential Moving Average (EMA) for model parameters. 】
    
    Maintains a shadow copy of the model's weights that updates slowly over time,
    leading to more stable and higher quality generation results.
    
    Attributes:
        decay (float): The decay factor for the moving average.
        shadow (dict): The dictionary containing the shadow weights.
    ---------------------------------------------------------------"""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay

        # 1. detach()：斷開計算圖，避免追蹤梯度。
        # 2. clone()：深拷貝，確保影子權重的記憶體獨立。
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()    # 禁用梯度計算，節省記憶體並加速。
    def update(self, model: torch.nn.Module) -> None:
        """Updates the EMA shadow weights based on the current model weights."""
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                # 1. In-place 操作 (mul_, add_)：直接修改記憶體數值以節省資源。
                # 2. EMA 公式：Shadow = Shadow * decay + New_Weight * (1 - decay)。
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                # 非浮點數（如訓練步數）不作平滑，直接覆蓋。
                self.shadow[k].copy_(v)

    def apply_to(self, model: torch.nn.Module) -> None:
        """Loads the shadow weights into the target model."""
        # 將影子權重覆蓋至目標模型（用於推論與評估）。
        model.load_state_dict(self.shadow)

    def state_dict(self) -> dict:
        # 匯出影子權重（用於 Checkpoint 存檔）。
        return self.shadow

    def load_state_dict(self, sd: dict) -> None:
        # 載入並深拷貝權重，確保記憶體獨立（用於 Checkpoint 讀檔）。
        self.shadow = {k: v.clone() for k, v in sd.items()}


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """---------------------------------------------------------------
    【 Denormalizes image tensors from [-1, 1] to [0, 1]. 】
    
    Args:
        x (torch.Tensor): Normalized tensor.
        
    Returns:
        torch.Tensor: Denormalized tensor clamped to [0, 1].
    ---------------------------------------------------------------"""

    # 1. clamp 把小於 -1.0 的變成 -1.0，大於 1.0 的變成 1.0，把數值「夾」在 [-1.0, 1.0] 之間。
    # 2. 將範圍從 [-1.0, 1.0] 向上平移，變成 [0.0, 2.0]。
    # 3. 將範圍從 [0.0, 2.0]  縮小一半，變成 [0.0, 1.0]。
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


def make_schedulers(beta_schedule: str) -> tuple[DDIMScheduler, DDPMScheduler]:
    """---------------------------------------------------------------
    【 Returns (DDIMScheduler, DDPMScheduler) configured with the given beta schedule. 】
    ---------------------------------------------------------------"""
    kw = (
        dict(beta_schedule="linear", beta_start=1e-4, beta_end=0.02)
        if beta_schedule == "linear"
        else dict(beta_schedule="squaredcos_cap_v2")
    )
    return (
        DDIMScheduler(num_train_timesteps=1000, **kw),
        DDPMScheduler(num_train_timesteps=1000, **kw),
    )


def count_parameters(model: torch.nn.Module) -> int:
    """---------------------------------------------------------------
    【 Counts the number of trainable parameters in a PyTorch model. 】
    ---------------------------------------------------------------"""

    # 1. model.parameters()：取出模型中所有的參數張量。
    # 2. if p.requires_grad：篩選條件，只保留需要計算梯度（可被訓練）的參數。
    # 3. p.numel()：計算該張量內的元素總數，最後透過 sum() 進行總和。
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
