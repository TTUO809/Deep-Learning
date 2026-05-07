"""Utility helpers: EMA, image (de)normalization, seed."""
from __future__ import annotations

import copy
import os
import random
from typing import Iterable

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    """Exponential moving average of model parameters (decay=0.9999 default).

    `apply_to(target_model)` copies the EMA weights into a target model;
    `state_dict()` / `load_state_dict()` lets us checkpoint the shadow weights.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def apply_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1] (clamped)."""
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
