"""Sampling routines: DDPM, DDIM, DDIM + Classifier Guidance.

The classifier-guidance path uses the provided ResNet18 evaluator (without modifying its
weights) — we subclass `evaluation_model` and add a `get_grad` method that backpropagates a
multi-label BCE through the (frozen) ResNet18 only with respect to the *input image*.
"""
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
    """Construct evaluator without depending on a hard-coded relative path.

    `evaluator.py` does `torch.load('./checkpoint.pth')`. To avoid editing the provided file,
    we cwd-trick: chdir to where checkpoint.pth lives, import, then chdir back.
    """
    from importlib import import_module
    cwd = os.getcwd()
    target_dir = checkpoint_path or str(_ROOT / "file")
    os.chdir(target_dir)
    try:
        ev_mod = import_module("evaluator")
        evaluator = ev_mod.evaluation_model()
    finally:
        os.chdir(cwd)
    return evaluator, ev_mod


class GuidedEvaluator:
    """Wraps the provided `evaluation_model` to expose:
        - `eval(images, labels)` — accuracy (delegates to the official one)
        - `get_grad(x, labels)` — gradient of multi-label BCE w.r.t. x
    Weights are NOT modified.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        self._inner, _ = _make_evaluator(checkpoint_path)
        self.resnet18 = self._inner.resnet18  # already on cuda + eval()
        for p in self.resnet18.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def eval(self, images: torch.Tensor, labels: torch.Tensor) -> float:
        return self._inner.eval(images, labels)

    def get_grad(self, x: torch.Tensor, labels_onehot: torch.Tensor,
                 img_size: int = 64) -> torch.Tensor:
        """Returns d(BCE) / dx, shape (B, 3, img_size, img_size).
        Caller is responsible for `requires_grad` and detach semantics; we always detach
        before returning.
        """
        x_in = x.detach().requires_grad_(True)
        # ResNet18 expects 64x64 (already correct here).
        if x_in.shape[-1] != img_size:
            x_eval = F.interpolate(x_in, size=img_size, mode="bilinear", align_corners=False)
        else:
            x_eval = x_in
        # The evaluator was loaded on cuda; make sure inputs are on the same device.
        device = next(self.resnet18.parameters()).device
        x_eval = x_eval.to(device)
        out = self.resnet18(x_eval)  # sigmoid output
        loss = F.binary_cross_entropy(out, labels_onehot.to(device).float(), reduction="sum")
        grad = torch.autograd.grad(loss, x_in)[0]
        return grad.detach()


@torch.no_grad()
def ddpm_sample(model, labels: torch.Tensor, scheduler, device,
                img_size: int = 64, num_steps: Optional[int] = None) -> torch.Tensor:
    model.eval()
    B = labels.size(0)
    x = torch.randn(B, 3, img_size, img_size, device=device)
    if num_steps is None:
        scheduler.set_timesteps(scheduler.config.num_train_timesteps)
    else:
        scheduler.set_timesteps(num_steps)
    for t in scheduler.timesteps:
        t_b = torch.full((B,), int(t), device=device, dtype=torch.long)
        eps = model(x, t_b, labels)
        x = scheduler.step(eps, t, x).prev_sample
    return x


@torch.no_grad()
def ddim_sample(model, labels: torch.Tensor, ddim_scheduler, device,
                steps: int = 100, img_size: int = 64) -> torch.Tensor:
    model.eval()
    B = labels.size(0)
    x = torch.randn(B, 3, img_size, img_size, device=device)
    ddim_scheduler.set_timesteps(steps)
    for t in ddim_scheduler.timesteps:
        t_b = torch.full((B,), int(t), device=device, dtype=torch.long)
        eps = model(x, t_b, labels)
        x = ddim_scheduler.step(eps, t, x).prev_sample
    return x


def ddim_guided_sample(model, labels: torch.Tensor, ddim_scheduler,
                       guided: GuidedEvaluator, device,
                       steps: int = 100, guidance_scale: float = 3.0,
                       img_size: int = 64) -> torch.Tensor:
    """DDIM sampling with classifier guidance (DDPM formula in eps-space).

    Applies guidance directly in eps-space using the proper DDPM formula:
        eps_guided = eps + sqrt(1-ᾱ_t)/sqrt(ᾱ_t) * s * ∇_{x0_hat} BCE

    This avoids two bugs in the x0-space approach:
      1. Per-sample gradient normalization made the perturbation ~guidance_scale per
         pixel (>> image range [-1,1]), saturating x0_guided via clamping.
      2. Converting saturated x0_guided back to eps blows up when sqrt(1-ᾱ_t)→0
         at late timesteps (t small).

    The eps-space formula naturally scales guidance down to zero at late timesteps
    (sqrt(1-ᾱ_t)/sqrt(ᾱ_t) → 0 as ᾱ_t → 1), preventing blowup and over-steering.
    """
    model.eval()
    B = labels.size(0)
    x = torch.randn(B, 3, img_size, img_size, device=device)
    ddim_scheduler.set_timesteps(steps)
    alphas_cumprod = ddim_scheduler.alphas_cumprod.to(device)

    for t in ddim_scheduler.timesteps:
        t_b = torch.full((B,), int(t), device=device, dtype=torch.long)
        with torch.no_grad():
            eps = model(x, t_b, labels)

        alpha_t = alphas_cumprod[int(t)]
        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_1m = (1.0 - alpha_t).sqrt()

        # Predict clean image; clamp to valid [-1, 1] range expected by classifier
        x0_hat = ((x - sqrt_1m * eps) / sqrt_alpha_t).clamp(-1.0, 1.0)

        # Raw gradient of BCE on predicted clean image (no per-sample normalization).
        # Raw gradients are naturally small (~1e-3/pixel for ResNet18), so the
        # timestep factor sqrt(1-ᾱ)/sqrt(ᾱ) provides sufficient implicit scaling.
        grad = guided.get_grad(x0_hat, labels, img_size=img_size)

        # DDPM classifier-guidance formula applied in eps-space:
        #   eps_guided = eps + sqrt(1-ᾱ_t)/sqrt(ᾱ_t) * s * ∇_{x0_hat} BCE
        # Equivalently, this moves x0 in the -∇BCE direction (minimizes BCE)
        # with timestep-dependent scaling: strong early, fades to 0 at t→0.
        eps_hat = eps + (sqrt_1m / sqrt_alpha_t) * guidance_scale * grad

        x = ddim_scheduler.step(eps_hat, t, x).prev_sample

    return x
