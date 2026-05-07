"""Conditional UNet for Lab 6 DDPM.

Design choices (combined from P1 DDPM + P2 ADM/Beat-GANs):
  * Sinusoidal time embedding -> MLP.
  * Multi-label one-hot -> linear MLP label embedding.
  * AdaGN ResBlocks: condition (time||label) is projected to per-channel scale/shift on the
    second GroupNorm of every ResBlock.
  * Multi-head self-attention at resolutions 32/16/8 with 64 channels per head.
  * BigGAN-style up/downsampling (interp+conv / strided conv).
  * Channels: base=128, mults=(1,2,2,4)  ->  128 / 256 / 256 / 512.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int, num_groups: int = 32) -> nn.GroupNorm:
    g = min(num_groups, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeEmbeddingMLP(nn.Module):
    def __init__(self, base_dim: int = 128, time_emb_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalTimeEmbedding(base_dim),
            nn.Linear(base_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


class LabelEmbedding(nn.Module):
    """Multi-label one-hot (B, num_classes) -> (B, label_emb_dim)."""

    def __init__(self, num_classes: int = 24, label_emb_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes, label_emb_dim),
            nn.SiLU(),
            nn.Linear(label_emb_dim, label_emb_dim),
        )

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.net(labels)


class ResBlock(nn.Module):
    """AdaGN ResBlock. cond = [time_emb || label_emb] is projected to scale/shift."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float = 0.1,
                 num_groups: int = 32):
        super().__init__()
        self.norm1 = _gn(in_ch, num_groups)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = _gn(out_ch, num_groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.adagn = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * out_ch))
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        scale, shift = self.adagn(cond).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention with 64 ch/head."""

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        num_heads = max(1, channels // 64)
        self.norm = _gn(channels, num_groups)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)
        # (B, heads, head_dim, HW)
        attn = torch.einsum("bhci,bhcj->bhij", q * self.scale, k)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhij,bhcj->bhci", attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class _ResStage(nn.Module):
    """Wrap a ResBlock and an optional AttentionBlock so we can keep skip bookkeeping clean."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, use_attn: bool, dropout: float):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, cond_dim, dropout=dropout)
        self.attn = AttentionBlock(out_ch) if use_attn else None

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.res(x, cond)
        if self.attn is not None:
            x = self.attn(x)
        return x


class ConditionalUNet(nn.Module):
    """64x64 RGB conditional UNet with AdaGN."""

    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 128,
        ch_mults: Sequence[int] = (1, 2, 2, 4),
        num_res: int = 2,
        attn_resolutions: Set[int] = frozenset({32, 16, 8}),
        time_emb_dim: int = 512,
        label_emb_dim: int = 256,
        num_classes: int = 24,
        dropout: float = 0.1,
        img_size: int = 64,
    ):
        super().__init__()
        self.cond_dim = time_emb_dim + label_emb_dim
        self.time_emb = TimeEmbeddingMLP(base_ch, time_emb_dim)
        self.label_emb = LabelEmbedding(num_classes, label_emb_dim)
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # ---------- Encoder ----------
        self.down_stages = nn.ModuleList()  # list of _ResStage
        self.down_samplers = nn.ModuleList()  # list of (Downsample or None) per level
        self.skip_channels: List[int] = [base_ch]  # init_conv adds 1 skip

        ch = base_ch
        res = img_size
        num_levels = len(ch_mults)
        for i, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            level_stages = nn.ModuleList()
            for _ in range(num_res):
                stage = _ResStage(ch, out_ch, self.cond_dim, res in attn_resolutions, dropout)
                level_stages.append(stage)
                ch = out_ch
                self.skip_channels.append(ch)
            self.down_stages.append(level_stages)
            if i != num_levels - 1:
                self.down_samplers.append(Downsample(ch))
                self.skip_channels.append(ch)
                res //= 2
            else:
                self.down_samplers.append(None)

        # ---------- Bottleneck ----------
        self.mid_res1 = ResBlock(ch, ch, self.cond_dim, dropout=dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_res2 = ResBlock(ch, ch, self.cond_dim, dropout=dropout)

        # ---------- Decoder ----------
        self.up_stages = nn.ModuleList()  # list of ModuleList of _ResStage per level
        self.up_samplers = nn.ModuleList()
        skip_chs = list(self.skip_channels)  # we'll pop from the end during forward
        # Build mirror — pop conceptually to compute in_channels for ResBlocks.
        for i, mult in reversed(list(enumerate(ch_mults))):
            out_ch = base_ch * mult
            level_stages = nn.ModuleList()
            for _ in range(num_res + 1):
                skip_ch = skip_chs.pop()
                stage = _ResStage(ch + skip_ch, out_ch, self.cond_dim,
                                   res in attn_resolutions, dropout)
                level_stages.append(stage)
                ch = out_ch
            self.up_stages.append(level_stages)
            if i != 0:
                self.up_samplers.append(Upsample(ch))
                res *= 2
            else:
                self.up_samplers.append(None)

        # ---------- Output ----------
        self.out_norm = _gn(ch, 32)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cond = torch.cat([self.time_emb(t), self.label_emb(labels)], dim=-1)

        h = self.init_conv(x)
        skips: List[torch.Tensor] = [h]

        # Encoder
        for level_stages, downsampler in zip(self.down_stages, self.down_samplers):
            for stage in level_stages:
                h = stage(h, cond)
                skips.append(h)
            if downsampler is not None:
                h = downsampler(h)
                skips.append(h)

        # Bottleneck
        h = self.mid_res1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_res2(h, cond)

        # Decoder
        for level_stages, upsampler in zip(self.up_stages, self.up_samplers):
            for stage in level_stages:
                h = torch.cat([h, skips.pop()], dim=1)
                h = stage(h, cond)
            if upsampler is not None:
                h = upsampler(h)

        return self.out_conv(self.out_act(self.out_norm(h)))
