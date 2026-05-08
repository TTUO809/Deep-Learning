"""===============================================================
【 Conditional UNet for Denoising Diffusion Probabilistic Models (DDPM). 】

This implementation combines design choices from the original DDPM and ADM (Beat-GANs) papers:
    - Backbone: UNet architecture with Group Normalization (GN).
    - Time Embedding: Sinusoidal positional encoding followed by an MLP.
    - Label Embedding: Multi-label one-hot (B, num_classes) mapped via a linear MLP.
    - Conditioning (AdaGN): The combined (time || label) embedding is projected to 
      per-channel scale and shift parameters for the second GroupNorm in each ResBlock.
    - Attention: Multi-head self-attention (64 channels/head) applied at 32x32, 16x16, and 8x8 resolutions.
    - Resampling: BigGAN-style up/downsampling (nearest-neighbor interp + conv / strided conv).
    - Channels: Base=128, multipliers=(1, 2, 2, 4), resulting in (128, 256, 256, 512) feature maps.
==============================================================="""

from __future__ import annotations

import math
from typing import List, Sequence, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int, num_groups: int = 32) -> nn.GroupNorm:
    """---------------------------------------------------------------
    【 Constructs GroupNorm dynamically adjusting if channels < num_groups. 】

    Automatically finds the largest valid divisor for Group Normalization 
    to ensure the number of groups evenly divides the channel count.
    
    Input:
        channels: The number of feature map channels to normalize.
        num_groups: The target/maximum number of groups (default: 32).
    Output:
        A PyTorch nn.GroupNorm layer with a valid group count.
    ---------------------------------------------------------------"""

    # 1. 初始分組數 g 取「目標分組數」與「通道數」中的較小值。
    #    若通道數本身就小於 32，則分組數不能超過通道數。
    g = min(num_groups, channels)

    # 2. 動態調整：利用 While 迴圈向下搜尋。
    #    PyTorch 要求 channels % g == 0，若無法整除則遞減 g 直到找到最大約數。
    while channels % g != 0:
        g -= 1

    # 3. 回傳配置好的 GroupNorm 層。
    return nn.GroupNorm(g, channels)


class SinusoidalTimeEmbedding(nn.Module):
    """---------------------------------------------------------------
    【 Standard Transformer sinusoidal positional embeddings for timesteps. 】
    
    Maps scalar timesteps into high-dimensional continuous vectors using 
    fixed sinusoidal frequencies, allowing the U-Net to perceive time.
    
    Input shape: (B,) -> Output shape: (B, dim)
    ---------------------------------------------------------------"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # 1. 計算頻率張量 (freqs)：將分母的指數運算轉換至對數空間 (Log-space) 以求數值穩定。
        #    原始公式：omega_i = 1 / (10000 ** (2i / d))
        #    對數轉換：omega_i = exp( -ln(10000) * (i / half) )
        #    其中 d 為嵌入維度 (self.dim)，half = d / 2，i 為 0 到 half-1 的張量
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        
        # 2. 計算對應的角度參數 (args)：利用廣播機制 (Broadcasting)。
        #    將形狀 (B, 1) 的時間步 t 與 (1, half) 的頻率張量 omega_i 相乘。
        #    計算結果：theta_i = t * omega_i
        args = t.float()[:, None] * freqs[None]

        # 3. 拼接正弦與餘弦：將角度分別取 sin 與 cos 後，在特徵維度 (dim=-1) 拼接。
        #    為求切片與計算方便，此處採用前半部全為 sin、後半部全為 cos 的排列方式：
        #    輸出格式：[sin(theta_0), ..., sin(theta_k), cos(theta_0), ..., cos(theta_k)]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeEmbeddingMLP(nn.Module):
    """---------------------------------------------------------------
    【 Projects sinusoidal embeddings to the required condition dimension. 】

    Transforms the fixed sinusoidal positional encoding into a learnable, 
    higher-dimensional latent space for better conditioning in the UNet.

    Input shape: (B,) -> Output shape: (B, time_emb_dim)
    ---------------------------------------------------------------"""
    
    def __init__(self, base_dim: int = 128, time_emb_dim: int = 512):
        super().__init__()
        # 1. 建立時間特徵的特徵抽取網路 (MLP)。
        #    將原本無參數的弦波編碼，投影到更高維度的連續潛在空間中。
        self.net = nn.Sequential(
            # 步驟 A：計算基礎的弦波位置編碼，輸出維度為 base_dim (預設 128)。
            SinusoidalTimeEmbedding(base_dim),

            # 步驟 B：第一層線性映射，將特徵維度擴展至 time_emb_dim (預設 512)。
            nn.Linear(base_dim, time_emb_dim),

            # 步驟 C：使用 SiLU (Swish) 激活函數引入非線性。
            #        SiLU (x * sigmoid(x)) 具有平滑的導數，在擴散模型中表現優於 ReLU。
            nn.SiLU(),

            # 步驟 D：第二層線性映射，進一步提煉特徵，輸出維度維持 time_emb_dim。
            nn.Linear(time_emb_dim, time_emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # 1. 接收純量時間步 t (形狀為 [B,])。
        # 2. 依次通過弦波編碼與兩層 MLP，回傳形狀為 (B, time_emb_dim) 的時間特徵矩陣。
        return self.net(t)


class LabelEmbedding(nn.Module):
    """---------------------------------------------------------------
    【 Projects multi-label one-hot vectors into a continuous latent space. 】

    Transforms sparse, discrete label information into a dense, 
    continuous latent representation suitable for U-Net conditioning.
    
    Input shape: (B, num_classes) -> Output shape: (B, label_emb_dim)
    ---------------------------------------------------------------"""

    def __init__(self, num_classes: int = 24, label_emb_dim: int = 256):
        super().__init__()
        # 1. 建立標籤特徵的映射網路 (MLP)。
        #    將原本離散、稀疏的 One-hot 矩陣，映射為富含語意的連續稠密向量 (Dense Vector)。
        self.net = nn.Sequential(
            # 步驟 A：第一層線性映射，將輸入的類別總數 (num_classes，預設 24) 
            #        投影至目標的潛在空間維度 (label_emb_dim，預設 256)。
            nn.Linear(num_classes, label_emb_dim),

            # 步驟 B：使用 SiLU (Swish) 激活函數引入非線性，增加特徵表達能力。
            nn.SiLU(),

            # 步驟 C：第二層線性映射，進一步提煉特徵，輸出維度維持不變。
            nn.Linear(label_emb_dim, label_emb_dim),
        )

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        # 1. 接收形狀為 (B, num_classes) 的多標籤 One-hot 張量。
        # 2. 通過 MLP 進行降維/升維與特徵轉換，回傳形狀為 (B, label_emb_dim) 的標籤特徵矩陣。
        return self.net(labels)


class ResBlock(nn.Module):
    """---------------------------------------------------------------
    【 Residual Block with Adaptive Group Normalization (AdaGN). 】
    
    Projects the combined condition [time_emb || label_emb] into scale (y_s) 
    and shift (y_b) parameters to modulate the feature map.

    Input shape: 
        x: (B, in_ch, H, W)
        cond: (B, cond_dim)
    Output shape: (B, out_ch, H, W)
    ---------------------------------------------------------------"""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float = 0.1,
                 num_groups: int = 32):
        super().__init__()
        # 1. 定義第一階段的卷積區塊：正規化 -> 卷積。
        self.norm1 = _gn(in_ch, num_groups)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        # 2. 定義第二階段的卷積區塊：正規化 (後接 AdaGN) -> Dropout -> 卷積。
        self.norm2 = _gn(out_ch, num_groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()

        # 3. 建立 AdaGN 投影層 (MLP)：將條件向量映射為 2 倍的輸出通道數，
        #    以便後續切分出 scale (縮放) 與 shift (平移) 兩個參數。
        self.adagn = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * out_ch))

        # 4. 殘差連接 (Shortcut)：若輸入與輸出通道不同，使用 1x1 卷積對齊維度；否則為恆等映射。
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        # 5. 零初始化：將最後一層卷積的權重與偏差設為 0。
        #    這能確保在訓練初期，殘差區塊的輸出等同於輸入 (Identity)，有助於深層網路的梯度穩定。
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # 1. 影像特徵預處理：經過 GroupNorm -> SiLU -> 第一層 3x3 卷積。
        #    = 把畫布上原本的「強弱對比」抹平，只保留純粹的「結構（形狀）」，讓它變成一張乾淨的底圖，準備接受新的條件。
        h = self.conv1(self.act(self.norm1(x)))

        # 2. 條件特徵轉換：將時間與標籤混合的 cond 輸入 AdaGN 投影層，
        #    並沿著通道維度 (dim=1) 將結果平分成 scale 與 shift 兩個張量。
        scale, shift = self.adagn(cond).chunk(2, dim=1)

        # 3. 執行 AdaGN (Adaptive Group Normalization) 條件注入：
        #    公式：h = (1 + scale) * GroupNorm(h) + shift
        #    特徵圖 h 的形狀： (B, C, H, W)。原參數 scale、shift 的形狀： (B, C)
        #    利用 [:, :, None, None] 廣播機制變 (B, C, 1, 1)，將通道層級的參數擴展到整張特徵圖的 (H, W) 上。
        #    = 根據目前的「時間與標籤」，重新決定這張特徵圖哪些通道（特徵）要被放大（強調），哪些要被縮小（抑制）。
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        # 4. 特徵後處理：經過 SiLU 激活與 Dropout 後，通過最後一層卷積。
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # 5. 殘差相加：將處理後的特徵 h 與捷徑連接 (Shortcut) 過來的原始特徵 x 相加。
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """---------------------------------------------------------------
    【 Self-attention block with fixed channels per head (default: 64). 】
    
    Applies multi-head self-attention to capture long-range dependencies 
    across the spatial dimensions (H, W) of the feature map.
    
    Input shape: (B, C, H, W) -> Output shape: (B, C, H, W)
    ---------------------------------------------------------------"""

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        # 1. 動態計算注意力頭數 (num_heads)：
        #    固定每個頭處理的通道數 (head_dim) 為 64。這能確保運算效率與特徵表達能力的平衡。
        num_heads = max(1, channels // 64)

        # 2. 特徵正規化：使用你定義的動態 GroupNorm。
        self.norm = _gn(channels, num_groups)

        # 3. Q, K, V 映射層：
        #    使用單一個 1x1 卷積，一次性將通道數放大 3 倍，後續再切分成 Query, Key, Value。
        #    這比分開寫 3 個卷積層在運算上更有效率。
        self.qkv = nn.Conv2d(channels, channels * 3, 1)

        # 4. 輸出投影層 (Projection)：將注意力計算完的結果投射回原始通道數。
        self.proj = nn.Conv2d(channels, channels, 1)

        # 5. 零初始化技巧 (Zero-Initialization)：
        #    將輸出層的權重與偏差設為 0。這樣在訓練初期，此模組會是一個恆等映射 (Identity)，
        #    特徵原封不動地通過，能大幅提升深層模型的初始訓練穩定度。
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        # 6. 紀錄注意力機制所需的超參數，並計算縮放因子 (1 / sqrt(d_k))，
        #    用來防止內積結果過大導致 Softmax 梯度消失。
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. 特徵預處理：通過正規化。
        h = self.norm(x)

        # 2. 計算 Q, K, V 並重塑形狀：
        #    原始 qkv 輸出形狀：(B, 3*C, H, W)
        #    重塑為：(B, 3, heads, head_dim, H*W)，將空間維度 (H, W) 攤平成一維序列。
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)

        #    利用 unbind(1) 將第 1 個維度 (長度為 3) 解開，完美分配給 q, k, v。
        #    此時 q, k, v 各自的形狀皆為：(B, heads, head_dim, HW)
        q, k, v = qkv.unbind(1)

        # 3. 計算注意力分數 (Attention Computation)：Softmax(Q * K^T / sqrt(d))
        #    使用 einsum 計算 Q 和 K 的內積：
        #    b: Batch, h: Heads, c: head_dim, i: 目標像素位置, j: 來源像素位置
        #    (b, h, c, i) 與 (b, h, c, j) 內積 -> 得到 (b, h, i, j) 的注意力權重矩陣。
        attn = torch.einsum("bhci,bhcj->bhij", q * self.scale, k)

        #    對最後一個維度 (j, 也就是來源像素) 取 Softmax，使其機率總和為 1。
        attn = attn.softmax(dim=-1)

        # 4. 套用注意力權重於 Value：
        #    將注意力矩陣 (b, h, i, j) 乘上 Value 矩陣 (b, h, c, j)，
        #    融合出新的特徵：(b, h, c, i)，也就是 (B, heads, head_dim, HW)。
        out = torch.einsum("bhij,bhcj->bhci", attn, v)

        # 5. 恢復空間維度：
        #    將攤平的序列重塑回 2D 圖片形狀 (B, C, H, W)。
        out = out.reshape(B, C, H, W)

        # 6. 殘差連接：
        #    經過 1x1 卷積投影後，與原始輸入 x 相加。
        return x + self.proj(out)


class Downsample(nn.Module):
    """---------------------------------------------------------------
    【 BigGAN-style downsampling block (Strided Conv). 】
    
    Halves the spatial resolution of the feature map using a strided 
    convolution, which is often preferred over pooling in generative models.
    
    Input shape: (B, ch, H, W) -> Output shape: (B, ch, H/2, W/2)
    ---------------------------------------------------------------"""

    def __init__(self, ch: int):
        super().__init__()
        # 1. 使用步長 (stride) 為 2 的 3x3 卷積層來進行空間降維。
        #    在生成模型中，使用 Strided Convolution 讓模型「學習」如何濃縮資訊，
        #    通常會比直接丟棄資訊的 Max Pooling 保留更多細節。
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 直接通過卷積層，長寬解析度減半，但通道數 (ch) 維持不變。
        return self.conv(x)


class Upsample(nn.Module):
    """---------------------------------------------------------------
    【 BigGAN-style upsampling block (Nearest Interpolation + Conv). 】
    
    Doubles the spatial resolution using nearest-neighbor interpolation 
    followed by a convolution to smooth out artifacts.
    
    Input shape: (B, ch, H, W) -> Output shape: (B, ch, H*2, W*2)
    ---------------------------------------------------------------"""
    def __init__(self, ch: int):
        super().__init__()
        # 1. 定義一個標準的 3x3 卷積層 (步長為 1，不改變圖片尺寸)。
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 空間放大：先使用「最近鄰插值 (Nearest-neighbor)」將特徵圖的長寬直接放大 2 倍。
        # 2. 特徵平滑：插值放大後的圖片會有很多鋸齒狀的邊緣，因此馬上接一個 3x3 卷積，
        #    讓模型把這些生硬的邊緣「抹平」，融合出自然的特徵。
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class _ResStage(nn.Module):
    """---------------------------------------------------------------
    【 Wrap a ResBlock and an optional AttentionBlock. 】
    
    Helper module to keep skip connections bookkeeping clean in the UNet.
    It sequentially applies a residual block and an optional self-attention block.
    
    Input shape: 
        x: (B, in_ch, H, W)
        cond: (B, cond_dim)
    Output shape: (B, out_ch, H, W)
    ---------------------------------------------------------------"""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, use_attn: bool, dropout: float):
        super().__init__()
        # 1. 建立核心殘差區塊 (ResBlock)：負責特徵提取與條件注入 (AdaGN)。
        self.res = ResBlock(in_ch, out_ch, cond_dim, dropout=dropout)

        # 2. 條件實例化注意力區塊 (AttentionBlock)：
        #    根據所在的解析度層級 (use_attn 標籤)，決定是否附加注意力機制。
        #    通常只在較小的特徵圖 (如 32x32, 16x16) 上啟用，以在運算成本與全局特徵捕捉間取得平衡。
        self.attn = AttentionBlock(out_ch) if use_attn else None

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # 1. 影像特徵 x 與條件特徵 cond 先通過殘差區塊。
        x = self.res(x, cond)

        # 2. 若該階段啟用了注意力機制，則將更新後的特徵進一步輸入注意力區塊。
        #    注意：AttentionBlock 內部不需要時間/標籤條件，它只專注於尋找空間上的自我關聯性。
        if self.attn is not None:
            x = self.attn(x)

        # 3. 回傳處理完畢的特徵圖，準備存入 skip connections 列表或往下一個階段傳遞。
        return x


class ConditionalUNet(nn.Module):
    """---------------------------------------------------------------
    【 Conditional UNet core architecture for Denoising Diffusion. 】
    
    64x64 RGB conditional UNet with AdaGN.
    
    Args:
        in_ch (int)             : Input image channels (3 for RGB).
        base_ch (int)           : Base channel width.
        ch_mults (tuple)        : Channel multipliers for each resolution level.
        num_res (int)           : Number of ResBlocks per level.
        attn_resolutions (set)  : Resolutions at which to apply self-attention.
        time_emb_dim (int)      : Dimension for time embedding.
        label_emb_dim (int)     : Dimension for label embedding.
        num_classes (int)       : Number of unique conditional classes.
        dropout (float)         : Dropout probability.
        img_size (int)          : Spatial resolution of the input image.
    ---------------------------------------------------------------"""

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

        # 1. 條件特徵初始化：
        #    將時間編碼與標籤編碼的維度相加，作為後續 AdaGN 的總條件維度 (cond_dim)。
        self.cond_dim = time_emb_dim + label_emb_dim
        self.time_emb = TimeEmbeddingMLP(base_ch, time_emb_dim)
        self.label_emb = LabelEmbedding(num_classes, label_emb_dim)

        # 2. 初始卷積：將 3 通道的 RGB 圖片轉換為 base_ch (預設 128) 的特徵圖。
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # ==========================================
        # ---------- Encoder (下採樣階段) ----------
        # ==========================================
        self.down_stages = nn.ModuleList()    # 存放每層的殘差與注意力區塊 (_ResStage)
        self.down_samplers = nn.ModuleList()  # 存放對應層級的下採樣模組 (Downsample)

        # 3. 建立 Skip Connections 紀錄表：
        #    用來記錄每個階段輸出特徵圖的通道數，方便 Decoder 階段進行拼接 (Concat)。
        #    先把 init_conv 產生的基礎通道數存進去。
        self.skip_channels: List[int] = [base_ch]

        ch = base_ch
        res = img_size
        num_levels = len(ch_mults)

        # 4. 逐層構建 Encoder：
        for i, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            level_stages = nn.ModuleList()

            # 步驟 A：在當前解析度下，建立 num_res 個殘差區塊 (與可選的注意力區塊)。
            for _ in range(num_res):
                stage = _ResStage(ch, out_ch, self.cond_dim, res in attn_resolutions, dropout)
                level_stages.append(stage)
                ch = out_ch
                self.skip_channels.append(ch)   # 每過一個區塊，就記錄一次通道數
            self.down_stages.append(level_stages)

            # 步驟 B：若尚未到達最深層 (最後一個 level)，則加入下採樣區塊將解析度減半。
            if i != num_levels - 1:
                self.down_samplers.append(Downsample(ch))
                self.skip_channels.append(ch)   # 下採樣後的特徵也會作為 skip connection
                res //= 2
            else:
                # 最深層不需要再下採樣。
                self.down_samplers.append(None)

        # ==========================================
        # ---------- Bottleneck (瓶頸層) ----------
        # ==========================================
        # 5. U 型網路的最底部，解析度最小，特徵最抽象。
        #    使用兩個殘差區塊夾著一個全局注意力機制，充分融合高階語意與時間/標籤條件。
        self.mid_res1 = ResBlock(ch, ch, self.cond_dim, dropout=dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_res2 = ResBlock(ch, ch, self.cond_dim, dropout=dropout)

        # ==========================================
        # ---------- Decoder (上採樣階段) ----------
        # ==========================================
        self.up_stages = nn.ModuleList()
        self.up_samplers = nn.ModuleList()

        # 6. 複製一份 skip_channels 準備「反向彈出 (pop)」以對齊通道數。
        skip_chs = list(self.skip_channels)

        # 7. 反向逐層構建 Decoder (像鏡子一樣對稱)：
        for i, mult in reversed(list(enumerate(ch_mults))):
            out_ch = base_ch * mult
            level_stages = nn.ModuleList()

            # 步驟 A：建立殘差區塊。迴圈次數是 num_res + 1。
            # 因為 Encoder 的下採樣 (Downsample) 特徵也會被 skip 過來，
            # 需要多一個區塊來消化它。
            for _ in range(num_res + 1):
                # 從紀錄表彈出對應 Encoder 層的通道數。
                skip_ch = skip_chs.pop()

                # 計算輸入通道數為 ch (當前特徵通道) + skip_ch (跳躍連接過來的通道)。
                # 因為 forward 時會使用 torch.cat([h, skip], dim=1) 進行通道拼接。
                stage = _ResStage(ch + skip_ch, out_ch, self.cond_dim,
                                   res in attn_resolutions, dropout)
                level_stages.append(stage)
                ch = out_ch
            self.up_stages.append(level_stages)

            # 步驟 B：若尚未回到最淺層，則加入上採樣區塊將解析度放大 2 倍。
            if i != 0:
                self.up_samplers.append(Upsample(ch))
                res *= 2
            else:
                # 已經回到原始尺寸，不需要上採樣。
                self.up_samplers.append(None)

        # ==========================================
        # ---------- Output (輸出層) ----------
        # ==========================================
        # 8. 最終輸出處理：特徵收斂回與目標圖片相同的通道數 (預設為 3 通道的預測雜訊)。
        self.out_norm = _gn(ch, 32)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, in_ch, 3, padding=1)

        # 9. 零初始化：擴散模型的標準做法。
        #    讓網路在一開始預測的雜訊為 0，能大幅穩定早期的訓練過程。
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy image tensor of shape (B, 3, H, W). 
               (The current noisy state of the image at timestep t)
            t: Timestep tensor of shape (B,).
            labels: Multi-label one-hot tensor of shape (B, num_classes).
            
        Returns:
            Noise prediction tensor of shape (B, 3, H, W).
        """

        # 1. 條件訊號準備 (Conditioning Vector)：
        #    分別計算時間特徵與標籤特徵，然後在特徵維度 (dim=-1) 上進行拼接。
        #    這個長向量 `cond` 包含了「現在是第幾步」與「我們要畫什麼」，
        #    接下來將送給所有殘差區塊內的 AdaGN 使用。
        cond = torch.cat([self.time_emb(t), self.label_emb(labels)], dim=-1)

        # 2. 初始卷積與建立 Skip 堆疊：
        #    將 3 通道的圖片轉換為高維度特徵圖。
        h = self.init_conv(x)

        #    建立一個名為 skips 的列表 (作為 Stack 使用)。
        #    U-Net 的精髓在於保留淺層的空間細節，因此將初始特徵先存入清單。
        skips: List[torch.Tensor] = [h]

        # ==========================================
        # ---------- Encoder pass (下採樣階段) ----------
        # ==========================================
        # 3. 逐層提煉特徵並縮小圖片解析度：
        for level_stages, downsampler in zip(self.down_stages, self.down_samplers):

            # 步驟 A：特徵提取
            for stage in level_stages:
                h = stage(h, cond)
                skips.append(h) # 每次經過殘差區塊，都將特徵存起來備用

            # 步驟 B：解析度減半
            if downsampler is not None:
                h = downsampler(h)
                skips.append(h) # 下採樣後的特徵同樣存入堆疊

        # ==========================================
        # ---------- Bottleneck pass (瓶頸層) ----------
        # ==========================================
        # 4. 網路的最深處，特徵圖尺寸最小，但通道數最多、語意最豐富。
        #    依序通過殘差 -> 全局注意力 -> 殘差，進一步融合時間與標籤條件。
        h = self.mid_res1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_res2(h, cond)

        # ==========================================
        # ---------- Decoder pass (上採樣階段) ----------
        # ==========================================
        # 5. 逐層放大圖片解析度，並透過跳躍連接補回遺失的空間細節：
        for level_stages, upsampler in zip(self.up_stages, self.up_samplers):

            # 步驟 A：特徵融合與重建
            for stage in level_stages:
                # Skip Connection：
                # skips.pop() 會從堆疊「最後面」拿出剛才 Encoder 存入的同尺寸特徵。
                # 接著使用 torch.cat(..., dim=1) 將它與當前 Decoder 的特徵在「通道維度」上拼接。
                h = torch.cat([h, skips.pop()], dim=1)

                # 拼接後的厚特徵圖 (ch + skip_ch) 通過殘差區塊，再次被壓縮回預期的輸出通道數。
                h = stage(h, cond)

            # 步驟 B：解析度放大 2 倍
            if upsampler is not None:
                h = upsampler(h)

        # ==========================================
        # ---------- Output pass (輸出層) ----------
        # ==========================================
        # 6. 預測雜訊：最後經過一次正規化、激活函數與卷積，
        #    輸出形狀與輸入圖片相同的 (B, 3, H, W) 張量。
        #    這個張量代表著「模型認為這張圖片裡被加了多少雜訊」。
        return self.out_conv(self.out_act(self.out_norm(h)))
