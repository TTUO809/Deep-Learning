import os
import random
import numpy as np

import torch
import torch.nn as nn

def set_seed(seed=42, deterministic=True):
    """
    Args:
        seed (int): 隨機種子值 (預設為 42)。
        deterministic (bool): 是否啟用 PyTorch 確定性算法與 cuDNN 設定 (預設為 True, 以確保完全可重現的結果)。
    Description:
        設定所有相關的隨機種子，以確保在不同運行環境中得到相同的結果。
        包括 Python 內建的 random 模組、NumPy、PyTorch 在 CPU 和 GPU 上的隨機種子設定。
        當 deterministic=True 時，還會強制 PyTorch 使用確定性算法，並禁用 cuDNN 的自動調整功能，以確保每次運行都使用相同的算法，進一步增強可重現性。
    Note:
        - 在某些情況下，啟用 deterministic 可能會導致性能下降，因為某些非確定性算法可能更快。請根據實際需求選擇是否啟用。
        - 即使設置了隨機種子，某些操作（如使用多 GPU 或特定的 cuDNN 操作）可能仍然具有非確定性行為，因此在這些情況下，完全可重現性可能無法保證。
    """

    random.seed(seed)       # Python 內建 random 模組可重現。
    np.random.seed(seed)    # NumPy 可重現。
    torch.manual_seed(seed) # PyTorch 在 CPU 上可重現。
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)     # PyTorch 在單個 GPU 上可重現。
        torch.cuda.manual_seed_all(seed) # PyTorch 在多個 GPU 上可重現。

    if deterministic:
        torch.backends.cudnn.deterministic = True   # 確保 cuDNN 使用確定性算法。
        torch.backends.cudnn.benchmark = False      # 禁用 cuDNN 的自動調整功能，確保每次運行都使用相同的算法。
        torch.use_deterministic_algorithms(True, warn_only=True)  # 強制所有 PyTorch 操作使用確定性算法。

def cal_dice_score(predict_logits, GT_masks, threshold=0.5, smooth=1e-6, use_pred_binary=True):
    """
    Args:
        predict_logits (torch.Tensor): 模型預測的輸出，形狀為 (B, 1, H, W)，包含每個像素屬於前景的概率。
        GT_masks (torch.Tensor): 真實的遮罩，形狀為 (B, 1, H, W)。
        threshold (float): 將概率轉換為二值遮罩的閾值。
        smooth (float): 平滑項，防止分母為零。
        use_pred_binary (bool): 
            True (default): 算出 Hard Dice Score ，先將預測概率轉換為二值遮罩。
            False: 算出 Soft Dice Score ，直接使用預測概率計算，保留連續機率梯度，為了在訓練過程中作為損失函數。
    Returns:
        (float): Dice Score 的平均值，範圍在 [0, 1] 之間，值越大表示預測與真實遮罩越接近。
    Description:
        Dice Score 是衡量預測遮罩與真實遮罩相似度的指標，計算公式如下：
        Dice = 2 * |A ∩ B| / (|A| + |B|)
    """

    probs_logits = torch.sigmoid(predict_logits)        # 將模型輸出轉換為 0~1 的概率值。
    
    if use_pred_binary:
        preds = (probs_logits > threshold).float()          # 【Hard Dice A】根據傳入的 threshold 進行二值化截斷，將概率值轉換為二值遮罩。
    else:
        preds = probs_logits                                # 【Soft Dice A】直接使用預測概率計算，保留連續機率梯度。
    
    intersection = (preds * GT_masks).sum(dim=(2, 3))           # 【A ∩ B】計算交集 (preds 和 GT_masks 都為 1 的像素數量)。
    union = preds.sum(dim=(2, 3)) + GT_masks.sum(dim=(2, 3))    # 【|A| + |B|】計算聯集 (preds 和 GT_masks 的像素總數)。

    dice_score = (2.0 * intersection + smooth) / (union + smooth)   # 【Dice = 2 * 【A ∩ B】 / (【|A| + |B|】)】計算 Dice Score。
    
    if use_pred_binary:
        return dice_score.mean().item() # 返回 Hard Dice Score 的平均值。
    else:
        return dice_score.mean()        # 返回 Soft Dice Score 的平均值，保留梯度信息。

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.5, gamma=2.0):
        """
        Args:
            alpha (float): 正樣本的權重，默認為 0.5。
            gamma (float): 調整難易樣本的參數，默認為 2.0。
        Description:
            Focal Loss 是用於處理正負樣本極度不平衡問題的損失函數，通過對難易樣本加權，使模型更關注難以分類的樣本。
        """
        
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predict_logits, GT_masks):
        """
        Args:
            predict_logits (torch.Tensor): 模型預測的輸出，形狀為 (B, 1, H, W)，包含每個像素屬於前景的概率。
            GT_masks (torch.Tensor): 真實的遮罩，形狀為 (B, 1, H, W)。
        Returns:
            (torch.Tensor): Focal Loss 的平均值。
        Description:
            Focal Loss 的計算過程如下：
            1. 首先計算每個像素的 BCE Loss。
            2. 計算 pt ，即模型對正確類別的預測概率。
            3. 根據 GT_masks 計算 alpha_t ，對正負樣本分別賦予不同的權重。(此處因為寵物圖像分割的正負樣本沒有嚴重不平衡，所以 alpha 設為 0.5 ，對正負樣本同等對待。)
            4. 最後計算 Focal Loss ，對難易樣本進行加權，使模型更關注難以分類的樣本。
        """

        bce_loss = self.bce_with_logits(predict_logits, GT_masks)  # 計算每個像素的 BCE Loss。
        pt = torch.exp(-bce_loss)  # pt 是模型對正確類別的預測概率。

        alpha_t = self.alpha * GT_masks + (1 - self.alpha) * (1 - GT_masks)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss  # 計算 Focal Loss，對難易樣本進行加權。(預測越準確的像素，權重越小；預測越錯的，權重越大。)

        return focal_loss.mean()

class FocalDiceLoss(nn.Module):

    def __init__(self, focal_weight=0.5, dice_weight=0.5, alpha=0.5, gamma=2.0):
        """
        Args:
            focal_weight (float): Focal Loss 的權重，默認為 0.5。
            dice_weight (float): Dice Loss 的權重，默認為 0.5。
            alpha (float): Focal Loss 中正樣本的權重，默認為 0.5。
            gamma (float): Focal Loss 中調整難易樣本的參數，默認為 2.0。
        Description:
            FocalDiceLoss 是 Focal Loss 和 Dice Loss 的組合損失函數，通過加權結合兩者的損失值，使模型同時關注難以分類的樣本和整體的分割質量。
            Focal Loss 用於處理正負樣本不平衡問題，而 Dice Loss 用於衡量預測遮罩與真實遮罩的相似度，兩者結合可以提升模型在分割任務中的表現，特別是在處理小物體或邊緣模糊的情況下。
        """

        super(FocalDiceLoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, predict_logits, GT_masks):
        """
        Args:
            predict_logits (torch.Tensor): 模型預測的輸出，形狀為 (B, 1, H, W)，包含每個像素屬於前景的概率。
            GT_masks (torch.Tensor): 真實的遮罩，形狀為 (B, 1, H, W)。
        Returns:
            (torch.Tensor): 結合 Focal Loss 和 Dice Loss 的加權總損失值。
        Description:
            FocalDiceLoss 的計算過程如下：
            1. 首先計算 Focal Loss。
            2. 計算 Dice Loss ，並將其轉換為損失值 (1 - Dice Score)。
            3. 最後將 Focal Loss 和 Dice Loss 按照指定的權重進行加權求和，得到最終的損失值。
        """
        focal_loss = self.focal(predict_logits, GT_masks)
        dice_loss = 1.0 - cal_dice_score(predict_logits, GT_masks, use_pred_binary=False)
        
        return (self.focal_weight * focal_loss) + (self.dice_weight * dice_loss)

class BCEDiceLoss(nn.Module):

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        """
        Args:
            bce_weight (float): BCE Loss 的權重，默認為 0.5。
            dice_weight (float): Dice Loss 的權重，默認為 0.5。
        Description:
            BCEDiceLoss 是 BCE Loss 和 Dice Loss 的組合損失函數，通過加權結合兩者的損失值，使模型同時關注像素級的分類準確性和整體的分割質量。
            BCE Loss 用於衡量每個像素的分類準確性，而 Dice Loss 用於衡量預測遮罩與真實遮罩的相似度，兩者結合可以做到在分割任務中既關注細節又關注整體。
        """

        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, predict_logits, GT_masks):
        """
        Args:
            predict_logits (torch.Tensor): 模型預測的輸出，形狀為 (B, 1, H, W)，包含每個像素屬於前景的概率。
            GT_masks (torch.Tensor): 真實的遮罩，形狀為 (B, 1, H, W)。
        Returns:
            (torch.Tensor): 結合 BCE Loss 和 Dice Loss 的加權總損失值。
        Description:
            BCEDiceLoss 的計算過程如下：
            1. 首先計算 BCE Loss。
            2. 計算 Dice Loss ，並將其轉換為損失值 (1 - Dice Score)。
            3. 最後將 BCE Loss 和 Dice Loss 按照指定的權重進行加權求和，得到最終的損失值。
        """
    
        bce_loss = self.bce(predict_logits, GT_masks)
        dice_loss = 1.0 - cal_dice_score(predict_logits, GT_masks, use_pred_binary=False)
        
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)

def resolve_model_config(args):
    """
    Args:
        args (argparse.Namespace): 需包含 model、model_path、save_model_dir 三個欄位。
    Returns:
        (Tuple[str, str]): (selected_model, model_path)
    Description:
        根據使用者提供的參數解析出要使用的模型類型和對應的模型權重路徑。
        1. 首先檢查是否提供了 model_path 參數，如果有，則直接使用該路徑，並根據文件名稱推斷模型類型 (unet 或 res_unet)。
        2. 如果沒有提供 model_path，則根據 model 參數和 save_model_dir 參數自動生成模型權重的路徑，假設模型權重文件命名為 "{model}_best.pth"。
    Note:
        - 這個函數的目的是為了讓用戶在推理時能夠靈活地指定模型權重的路徑，無論是直接提供完整路徑還是通過模型名稱和保存目錄自動生成路徑。
        - 在使用 model_path 參數時，函數會嘗試從文件名稱中推斷模型類型，這要求用戶在命名模型權重文件時包含模型名稱 (如 "unet" 或 "res_unet")，以確保正確解析。
    """
    selected_model = args.model

    if args.model_path:
        model_path = args.model_path
        model_filename = os.path.basename(model_path).lower()

        if 'res_unet' in model_filename  or 'res_unet' in args.model:
            selected_model = 'res_unet'
        elif 'unet' in model_filename or 'unet' in args.model:
            selected_model = 'unet'

        return selected_model, model_path

    model_path = os.path.join(args.save_model_dir, f"{selected_model}_best.pth")
    return selected_model, model_path