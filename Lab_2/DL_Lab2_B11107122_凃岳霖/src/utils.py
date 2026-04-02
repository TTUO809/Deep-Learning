import os
import random
import multiprocessing
import numpy as np

import torch
import torch.nn as nn

def set_seed(seed=42, deterministic=True):
    '''
    設定所有隨機數生成器的種子，確保實驗的完全可重現性。
    
    這個函數設定以下的隨機種子：
    - Python 內建 random 模組
    - NumPy 的隨機數生成器
    - PyTorch 在 CPU 上的隨機數生成器
    - PyTorch 在 GPU 上的隨機數生成器（包括多 GPU）
    - cuDNN 後端（禁用自動調整，使用確定性算法）
    
    Args:
        seed (int): 隨機種子值，默認為 42。
        deterministic (bool): 是否啟用 PyTorch 確定性算法與 cuDNN 設定。
    '''
    random.seed(seed)       # Python 內建 random 模組可重現。如: 
    np.random.seed(seed)    # NumPy 可重現。
    torch.manual_seed(seed) # PyTorch 在 CPU 上可重現。
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)     # PyTorch 在單個 GPU 上可重現。
        torch.cuda.manual_seed_all(seed) # PyTorch 在多個 GPU 上可重現。

    if deterministic:
        torch.backends.cudnn.deterministic = True   # 確保 cuDNN 使用確定性算法。
        torch.backends.cudnn.benchmark = False      # 禁用 cuDNN 的自動調整功能，確保每次運行都使用相同的算法。
        torch.use_deterministic_algorithms(True, warn_only=True)  # 強制所有 PyTorch 操作使用確定性算法。

def detect_optimal_num_workers(batch_size=16):
    '''
    Detect the optimal number of workers for DataLoader based on system resources.
    
    Args:
        batch_size (int): The batch size used in DataLoader (optional, for reference).
    
    Returns:
        dict: Contains "cpu_count", "recommended", and "safe_range".
    
    Guideline:
        - num_workers should NOT exceed CPU count
        - Too many workers cause memory overhead and process management costs
        - Typical rule: num_workers = min(batch_size * 2, cpu_count)
        - Start conservatively (2-4) and increase if data loading is the bottleneck
        - Monitor GPU utilization: if GPU is idle, increase num_workers
    '''
    cpu_count = multiprocessing.cpu_count()
    
    # Common heuristic: num_workers = min(batch_size * 2, cpu_count)
    recommended = min(batch_size * 2, cpu_count)
    
    # Clip to reasonable range
    recommended = max(0, min(recommended, cpu_count))
    
    # Safe range: [0, cpu_count], where 0 = main process only
    safe_range = (0, cpu_count)
    
    result = {
        'cpu_count': cpu_count,
        'recommended': recommended,
        'safe_range': safe_range,
        'advice': f"Start with num_workers={recommended} (based on batch_size={batch_size} and {cpu_count} CPUs). Increase if data loading is the bottleneck."
    }
    
    return result

def resolve_model_config(args):
    '''
    優先使用使用者指定的模型路徑；若未提供，則依既有規則自動組裝。
    Args:
        args (argparse.Namespace): 需包含 model、model_path、save_model_dir 三個欄位。
    Returns:
        Tuple[str, str]: (selected_model, model_path)
    '''
    selected_model = args.model

    if args.model_path:
        model_path = args.model_path
        model_filename = os.path.basename(model_path).lower()

        if 'res_unet' in model_filename:
            selected_model = 'res_unet'
        elif 'unet' in model_filename:
            selected_model = 'unet'

        return selected_model, model_path

    model_path = os.path.join(args.save_model_dir, f"{selected_model}_best.pth")
    return selected_model, model_path

def cal_dice_score(predict_logits, GT_masks, threshold=0.5, smooth=1e-6, use_pred_binary=True):
    '''
    Args:
        predict_logits (torch.Tensor): 模型預測的輸出，形狀為 (B, 1, H, W)，包含每個像素屬於前景的概率。
        GT_masks (torch.Tensor): 真實的遮罩，形狀為 (B, 1, H, W)。
        threshold (float): 將概率轉換為二值遮罩的閾值。
        smooth (float): 平滑項，防止分母為零。
        use_pred_binary (bool): 
            True (default): 算出 Hard Dice Score，先將預測概率轉換為二值遮罩。
            False: 算出 Soft Dice Score，直接使用預測概率計算，保留連續機率梯度，為了在訓練過程中作為損失函數。
    Returns:
        (float): Dice Score 的平均值，範圍在 [0, 1] 之間，值越大表示預測與真實遮罩越接近。
    *** Dice = 2 * |A ∩ B| / (|A| + |B|)
    '''

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
    '''
    用於處理正負樣本極度不平衡問題。
    '''
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        '''
        Args:
            alpha (float): 正樣本的權重，默認為 0.5。
            gamma (float): 調整難易樣本的參數，默認為 2.0。
            reduction (str): 指定損失的減少方式，默認為 'mean'。
        '''
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.reduction = reduction

    def forward(self, predict_logits, GT_masks):
        '''
        Args:
            predict_logits (torch.Tensor): 模型預測的輸出，形狀為 (B, 1, H, W)，包含每個像素屬於前景的概率。
            GT_masks (torch.Tensor): 真實的遮罩，形狀為 (B, 1, H, W)。
        Returns:
            (torch.Tensor): Focal Loss 的平均值。
        '''
        bce_loss = self.bce_with_logits(predict_logits, GT_masks)  # 計算每個像素的 BCE Loss。
        pt = torch.exp(-bce_loss)  # pt 是模型對正確類別的預測概率。

        alpha_t = self.alpha * GT_masks + (1 - self.alpha) * (1 - GT_masks)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss  # 計算 Focal Loss，對難易樣本進行加權。(預測越準確的像素，權重越小；預測越錯的，權重越大。)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class FocalDiceLoss(nn.Module):
    '''
    Focal Loss + Dice Loss 的組合損失函數。
    '''
    def __init__(self, focal_weight=0.5, dice_weight=0.5, alpha=0.5, gamma=2.0):
        '''
        Args:
            focal_weight (float): Focal Loss 的權重，默認為 0.5。
            dice_weight (float): Dice Loss 的權重，默認為 0.5。
            alpha (float): Focal Loss 中正樣本的權重，默認為 0.5。
            gamma (float): Focal Loss 中調整難易樣本的參數，默認為 2.0。
        '''
        super(FocalDiceLoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, predict_logits, GT_masks):
        '''
        Args:
            predict_logits (torch.Tensor): 模型預測的輸出，形狀為 (B, 1, H, W)，包含每個像素屬於前景的概率。
            GT_masks (torch.Tensor): 真實的遮罩，形狀為 (B, 1, H, W)。
        Returns:
            (torch.Tensor): 結合 Focal Loss 和 Dice Loss 的加權總損失值。
        '''
        focal_loss = self.focal(predict_logits, GT_masks)
        dice_loss = 1.0 - cal_dice_score(predict_logits, GT_masks, use_pred_binary=False)
        
        return (self.focal_weight * focal_loss) + (self.dice_weight * dice_loss)

class BCEDiceLoss(nn.Module):
    '''
    BCE + Dice Loss 的組合損失函數。
    '''
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        '''
        Args:
            bce_weight (float): BCE Loss 的權重，默認為 0.5。
            dice_weight (float): Dice Loss 的權重，默認為 0.5。
        '''
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, predict_logits, GT_masks):
        '''
        Args:
            predict_logits (torch.Tensor): 模型預測的輸出，形狀為 (B, 1, H, W)，包含每個像素屬於前景的概率。
            GT_masks (torch.Tensor): 真實的遮罩，形狀為 (B, 1, H, W)。
        Returns:
            (torch.Tensor): 結合 BCE Loss 和 Dice Loss 的加權總損失值。
        '''
        bce_loss = self.bce(predict_logits, GT_masks)
        dice_loss = 1.0 - cal_dice_score(predict_logits, GT_masks, use_pred_binary=False)
        
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)