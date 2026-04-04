import argparse
import torch
import time
import os
import gc
import sys
import importlib
import statistics
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "DL_Lab2_B11107122_凃岳霖", 'src'))
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'DL_Lab2_B11107122_凃岳霖', 'dataset', 'oxford-iiit-pet'))

# 想換專案時，直接改這裡的 symbol 名稱即可。
DEFAULT_SYMBOL_MAP = {
    'dataset_class': 'OxfordPetDataset',
    'build_model': '_build_model',
    'process_images': '_process_images',
    'build_criterion': '_build_criterion',
    'build_optimizer_scheduler': '_build_optimizer_scheduler',
}

def load_project_components(src_dir, dataset_module, train_module, symbol_map=None):
    """
    Args:
        src_dir (str): 包含 dataset.py 和 train.py 的源碼目錄路徑。
        dataset_module (str): dataset 模組名稱（不帶 .py 後綴），必須提供 OxfordPetDataset 類別。
        train_module (str): train 模組名稱（不帶 .py 後綴）。
        symbol_map (dict | None): 自訂 symbol 對應。若為 None，使用預設：
            - dataset_class -> OxfordPetDataset
            - build_model -> _build_model
            - process_images -> _process_images
            - build_criterion -> _build_criterion
            - build_optimizer_scheduler -> _build_optimizer_scheduler
    Returns:
        dict: 包含從 dataset_module 和 train_module 加載的必要類別和函數的字典。
    Description:
        這個函數負責動態加載指定源碼目錄中的 dataset_module 和 train_module ，並檢查它們是否提供了必要的類別和函數。
    Note:
        - 確保在執行這個函數之前， src_dir 中的 dataset_module 和 train_module 已經存在，並且包含了必要的類別和函數。
        - 這個函數的目的是為了讓 HardwareTuner 能夠使用與訓練過程相同的模型配置和數據處理邏輯，以確保在測試硬體參數時能夠獲得準確的結果。
    """
    
    # 將 src_dir 添加到 sys.path 中，以確保 Python 能夠找到這些模組
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # 動態導入 dataset_module 和 train_module
    ds_mod = importlib.import_module(dataset_module)
    tr_mod = importlib.import_module(train_module)

    if symbol_map is None:
        symbol_map = DEFAULT_SYMBOL_MAP

    required_keys = {
        'dataset_class',
        'build_model',
        'process_images',
        'build_criterion',
        'build_optimizer_scheduler',
    }
    missing_keys = required_keys - set(symbol_map.keys())
    if missing_keys:
        raise KeyError(f"symbol_map is missing keys: {sorted(missing_keys)}")

    # 檢查到缺少任何必要的類別或函數，引發 AttributeError。
    dataset_symbol = symbol_map['dataset_class']
    train_symbols = {
        'build_model': symbol_map['build_model'],
        'process_images': symbol_map['process_images'],
        'build_criterion': symbol_map['build_criterion'],
        'build_optimizer_scheduler': symbol_map['build_optimizer_scheduler'],
    }

    if not hasattr(ds_mod, dataset_symbol):
        raise AttributeError(f"{dataset_module} does not provide {dataset_symbol}")

    for api_key, symbol in train_symbols.items():
        if not hasattr(tr_mod, symbol):
            raise AttributeError(f"{train_module} does not provide {symbol} (for {api_key})")

    return {
        'Dataset': getattr(ds_mod, dataset_symbol),
        'build_model': getattr(tr_mod, train_symbols['build_model']),
        'process_images': getattr(tr_mod, train_symbols['process_images']),
        'build_criterion': getattr(tr_mod, train_symbols['build_criterion']),
        'build_optimizer_scheduler': getattr(tr_mod, train_symbols['build_optimizer_scheduler']),
    }

# ==========================================
# 2. 核心測試工具類別 (針對語意分割優化)
# ==========================================
class HardwareTuner:
    def __init__(self, args, dataset, project_api, max_vram_ratio=0.9):
        '''
        Args:
            args (argparse.Namespace): 包含模型、損失函數、優化器等設定的參數。
            dataset (torch.utils.data.Dataset): 你要測試的資料集實例 (例如 OxfordPetDataset(...))。
            max_vram_ratio (float): VRAM 使用的安全水位比例，預設為 0.9 (90%)。
        Description:
            這個類別包含兩個主要方法：
            1. find_best_batch_size: 自動尋找最大可用的 Batch Size，直到 VRAM 使用率達到設定的安全水位。
            2. find_best_num_workers: 在找到最佳 Batch Size 後，測試不同的 num_workers 設定，以找到最佳的數據加載效率。
        Note:
            - 在測試過程中，會使用 torch.cuda.max_memory_allocated() 來監控 VRAM 的使用情況。
            - 建議在測試過程中密切監控系統資源，特別是在增加 num_workers 時，以避免過度使用 CPU 資源導致系統不穩定。
        '''

        self.args = args
        self.dataset = dataset
        self.api = project_api
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_vram_ratio = max_vram_ratio

        self.model = self.api['build_model'](args, self.device)
        self.amp_enabled = (not args.no_amp) and (self.device.type == 'cuda') # 只有在 CUDA 可用且未禁用 AMP 時才啟用 AMP
        
        # 獲取 GPU 總記憶體大小 (Bytes 轉為 MB)
        self.total_vram = torch.cuda.get_device_properties(self.device).total_memory / (1024**2) if self.device.type == 'cuda' else 0
        print(f"[System] Testing Model: {args.model}")
        print(f"[System] Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"[System] GPU device: {torch.cuda.get_device_name(self.device)}")
            print(f"[System] GPU total memory: {self.total_vram:.0f} MB")

    def clear_memory(self):
        """
        Description:
            這個方法用於清理 GPU 記憶體，確保在測試不同 Batch Size 和 num_workers 時，能夠獲得準確的 VRAM 使用情況。它會執行以下步驟：
            1. 呼叫 Python 的垃圾回收機制來釋放不再使用的物件。
            2. 使用 torch.cuda.empty_cache() 來清空 PyTorch 的 GPU 緩存，這有助於釋放未被使用的 GPU 記憶體。
            3. 使用 torch.cuda.reset_peak_memory_stats() 來重置 GPU 的峰值記憶體統計，這樣在下一次測試時能夠獲得準確的峰值記憶體使用情況。
        Note:
            - 在測試過程中，建議在每次測試新的 Batch Size 或 num_workers 設定之前，呼叫這個方法來確保測試環境的清潔和準確。
        """

        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def find_best_batch_size(self, starting_bs=2):
        """
        Args:
            starting_bs (int): 開始測試的 Batch Size，預設為 2。
        Returns:
            (int): 最佳的 Batch Size 設定。
        Description:
            這個方法會從指定的起始 Batch Size 開始，逐步增加 Batch Size，並在每次增加後測試模型的 VRAM 使用情況。當 VRAM 使用率達到設定的安全水位 (max_vram_ratio) 時，會停止增加 Batch Size，並返回上一個成功的 Batch Size 作為最佳 Batch Size。
        Note:
            - 在測試過程中，建議密切監控系統資源，特別是在增加 Batch Size 時，以避免過度使用 GPU 資源導致系統不穩定或崩潰。
        """

        print("\n" + "="*50)
        print("🚀 Stage 1 : Finding the Maximum Batch Size")
        print("="*50)
        
        current_bs = starting_bs
        best_bs = starting_bs
        use_fine_search = False

        while True:
            self.clear_memory()
            print(f"👉 Testing Batch Size: {current_bs:2d} ... ", end="", flush=True)
            
            try:
                # 每次 batch size 測試都重建模型/優化器，避免前一次測試影響結果。
                self.model = self.api['build_model'](self.args, self.device)
                criterion = self.api['build_criterion'](self.args)
                optimizer, _ = self.api['build_optimizer_scheduler'](self.args, self.model)
                scaler = GradScaler(self.device.type, enabled=self.amp_enabled)

                # 建立 num_workers=0 的 DataLoader 排除 CPU 干擾
                loader = DataLoader(
                    self.dataset,
                    batch_size=current_bs,
                    num_workers=0,
                    shuffle=True,
                    pin_memory=(self.device.type == 'cuda')
                )
                
                # 試跑 3 個 Step 來測量峰值記憶體
                self.model.train()
                for i, (imgs, masks) in enumerate(loader):
                    if i >= 3: break

                    imgs = self.api['process_images'](imgs, self.args.model).to(self.device)
                    masks = masks.to(self.device)
                    
                    optimizer.zero_grad()
                    with autocast(device_type=self.device.type, enabled=self.amp_enabled):
                        outputs = self.model(imgs)
                        loss = criterion(outputs, masks)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                
                # 檢查 VRAM 佔用率
                max_used = torch.cuda.max_memory_allocated(self.device) / (1024**2) if self.device.type == 'cuda' else 0
                usage_ratio = (max_used / self.total_vram) if self.device.type == 'cuda' else 0
                print(f"VRAM Used: {max_used:4.0f} MB ({usage_ratio*100:.1f}%)")
                
                # 如果超過安全水位，就停止
                if usage_ratio > self.max_vram_ratio:
                    if current_bs - best_bs > 2:
                        use_fine_search = True
                        current_bs = best_bs + 2
                        print(f"⚠️ Reached VRAM safety limit ({self.max_vram_ratio*100:.0f}%), switch to fine search from BS={current_bs}.")
                        continue
                    print(f"⚠️ Reached VRAM safety limit ({self.max_vram_ratio*100:.0f}%), reverting to previous setting.")
                    break
                elif usage_ratio > self.max_vram_ratio * 0.85 or use_fine_search:
                    best_bs = current_bs
                    current_bs += 2 
                else:
                    best_bs = current_bs
                    current_bs *= 2   # 低使用率時採翻倍，加速逼近可用上限。
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if current_bs - best_bs > 2:
                        use_fine_search = True
                        current_bs = best_bs + 2
                        print(f"❌ OOM occurred! Switching to fine search from BS={current_bs}.")
                        continue
                    print(f"❌ OOM occurred! Reverting to previous setting.")
                    break
                else:
                    raise e # 拋出其他未知的錯誤
                    
        print(f"\n✅ [Stage 1 Result] Best Batch Size: {best_bs}")
        return best_bs

    def find_best_num_workers(self, batch_size):
        """
        Args:
            batch_size (int): 已經找到的最佳 Batch Size。
        Returns:
            (int): 最佳的 num_workers 設定。
        Description:
            這個方法會在找到最佳 Batch Size 後，測試不同的 num_workers 設定，從 0 開始，逐步增加到 CPU 核心數的一半 (或最多 8)，並測量每個設定的數據加載效率（以圖片/秒為單位）。
            當發現 num_workers 的增加不再帶來顯著的效率提升時，會停止測試並返回最佳的 num_workers 設定。
        Note:
            - 在測試過程中，建議密切監控系統資源，特別是在增加 num_workers 時，以避免過度使用 CPU 資源導致系統不穩定或崩潰。
            - num_workers 的最佳值可能因系統配置、數據集大小和存儲設備性能而異。建議從推薦值開始，根據實際情況進行調整。
        """
        
        print("\n" + "="*50)
        print("🚀 Stage 2 : Finding the Best Num Workers")
        print("="*50)
        
        cpu_cores = os.cpu_count() or 4 # 預設至少 4 核心
        
        # 測試不同的 num_workers 設定，從 0 開始，逐步增加到 CPU 核心數的一半 (或最多 8)
        test_workers = [0]
        w = 2
        while w <= max(2, cpu_cores // 2):
            test_workers.append(w)
            w *= 2
        
        best_worker = 0
        best_throughput = 0

        # 第二階段只看 DataLoader 吞吐，固定 eval 模式可減少訓練態波動。
        self.model.eval()
        
        for nw in test_workers:
            trial_throughputs = []
            for _ in range(self.args.num_workers_trials):
                self.clear_memory()
                loader = DataLoader(
                    self.dataset,
                    batch_size=batch_size,
                    num_workers=nw,
                    shuffle=True,
                    pin_memory=(self.device.type == 'cuda'),
                    persistent_workers=(nw > 0)                 # num_workers > 0 時啟用 persistent_workers 以減少 worker 啟動開銷，提升測試效率
                )

                iterator = iter(loader)
                try:
                    next(iterator)  # 預熱一次 DataLoader，確保 num_workers 的影響已經體現出來
                except StopIteration:
                    continue

                # 測試前 N 個 Batch 的加載速度，計算吞吐量 (images/sec)
                start_time = time.perf_counter()
                loaded_batches = 0
                for _ in range(self.args.num_workers_test_batches):
                    try:
                        imgs, masks = next(iterator)    # 只測試前 N 個 Batch 的加載速度
                        imgs = self.api['process_images'](imgs, self.args.model).to(self.device)
                        with torch.no_grad():
                            with autocast(device_type=self.device.type, enabled=self.amp_enabled):
                                _ = self.model(imgs)
                        loaded_batches += 1             # 成功加載的 Batch 數量
                    except StopIteration:
                        break

                elapsed = time.perf_counter() - start_time
                num_images = loaded_batches * batch_size
                throughput = (num_images / elapsed) if elapsed > 0 else 0.0     # 圖片/秒
                trial_throughputs.append(throughput)

            # 計算這個 num_workers 設定的吞吐量中位數，並與目前最佳的吞吐量進行比較。
            throughput = statistics.median(trial_throughputs) if trial_throughputs else 0.0
            trial_text = ', '.join(f"{x:.1f}" for x in trial_throughputs)
            print(f"👉 Testing Num Workers: {nw:2d} | Throughput median: {throughput:5.1f} images/sec | trials: [{trial_text}]")
            
            
            # 如果吞吐量提升超過 5%，就更新最佳設定；否則認為已經達到效率飽和，停止測試。
            if throughput > best_throughput * 1.05:
                best_throughput = throughput
                best_worker = nw
            else:
                print(f"⚠️  Throughput did not improve significantly (diminishing returns), GPU is likely saturated, stopping further increase in workers.")
                break
                
        print(f"\n✅ [Stage 2 Result] Best Num Workers: {best_worker}")
        return best_worker


def get_args():
    """
    Returns:
        argparse.Namespace: 包含從命令列解析的參數。
    Description:
        用於解析命令列參數，並提供一些預設值和說明，並且允許注入來自 train.py 的其他參數，以確保在測試過程中能夠使用與訓練過程相同的模型配置。
    """
    parser = argparse.ArgumentParser(description='自動偵測最佳硬體參數 (Batch Size & Num Workers)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--vram_ratio'                  , type=float, default=0.95              , help='Max VRAM usage ratio')
    parser.add_argument('--src_dir'                     , type=str  , default=DEFAULT_SRC_DIR   , help='Path to source directory that contains dataset/train modules')
    parser.add_argument('--data_dir'                    , type=str  , default=DEFAULT_DATA_DIR  , help='Path to dataset directory')
    parser.add_argument('--dataset_module'              , type=str  , default='oxford_pet'      , help='Dataset module name (must provide OxfordPetDataset)')
    parser.add_argument('--train_module'                , type=str  , default='train'           , help='Train module name (must provide builder helper functions)')
    parser.add_argument('--num_workers_trials'          , type=int  , default=3                 , help='Repeat count for each num_workers test (median is used)')
    parser.add_argument('--num_workers_test_batches'    , type=int  , default=5                 , help='How many batches per trial when measuring num_workers throughput')

    # 注入 train.py 中 _build 函數依賴的預設參數
    parser.add_argument('--model'           , type=str  , default='unet', choices=['unet', 'res_unet'])
    parser.add_argument('--no_amp'          , action='store_true')
    parser.add_argument('--use_bce'         , action='store_true')
    parser.add_argument('--focal_weight'    , type=float, default=0.5)
    parser.add_argument('--bce_weight'      , type=float, default=0.5)
    parser.add_argument('--dice_weight'     , type=float, default=0.5)
    parser.add_argument('--focal_alpha'     , type=float, default=0.5)
    parser.add_argument('--focal_gamma'     , type=float, default=2.0)
    parser.add_argument('--lr'              , type=float, default=5e-4)
    parser.add_argument('--wd'              , type=float, default=1e-4)
    parser.add_argument('--warmup_epochs'   , type=int  , default=5)
    parser.add_argument('--epochs'          , type=int  , default=20)
    
    args, _ = parser.parse_known_args() # 允許在命令列中注入其他參數（例如來自 train.py 的參數），但不會因為未定義的參數而報錯
    return args

if __name__ == "__main__":
    args = get_args()

    # 加載專案的 Dataset 類別和訓練相關的函數，確保在測試過程中使用與訓練過程相同的模型配置和數據處理邏輯。
    project_api = load_project_components(args.src_dir, args.dataset_module, args.train_module, symbol_map=DEFAULT_SYMBOL_MAP)
    
    # 確認 dataset 目錄存在。
    if not os.path.exists(args.data_dir):
        print(f"❌ Could not find dataset directory: {args.data_dir}")
        sys.exit(1)
    
    # 初始化 Dataset 實例。
    print(f"[System Initialization] Successfully located dataset directory: {args.data_dir}")
    train_dataset = project_api['Dataset'](args.data_dir, split="train")
        
    # 啟動自動調校器。
    tuner = HardwareTuner(args=args, dataset=train_dataset, project_api=project_api, max_vram_ratio=args.vram_ratio)    
    
    # 執行第一階段：尋找極限 Batch Size。
    optimal_bs = tuner.find_best_batch_size(starting_bs=2)
    
    # 執行第二階段：尋找最有效率的 Num Workers。
    optimal_nw = tuner.find_best_num_workers(batch_size=optimal_bs)
    
    print("\n" + "="*25)
    print("🎉 Optimal Hardware Parameters Found! 🎉")
    print(f" --batch_size {optimal_bs}")
    print(f" --num_workers {optimal_nw} ")
    print("="*25 + "\n")