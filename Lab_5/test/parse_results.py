import os
import json
import re
import pandas as pd
import subprocess

def clean_ansi(text):
    """
    清除 Log 檔案中因為進度條或終端機顏色產生的 ANSI 控制碼。
    這對於使用正則表達式 (Regex) 精準抓取文字非常重要，否則隱藏的控制碼會導致比對失敗。
    """
    # 建立一個正則表達式物件，用於匹配標準的 ANSI 控制碼格式
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    # 將找到的控制碼替換為空字串
    return ansi_escape.sub('', text)

def format_val(val, is_default=False):
    """
    格式化輸出到 CSV/Excel 的數值。
    主要解決兩個 Excel 的惱人問題：
    1. 避免大於 10 萬的數字變成科學記號 (如 1E+05)。
    2. 如果是預設值，加上中括號 [ ] 方便視覺辨識 (使用 ( ) 會被 Excel 當成負數)。
    """
    # 如果值是空的或 NaN，回傳 '-' 保持版面整潔
    if pd.isna(val) or val is None: return "-"
    
    # 如果是數字且大於等於 100,000，強制轉為不帶小數點的字串，避免 Excel 自動轉成科學記號
    str_val = f"{int(val)}" if isinstance(val, (int, float)) and val >= 100000 else str(val)
    
    # 如果這個值是系統預設值 (不是使用者下指令改的)，就在外面包上中括號
    return f"[{str_val}]" if is_default else str_val

def extract_task_name(raw_name):
    """
    從 wandb-run-name (例如: ablation_nstep_only_20260503_134632) 
    中提取乾淨的 Task 名稱。
    """
    # 如果沒有提供名字，回傳 unknown
    if not raw_name: return "[unknown]"
    
    # 把結尾的時間戳記從原本的名字中刪除，留下乾淨的 Task 名稱
    task_name = re.sub(r'_\d{8}_\d{6}$', '', raw_name)
    
    return task_name

def parse_all_args(args_list):
    """
    解析 wandb 記錄的指令參數 (args_list)。
    這會把輸入的指令陣列轉換成字典，並區分哪些是你手動設定的，哪些是預設值。
    """
    # 定義你程式中 argparse 的預設值
    defaults = {
        "Env": "CartPole-v1", "Batch": 32, "Memory": 100000, "LR": 0.0001,
        "Discount": 0.99, "Eps Start": 1.0, "Eps Decay": 0.999999, "Eps Min": 0.05,
        "Target Update": 1000, "Replay Start": 50000, "Max Eps Steps": 10000,
        "Train/Step": 1, "PER Alpha": 0.6, "PER Beta": 0.4, "PER Beta Steps": 600000,
        "N-Step": 1, "Episodes": 10000, "DDQN": "False", "PER": "False"
    }
    
    # 定義指令 flag (例如 --lr) 對應到輸出表格的欄位名稱 (例如 LR)
    mapping = {
        "--env": "Env", "--batch-size": "Batch", "--memory-size": "Memory",
        "--lr": "LR", "--discount-factor": "Discount", "--epsilon-start": "Eps Start",
        "--epsilon-decay": "Eps Decay", "--epsilon-min": "Eps Min",
        "--target-update-frequency": "Target Update", "--replay-start-size": "Replay Start",
        "--max-episode-steps": "Max Eps Steps", "--train-per-step": "Train/Step",
        "--per-alpha": "PER Alpha", "--per-beta": "PER Beta",
        "--per-beta-steps": "PER Beta Steps", "--n-step": "N-Step", "--episodes": "Episodes"
    }
    
    # 初始化一個字典，先把所有預設值填進去，並使用 format_val 標記為 is_default=True (會加上 [ ])
    results = {k: format_val(v, is_default=True) for k, v in defaults.items()}
    # 記錄每個參數是否有被手動指定
    explicit = {k: False for k in defaults.keys()}

    # 檢查是否有啟用 Double DQN (--use-double 或相容舊版的 --use-ddqn)
    if "--use-double" in args_list or "--use-ddqn" in args_list:
        results["DDQN"], explicit["DDQN"] = "True", True
    
    # 檢查是否有啟用 PER (--use-per)
    if "--use-per" in args_list:
        results["PER"], explicit["PER"] = "True", True

    # 迴圈檢查每個 Key-Value 參數
    for flag, key in mapping.items():
        if flag in args_list:
            try:
                # 找到 flag 的位置，它的下一個元素就是設定的值
                val = args_list[args_list.index(flag) + 1]
                # 覆蓋原本帶括號的預設值，改為 is_default=False (不帶括號)
                results[key] = format_val(val, is_default=False)
                explicit[key] = True
            except: pass
    
    # 嘗試抓取 wandb-run-name，如果沒有則回傳空字串
    raw_name = args_list[args_list.index("--wandb-run-name") + 1] if "--wandb-run-name" in args_list else ""

    # 擷取模型儲存目錄
    save_dir = ""
    if "--save-dir" in args_list:
        try:
            save_dir = args_list[args_list.index("--save-dir") + 1]
        except: pass
        
    return results, raw_name, save_dir

def evaluate_via_shell(save_dir, milestones):
    """
    使用 subprocess 直接執行 run_task3.sh 並解析其即時輸出。
    """
    print(f"  [Auto-Eval] 正在啟動 Shell 評估目錄: {save_dir} ... (可能需要幾分鐘)")
    cmd = ["bash", "run_task3.sh", save_dir, "--eval-only"]
    
    ms_results = {}
    best_reward = None
    
    try:
        # capture_output=True 會抓取標準輸出，check=False 以免找不到 pt 檔時腳本崩潰
        result = subprocess.run(cmd, capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        snapshot_pattern = re.compile(r"--- Snapshot:\s*(\d+)\s*env steps ---")
        reward_pattern = re.compile(r"Average reward:\s*([\d.]+)")
        
        current_ms = None
        is_best_model = False
        
        for line in lines:
            line = clean_ansi(line)
            
            # 判斷是否進入 Best Model 評估階段
            if "[3/4] Evaluating best_model.pt" in line:
                is_best_model = True
                continue
                
            # 尋找 Snapshot 宣告
            ms_match = snapshot_pattern.search(line)
            if ms_match:
                current_ms = int(ms_match.group(1))
                is_best_model = False
                continue
                
            # 尋找 Average Reward
            rw_match = reward_pattern.search(line)
            if rw_match:
                avg_reward = float(rw_match.group(1))
                
                if is_best_model:
                    best_reward = avg_reward
                elif current_ms in milestones:
                    ms_results[current_ms] = "PASS" if avg_reward >= 19.0 else f"Fail({avg_reward})"
                    current_ms = None
                    
    except Exception as e:
        print(f"  [Auto-Eval Error] 執行 shell 評估失敗: {save_dir}, 錯誤: {e}")
        
    return ms_results, best_reward

def parse_logs(base_dir="wandb"):
    """
    主程式：讀取 base_dir (預設為 wandb 資料夾) 下的所有實驗紀錄。
    """
    experiment_data = [] # 用來收集所有實驗紀錄的列表
    # 正則表達式：用於匹配 wandb 產生的資料夾名稱格式 run-YYYYMMDD_HHMMSS-隨機ID
    folder_pattern = re.compile(r'run-(\d{8}_\d{6})-(.*)')
    # 定義要追蹤的里程碑步數
    milestones = [600000, 1000000, 1500000, 2000000, 2500000]

    # 如果找不到目標資料夾，直接結束
    if not os.path.exists(base_dir): return
    
    # 遍歷 base_dir 裡面的所有檔案和資料夾
    for folder in os.listdir(base_dir):
        # 檢查該項目是否符合 run-YYYYMMDD_HHMMSS-ID 的格式
        m_folder = folder_pattern.match(folder)
        if not m_folder: continue # 不符合就跳過
        
        # 組合出存放 json 和 log 檔案的 files 資料夾路徑
        folder_path = os.path.join(base_dir, folder, "files")
        
        # 初始化這筆實驗的字典，強制 Timestamp 以資料夾名稱擷取的時間為準
        folder_time = m_folder.group(1)
        record = {"Folder Time": folder_time, "Timestamp": folder_time}
        
        # --- 1. 解析參數與名稱 (從 wandb-metadata.json) ---
        meta_json = os.path.join(folder_path, "wandb-metadata.json")
        raw_name = ""
        save_dir = ""
        if os.path.exists(meta_json):
            with open(meta_json, 'r') as f:
                meta = json.load(f) # 讀取 JSON
                # 呼叫 parse_all_args 函數解析 args 陣列
                args_results, raw_name, save_dir = parse_all_args(meta.get("args", []))
                # 將解析出來的參數合併到 record 字典中
                record.update(args_results)
        
        # 呼叫 extract_task_name 處理名字 (時間戳記已經從資料夾取得了)
        record["Task"] = extract_task_name(raw_name)

        # --- 2. 解析分數 (從 output.log) ---
        log_file = os.path.join(folder_path, "output.log")
        best_r, best_sc = -999.0, 0     # 初始化最高分與對應的步數
        reached_19_sc = None            # 初始化達到 19 分的步數
        
        ms_status = {m: "-" for m in milestones} # 初始化里程碑狀態
        ms_closest_diff = {m: float('inf') for m in milestones} # 追蹤最接近里程碑的步數差
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                current_sc = 0 # 追蹤當前讀取到的環境步數 (Step Count)
                for line in f:
                    # 先清除 ANSI 控制碼，確保正則表達式能正確比對
                    line = clean_ansi(line)
                    
                    # 隨時更新目前的環境步數 SC
                    sc_match = re.search(r'SC:\s*(\d+)', line)
                    if sc_match: current_sc = int(sc_match.group(1))

                    # === 精準抓取「存檔」的最佳分數 ===
                    # 優先尋找 "Saved new best model ... reward [數值]" 這行字
                    # 這樣就能避免抓到單一回合偶然出現的高分，確保抓到的是模型穩定進步的分數
                    best_match = re.search(r'Saved new best model.*?reward\s*([-\d.]+)', line)
                    if best_match:
                        # 更新最佳分數
                        best_r = float(best_match.group(1))
                        # 將最佳分數與當前記錄到的 SC 綁定
                        best_sc = current_sc

                    # === 判定 19 分達標與里程碑 ===
                    # 為了避免抓到單一 Episode 波動極大的 Reward，
                    # 這裡我們只認列被記錄為 "[TrueEval]" 或正式存檔點後的分數作為里程碑和達標依據
                    true_eval_match = re.search(r'\[TrueEval\].*?Reward:\s*([-\d.]+)\s*SC:\s*(\d+)', line)
                    
                    eval_r, eval_sc = None, None
                    
                    if true_eval_match:
                        # [TrueEval] 評估的分數
                        eval_r, eval_sc = float(true_eval_match.group(1)), int(true_eval_match.group(2))
                    elif best_match:
                         # 如果剛好觸發儲存新模型，那這個分數也是真實且經過 Eval 驗證的
                         eval_r, eval_sc = float(best_match.group(1)), current_sc
                    elif "cartpole" in str(record.get("Env", "")).lower():
                         # CartPole 可能沒有 TrueEval 或是 Saved best，退回使用普通 Reward 判斷
                         m = re.search(r'Reward:\s*([-\d.]+)\s*SC:\s*(\d+)', line)
                         if m:
                             eval_r, eval_sc = float(m.group(1)), int(m.group(2))
                             if eval_r > best_r: best_r, best_sc = eval_r, eval_sc

                    # 進行判定
                    if eval_r is not None and eval_sc is not None:
                        # 檢查是否首次達到 19 分 (僅限 Atari)
                        if eval_r >= 19.0 and reached_19_sc is None:
                            reached_19_sc = eval_sc
                        
                        # 檢查里程碑
                        for ms in milestones:
                            diff = abs(eval_sc - ms)
                            # 尋找距離該里程碑「步數最接近」的真實評估點 (容許在前後 50000 步內的尋找)
                            # 因為 Snapshot 雖然在 600k 存檔，但真正的 Eval 可能在 605k 或是 595k
                            if diff < ms_closest_diff[ms] and diff < 50000: 
                                ms_closest_diff[ms] = diff
                                ms_status[ms] = "PASS" if eval_r >= 19.0 else f"Fail({eval_r})"

        # --- 3. 觸發精準的 Shell 自動評估 (覆蓋 Log 結果) ---
        is_atari = "ale" in str(record.get("Env", "")).lower() or "pong" in str(record.get("Env", "")).lower()
        
        # 只有在確認是 Atari 環境，且確實有儲存目錄時才執行
        if is_atari and save_dir and os.path.exists(save_dir):
            print(f"[{folder_time}] 偵測到 Task 3 ({record['Task']})，準備進行精準驗證...")
            shell_ms_results, shell_best_reward = evaluate_via_shell(save_dir, milestones)
            
            # 使用 shell 的精準結果覆蓋
            for ms, status in shell_ms_results.items():
                ms_status[ms] = status
                
            if shell_best_reward is not None:
                # 更新最佳分數，若大於 19 且先前未判定達標，這裡可以補強
                best_r = max(best_r, shell_best_reward)
                if best_r >= 19.0 and not reached_19_sc:
                    reached_19_sc = "Reached in Final Eval"

        # 將解析結果寫入 record 字典
        record["Best Reward"], record["At Step"] = best_r, best_sc
        record["19-Score Step"] = reached_19_sc if reached_19_sc else "Not Reached"
        
        # --- 3. 達標判定 (Status) ---
        # 判斷是否為 Atari 環境
        if not is_atari: # 如果是 CartPole
            record["Status"] = "✅ PASS" if best_r >= 480 else "Fail"
        else: # 如果是 Atari
            record["Status"] = "🎯 REACHED" if reached_19_sc else "Learning"

        # 將里程碑狀態合併到 record 中
        for ms in milestones: record[f"MS_{ms}"] = ms_status[ms]
        
        # 將這筆實驗記錄加入總列表
        experiment_data.append(record)

    # --- 4. 排序與輸出 ---
    # 將字典列表轉換為 Pandas DataFrame，並依照資料夾時間排序
    df = pd.DataFrame(experiment_data).sort_values("Timestamp")
    
    # 定義輸出到 CSV 的欄位順序
    cols = ["Task", "Timestamp", "Status", "Env", "DDQN", "PER", "N-Step", "LR", "Batch", "Memory", "Discount", 
            "Eps Start", "Eps Decay", "Eps Min", "Target Update", "Replay Start", "Max Eps Steps", "Train/Step", 
            "PER Alpha", "PER Beta", "PER Beta Steps", "Episodes", "Best Reward", "At Step", "19-Score Step"] + \
           [f"MS_{m}" for m in milestones]
    
    # 篩選並排列 DataFrame 的欄位
    df[cols].to_csv("final_experiment_report.csv", index=False, encoding='utf-8-sig')
    print("結果存於 final_experiment_report.csv")
    # 印出預覽畫面到終端機 (使用 Markdown 表格格式)
    print(df[["Task", "Timestamp", "Best Reward", "At Step"]].tail(5).to_markdown(index=False))

# 程式進入點
if __name__ == "__main__":
    parse_logs()
