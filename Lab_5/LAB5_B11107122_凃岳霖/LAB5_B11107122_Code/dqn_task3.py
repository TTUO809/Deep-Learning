# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, num_actions, input_channels=None):
        super(DQN, self).__init__()
        ########## YOUR CODE HERE (5~10 lines) ##########
        self.is_atari = input_channels is not None

        if self.is_atari:
            # Atari CNN（Task 2 / Task 3）: input_channels=4 (stacked 4 frames), output=num_actions=6 (Pong action space)
            self.network = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),             nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),             nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
                nn.Linear(512, num_actions),
            )
        else:
            # CartPole FC（Task 1）: input_dim=4 (state dimension), output=num_actions=2 (CartPole action space)
            self.network = nn.Sequential(
                nn.Linear(4, 128),   nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, num_actions),
            )
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        if self.is_atari:   # Normalize pixel values for Atari input
            x = x / 255.0
        return self.network(x)


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class SumTree:
    """
    Binary segment tree backing PER. Each internal node stores the sum of its
    children's priorities; leaves store the priorities of individual transitions.

    Layout (flat array of size 2*capacity - 1):
        tree[0]                          ← root (= total priority)
        tree[1..capacity-2]              ← internal nodes
        tree[capacity-1..2*capacity-2]   ← leaves, one per transition slot

    Index math:
        parent(i)      = (i - 1) // 2
        left_child(i)  = 2*i + 1
        right_child(i) = 2*i + 2
        data slot k    ↔ tree index (k + capacity - 1)

    Both update (leaf → root) and get_leaf (root → leaf) are O(log N).
    """
    def __init__(self, capacity):
        self.capacity = capacity
        # 建立一維陣列來模擬二元樹，長度為 2 * capacity - 1。
        # 前半段 (1 到 capacity-2) 是內部節點（存子節點的優先度總和）
        # 後半段 (capacity-1 到 2*capacity-2) 是葉節點（存每個樣本真實的優先度）
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)

        # 建立一個陣列來存真正的資料（即 transition: s, a, r, s', done），型態為 object
        self.data = np.empty(capacity, dtype=object)

        self.write = 0      # 寫入指標（Ring buffer 的概念，滿了會從 0 覆蓋）
        self.n_entries = 0  # 記錄目前存了幾筆資料

    def _propagate(self, idx, change):
        # 當某個葉節點的優先度改變時，這個改變（change）必須像泡泡一樣往上傳遞到根節點
        while idx != 0:
            idx = (idx - 1) // 2        # 找到父節點的 index
            self.tree[idx] += change    # 父節點的優先度總和要加上這個改變值

    def _retrieve(self, s):
        # s 是一個介於 0 到「總優先度」之間的隨機亂數。這函式負責找出這個 s 落在哪個葉節點。
        idx = 0     # 從根節點開始往下找
        while True:
            left = 2 * idx + 1  # 左子節點的 index
            if left >= len(self.tree):
                return idx      # 如果左子節點超出陣列長度，代表現在已經到底（葉節點）了，回傳！
        
            right = left + 1    # 右子節點的 index

            # 如果 s 小於左節點的值，代表目標在左半邊(因為左子節點的優先度總和包含了所有左半邊的優先度)，就往左子節點繼續往下找
            if s <= self.tree[left]:
                idx = left      # 往左子節點繼續往下找
            else:
                s -= self.tree[left]    # 從 s 中扣掉左子節點的優先度，因為要找的是剩下的部分
                idx = right     # 往右子節點繼續往下找

    def total(self):
        return float(self.tree[0]) # tree[0] = 根節點，存著全場優先度的總和。

    def add(self, priority, data):
        # 計算在 tree 陣列中的葉節點 index (要加上 capacity - 1 的偏移量)
        idx = self.write + self.capacity - 1

        self.data[self.write] = data    # 把 transition 存進 data 陣列
        self.update(idx, priority)      # 更新 SumTree 的優先度（會自動 propagate）

        self.write = (self.write + 1) % self.capacity   # 更新寫入指標，滿了就從頭開始覆蓋
        if self.n_entries < self.capacity:
            self.n_entries += 1     # 記錄目前存了幾筆資料（最多到 capacity）

    def update(self, idx, priority):
        change = priority - self.tree[idx]  # 計算優先度的改變值（新的優先度 - 舊的優先度）
        self.tree[idx] = priority           # 更新這個葉節點的優先度
        self._propagate(idx, change)        # 把這個改變值往上傳遞到根節點，讓所有包含這個葉節點的父節點的優先度總和都更新

    def get_leaf(self, s):
        idx = self._retrieve(s)             # 根據 s 找到對應的葉節點 index
        data_idx = idx - self.capacity + 1  # 計算對應的 data index（因為葉節點在 tree 陣列中的 index 是從 capacity-1 開始的，所以要扣掉 capacity - 1）
        return idx, float(self.tree[idx]), self.data[data_idx]  # 回傳葉節點的 index、優先度值、以及對應的 transition 資料

    def __len__(self):
        return self.n_entries


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952

        SumTree-backed implementation: add / sample / update are all O(log N).
        Sample returns (samples, indices, weights), where `indices` are SumTree
        node indices and must be passed back to `update_priorities` unchanged.
    """ 
    eps = 1e-6

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # 決定我們要多「依賴」優先度 (0=完全隨機, 1=完全看優先度)
        self.beta = beta    # IS 權重的校正強度 (0=不校正, 1=完全校正)
        self.tree = SumTree(capacity)
        self.max_priority = 1.0  # 新增 transition 的初始優先度（可以設為目前的 max priority，確保新 transition 至少被抽到一次）

    def add(self, transition, error=None):
        ########## YOUR CODE HERE (for Task 3) ##########
        # 新的經驗還沒有被訓練過，不知道 TD error 多少。
        # 為了確保它「至少被抽中訓練一次」，直接賦予它「目前全場最高」的優先度！
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)
        ########## END OF YOUR CODE (for Task 3) ##########
        return

    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ##########
        N = len(self.tree)
        total = self.tree.total()
        segment = total / batch_size    # 把總優先度分成 batch_size 等份，每次從其中一份隨機抽一個 s，確保整個優先度範圍都被覆蓋到

        samples = []    # 實際抽到的 transition 樣本
        indices = np.empty(batch_size, dtype=np.int64)  # 對應的 SumTree 葉節點 index（用來之後更新優先度）
        priorities = np.empty(batch_size, dtype=np.float64) # 抽到的 transition 的優先度值（用來計算 IS 權重）

        for i in range(batch_size):
            # 從 segment * i 到 segment * (i + 1) 的區間內隨機抽一個 s，這樣可以確保整個優先度範圍都被覆蓋到，不會只集中在高優先度的區域
            s = np.random.uniform(segment * i, segment * (i + 1))
            # 避免 s 落在 total 的邊界上（因為 get_leaf 需要 s < total），所以稍微減一點 epsilon
            s = min(s, total - self.eps)
            # 根據 s 找到對應的葉節點 index、優先度值、以及 transition 資料
            idx, p, data = self.tree.get_leaf(s)
            # 防呆：如果因為數值問題導致 p <= 0 或 data 是 None，就重新抽一個 s，直到抽到有效的樣本
            while data is None or p <= 0:
                s = np.random.uniform(0, total - self.eps)
                idx, p, data = self.tree.get_leaf(s)

            samples.append(data)
            indices[i] = idx
            priorities[i] = p

        # 計算優先度對應的抽樣機率，然後根據 beta 計算 IS 權重，最後做個 normalize（除以 max weight）讓它們不會太大
        probs = priorities / total  # P(i) = p_i / p_total
        weights = (N * probs) ** (-self.beta)   # w_i = (N * P(i))^(-beta)
        weights /= weights.max()
        ########## END OF YOUR CODE (for Task 3) ##########
        return samples, indices, weights.astype(np.float32)

    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ##########
        # 根據 TD error 更新對應 index 的優先度。
        # 優先度 = (|error| + eps)^alpha，並且更新 max_priority 以便新加入的 transition 可以有適當的初始優先度。
        for idx, err in zip(indices, errors):
            raw = abs(float(err)) + self.eps
            self.tree.update(int(idx), raw ** self.alpha)   # P_i^alpha = (|error| + eps)^alpha

            if raw > self.max_priority:
                self.max_priority = raw
        ########## END OF YOUR CODE (for Task 3) ##########
        return

    def __len__(self):
        return len(self.tree)


class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.seed = args.seed
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.env.action_space.seed(self.seed)
        self.test_env.action_space.seed(self.seed)
        self.num_actions = self.env.action_space.n
        self.train_preprocessor = AtariPreprocessor()
        self.test_preprocessor  = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # Detect Atari vs CartPole from env_name
        self.is_atari = "ALE" in env_name or "Pong" in env_name

        input_channels = 4 if self.is_atari else None
        self.q_net = DQN(self.num_actions, input_channels=input_channels).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions, input_channels=input_channels).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr, eps=1.5e-4)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Replay buffer — uniform or PER
        self.use_per = args.use_per
        if self.use_per:
            # Prioritized replay buffer: use the SumTree-backed implementation defined above
            self.memory = PrioritizedReplayBuffer(
                capacity=args.memory_size,
                alpha=args.per_alpha,
                beta=args.per_beta
            )
        else:
            # Uniform replay buffer: simple deque with maxlen for fixed capacity
            self.memory = deque(maxlen=args.memory_size)

        # Multi-step return
        self.n_step = args.n_step
        self.n_step_buffer = deque(maxlen=self.n_step)

        # Double DQN flag
        self.use_double = args.use_double

        # best_reward init: 0 for CartPole, -21 for Pong
        self.best_reward = 0 if not self.is_atari else -21

        # Beta annealing: linearly from initial beta to 1.0 over per_beta_steps env steps
        self.per_beta_steps = args.per_beta_steps

    def _get_n_step_info(self):
        """為了記住連續的 n 步（存在 self.n_step_buffer 裡），打包成一筆等效的 (state, action, reward, next_state, done) 經驗，讓神經網路可以一次學到更長遠的結果。"""
        # 取出 n 步後的 reward、next_state、done
        reward, next_state, done = (self.n_step_buffer[-1][2],
                                    self.n_step_buffer[-1][3],
                                    self.n_step_buffer[-1][4])
        # 從後往前把 n 步的 reward 加總起來
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, ns, d = transition[2], transition[3], transition[4]
            reward = r + self.gamma * reward * (1 - d)  # R = r_0 + \gamma * R_{n-1}，如果 d=1（done），後續 reward 就不再加了
            # 如果在 n 步內就已經 done 了，那麼 next_state 和 done 就要更新成最早的那個終止狀態，因為這筆經驗的結果就是在那裡結束了。
            if d:
                next_state, done = ns, d
        # state 和 action 就是 n 步前的那個 state 和 action，
        # reward、next_state、done 是 n 步後的結果，這樣神經網路就可以學到從 state 出發，
        # 執行 action 後 n 步的累積 reward 和最終結果。
        state  = self.n_step_buffer[0][0]
        action = self.n_step_buffer[0][1]
        return state, action, reward, next_state, done

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=1000):
        for ep in range(episodes):
            # 重置環境，第一集使用固定 seed 以確保 reproducibility，後續集數則不設 seed 讓環境隨機演化
            obs, _ = self.env.reset(seed=self.seed if ep == 0 else None)

            # 如果是 Atari 遊戲，隨機執行 0-30 次「無操作」，增加初始狀態的隨機性。
            # if self.is_atari:
            #     noop_steps = random.randint(0, 30)
            #     for _ in range(noop_steps):
            #         obs, _, term, trunc, _ = self.env.step(0)   # 0 是 Atari 的 NOOP action
            #         if term or trunc:   # 如果在無操作過程中就已經 done 了，立刻重新 reset，確保正式訓練前的初始狀態是有效的。
            #             obs, _ = self.env.reset(seed=None)
            #             break
            
            # 根據環境類型選擇適當的 state 表示：CartPole 就直接用原始觀測值，Atari 就用預處理器把畫面轉成灰階、縮放、堆疊成 4 張圖的形式。
            if self.is_atari:
                state = self.train_preprocessor.reset(obs)
            else:
                state = obs
            # 每一集開始都要清空 n_step_buffer，確保它只記錄當前集數的 n 步經驗，避免跨集數混淆。
            self.n_step_buffer.clear()
            done = False
            total_reward = 0
            step_count = 0

            # 在一集的主循環中，持續選擇動作、與環境互動、收集經驗、訓練神經網路，直到 episode 結束或達到最大步數。
            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)  # 根據 epsilon-greedy 策略選擇動作
                next_obs, reward, terminated, truncated, _ = self.env.step(action)  # 與環境互動，得到下一個觀測值、獎勵、以及 episode 是否結束的訊號
                done = terminated or truncated  # episode 結束的條件：要麼是遊戲結束（terminated），要麼是達到時間限制（truncated）
                
                if self.is_atari:
                    next_state = self.train_preprocessor.step(next_obs)
                else:
                    next_state = next_obs

                # 把這一步的經驗 (state, action, reward, next_state, done) 存到 n_step_buffer 裡。
                self.n_step_buffer.append((state, action, reward, next_state, done))
                # 當 n_step_buffer 滿了 n 步，就把它打包成一筆等效的 (state, action, reward, next_state, done) 經驗，
                # 存到 replay buffer 裡，讓神經網路可以學到更長遠的結果。
                if len(self.n_step_buffer) == self.n_step:
                    s, a, r, ns, d = self._get_n_step_info()
                    transition = (s, a, r, ns, d)
                    if self.use_per:
                        # 新 transition 還沒有被訓練過，不知道 TD error 多少。為了確保它「至少被抽中訓練一次」，直接賦予它「目前全場最高」的優先度。
                        self.memory.add(transition, error=1.0)
                    else:
                        self.memory.append(transition)

                # 每一步都訓練 train_per_step 次神經網路，這樣可以讓 agent 更快地從收集到的經驗中學習。
                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                    
                    ########## END OF YOUR CODE ##########   

            # 當 Episode 結束，清空 n_step_buffer 中剩餘的資料
            if len(self.n_step_buffer) == self.n_step:
                # 如果 n_step_buffer 已經滿了 n 步了，代表最後一筆 transition 已經在主循環中存到 replay buffer 裡了，
                # 所以這裡就不用再存一次了，直接把它從 buffer 裡丟掉，然後再處理剩下的部分。
                self.n_step_buffer.popleft()
            while len(self.n_step_buffer) > 0:
                # 剩下的資料雖然湊不滿 n 步，但依然包含重要的獎勵資訊，應該要存到 replay buffer 裡讓神經網路學習到 episode 結束的結果。
                s, a, r, ns, d = self._get_n_step_info()
                transition = (s, a, r, ns, d)
                if self.use_per:
                    self.memory.add(transition, error=1.0)
                else:
                    self.memory.append(transition)
                self.n_step_buffer.popleft()    # 處理完最早的一筆 transition 後，就把它從 buffer 裡丟掉，繼續處理下一筆，直到 buffer 清空。

            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########  
            if ep % 100 == 0:
                # 每 100 集保存一次模型 checkpoint，並且在檔名中標註集數，方便之後回顧訓練過程或做進一步分析。
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                # 每 20 集進行一次真正的評估（evaluate），在測試環境上跑 5 個 episode，計算平均 reward，
                # 並且如果這次的評估結果比之前的 best_reward 還好，就保存這個新的 best model。
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

            # Milestone auto-saving
            for ms in [600_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000]:
                attr = f"_saved_{ms}"   # 用一個動態屬性來記錄是否已經在這個里程碑存過模型了，避免同一個里程碑存多次
                if self.env_count >= ms and not getattr(self, attr, False):
                    setattr(self, attr, True)
                    mpath = os.path.join(self.save_dir, f"model_{ms}.pt")
                    torch.save(self.q_net.state_dict(), mpath)  # 每當環境步數達到某個里程碑（例如 600k、1M、1.5M 等），就自動保存一次模型快照，並且在檔名中標註里程碑的步數，方便之後回顧訓練過程或做進一步分析。
                    print(f"[Milestone] Saved {ms} steps snapshot → {mpath}")

    def evaluate(self, num_seeds=20):
        rewards = []
        # 在測試環境上跑 num_seeds 個 episode，計算平均 reward，並且回傳這個平均 reward 作為評估結果。
        for seed in range(num_seeds):
            obs, _ = self.test_env.reset(seed=seed)
            if self.is_atari:
                state = self.test_preprocessor.reset(obs)
            else:
                state = obs
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.from_numpy(np.array(state)).unsqueeze(0).to(self.device).float()
                with torch.no_grad():
                    action = self.q_net(state_tensor).argmax().item()
                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated
                total_reward += reward
                if self.is_atari:
                    state = self.test_preprocessor.step(next_obs)
                else:
                    state = next_obs

            rewards.append(total_reward)

        return float(np.mean(rewards))


    def train(self):

        # 如果 replay buffer 中的樣本數量還沒有達到 replay_start_size，
        # 就先不訓練，繼續收集經驗，確保神經網路在有足夠多樣化的經驗後再開始學習，避免一開始就被少數幾筆經驗誤導。
        buf_len = len(self.memory)
        if buf_len < self.replay_start_size:
            return 
        
        # 隨著訓練的進行，epsilon 會慢慢衰減，但不會低於 epsilon_min，
        # 這樣 agent 就會從一開始的高度探索逐漸過渡到更多利用已學到的知識，
        # 同時保留一定程度的探索性，避免完全陷入局部最優解。
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########
        # 當使用 PER 時，從優先經驗回放緩衝區中抽取一個 mini-batch 的樣本，並且獲取對應的 indices 和 IS 權重；
        # 當使用普通的 uniform replay buffer 時，直接從緩衝區中隨機抽取一個 mini-batch 的樣本。
        if self.use_per:
            # 隨著環境步數的增加，beta 會從初始值逐漸線性增加到 1.0，這樣在訓練的後期就會完全校正優先抽樣帶來的偏差，讓學習更穩定。
            self.memory.beta = min(1.0, 0.4 + 0.6 * (self.env_count / self.per_beta_steps))
            samples, indices, is_weights = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states, dones = zip(*samples)
            is_weights = torch.from_numpy(is_weights).to(self.device)
        else:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            indices, is_weights = None, None
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates 
        with torch.no_grad():
            if self.use_double:
                # Double DQN: main net selects action, target net evaluates it
                best_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            else:
                # Vanilla DQN: max Q from target net
                next_q = self.target_net(next_states).max(1)[0]

            target_q = rewards + (self.gamma ** self.n_step) * next_q * (1 - dones)

        # 算 TD errors（target Q - current Q），這個值不但用來計算 loss，
        # 還會用來更新 PER 中的優先度，讓 TD error 大的樣本更有機會被抽到訓練。
        td_errors = target_q - q_values

        if self.use_per:
            # PER 路徑用 element-wise Huber loss，與 vanilla 路徑一致。
            # 用 MSE (td_errors**2) 會讓高 TD error 樣本的梯度過大，加上 PER 本來就偏向高 error 樣本，
            # 兩者疊加容易造成訓練不穩定。改用 Huber 可以截斷異常大的梯度。
            loss = (is_weights * F.smooth_l1_loss(q_values, target_q, reduction='none')).mean()
            self.memory.update_priorities(indices, td_errors.abs().detach().cpu().numpy())
        else:
            # Uniform replay loss: standard Huber loss
            loss = F.smooth_l1_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # 為了避免梯度爆炸，對所有可訓練參數的梯度進行裁剪，確保它們的 L2 範數不超過 10，這樣可以讓訓練更穩定。
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10)
        self.optimizer.step()
        ########## END OF YOUR CODE ##########  

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")
            
            wandb.log({
                "Train/Loss": loss.item(),
                "Train/Q_mean": q_values.mean().item(),
                "Train/Q_std": q_values.std().item()
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999999)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--env",        type=str,   default="CartPole-v1")
    parser.add_argument("--use-double", action="store_true")
    parser.add_argument("--use-per",    action="store_true")
    parser.add_argument("--per-alpha",      type=float, default=0.6)
    parser.add_argument("--per-beta",       type=float, default=0.4)
    parser.add_argument("--per-beta-steps", type=int,   default=600_000)
    parser.add_argument("--n-step",         type=int,   default=1)
    parser.add_argument("--episodes",   type=int,   default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wandb.init(project="DLP-Lab5-DQN-CartPole", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(env_name=args.env, args=args)
    agent.run(episodes=args.episodes)