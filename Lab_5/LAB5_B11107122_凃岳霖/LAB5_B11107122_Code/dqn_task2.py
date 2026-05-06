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


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition, error):
        ########## YOUR CODE HERE (for Task 3) ########## 
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        ########## END OF YOUR CODE (for Task 3) ##########
        return

    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ##########
        N = len(self.buffer)
        prios = self.priorities[:N]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(N, batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        weights = (N * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights
        ########## END OF YOUR CODE (for Task 3) ##########

    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ##########
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + 1e-6
        ########## END OF YOUR CODE (for Task 3) ##########
        return

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.seed = getattr(args, 'seed', 42)
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
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

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
        self.use_per = getattr(args, 'use_per', False)
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(
                capacity=args.memory_size,
                alpha=getattr(args, 'per_alpha', 0.6),
                beta=getattr(args, 'per_beta', 0.4)
            )
        else:
            self.memory = deque(maxlen=args.memory_size)

        # Multi-step return
        self.n_step = getattr(args, 'n_step', 1)
        self.n_step_buffer = deque(maxlen=self.n_step)

        # Double DQN flag
        self.use_double = getattr(args, 'use_double', False)

        # best_reward init: 0 for CartPole, -21 for Pong
        self.best_reward = 0 if not self.is_atari else -21

    def _get_n_step_info(self):
        """Compute n-step discounted return from self.n_step_buffer."""
        reward, next_state, done = (self.n_step_buffer[-1][2],
                                    self.n_step_buffer[-1][3],
                                    self.n_step_buffer[-1][4])
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, ns, d = transition[2], transition[3], transition[4]
            reward = r + self.gamma * reward * (1 - d)
            if d:
                next_state, done = ns, d
        state  = self.n_step_buffer[0][0]
        action = self.n_step_buffer[0][1]
        return state, action, reward, next_state, done

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=1000):
        for ep in range(episodes):
            obs, _ = self.env.reset(seed=self.seed if ep == 0 else None)

            if self.is_atari:
                state = self.train_preprocessor.reset(obs)
            else:
                state = obs
            self.n_step_buffer.clear()
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                if self.is_atari:
                    next_state = self.train_preprocessor.step(next_obs)
                else:
                    next_state = next_obs

                self.n_step_buffer.append((state, action, reward, next_state, done))
                if len(self.n_step_buffer) == self.n_step:
                    s, a, r, ns, d = self._get_n_step_info()
                    transition = (s, a, r, ns, d)
                    if self.use_per:
                        self.memory.add(transition, error=1.0)  # Initial error can be set to max priority or a large value
                    else:
                        self.memory.append(transition)

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

            # Flush remaining n_step_buffer at episode end
            # If buffer reached full n_step, the last full transition was already stored in the main loop;
            # skip it here and only store the remaining partial-step transitions.
            if len(self.n_step_buffer) == self.n_step:
                self.n_step_buffer.popleft()
            while len(self.n_step_buffer) > 0:
                s, a, r, ns, d = self._get_n_step_info()
                if self.use_per:
                    self.memory.add((s, a, r, ns, d), error=1.0)
                else:
                    self.memory.append((s, a, r, ns, d))
                self.n_step_buffer.popleft()

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
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
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
                attr = f"_saved_{ms}"
                if self.env_count >= ms and not getattr(self, attr, False):
                    setattr(self, attr, True)
                    mpath = os.path.join(self.save_dir, f"model_{ms}.pt")
                    torch.save(self.q_net.state_dict(), mpath)
                    print(f"[Milestone] Saved {ms} steps snapshot → {mpath}")

    def evaluate(self, num_seeds=5):
        rewards = []
        for seed in range(num_seeds):
            obs, _ = self.test_env.reset(seed=seed)
            if self.is_atari:
                state = self.test_preprocessor.reset(obs)
            else:
                state = obs
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
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

        # Check if replay buffer has enough samples for training
        buf_len = len(self.memory) if not self.use_per else len(self.memory.buffer)
        if buf_len < self.replay_start_size:
            return 
        
        # Decay function for epsilon-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        if self.use_per:
            self.memory.beta = min(1.0, self.memory.beta + 6e-8) # Anneal PER beta from initial value toward 1.0 over training
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

        # Calculate Loss and perform a gradient descent step
        td_errors = target_q - q_values

        if self.use_per:
            # PER loss: weight the TD errors by importance sampling weights and update priorities
            loss = (is_weights * F.smooth_l1_loss(q_values, target_q, reduction='none')).mean()
            self.memory.update_priorities(indices, td_errors.abs().detach().cpu().numpy())
        else:
            # Uniform replay loss: standard Huber loss
            loss = F.smooth_l1_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
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
    parser.add_argument("--per-alpha",  type=float, default=0.6)
    parser.add_argument("--per-beta",   type=float, default=0.4)
    parser.add_argument("--n-step",     type=int,   default=1)
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