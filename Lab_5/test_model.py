import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
from collections import deque
import argparse

gym.register_envs(ale_py)


class DQN(nn.Module):
    def __init__(self, num_actions, input_channels=None):
        super(DQN, self).__init__()
        self.is_atari = input_channels is not None
        if self.is_atari:
            self.network = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),             nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),             nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
                nn.Linear(512, num_actions),
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(4, 128),   nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, num_actions),
            )

    def forward(self, x):
        if self.is_atari:
            x = x / 255.0
        return self.network(x)

class AtariPreprocessor:
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

def evaluate(args):
    test_env = gym.make(args.env_name, render_mode="rgb_array")
    num_actions = test_env.action_space.n
    test_preprocessor = AtariPreprocessor()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_atari = "ALE" in args.env_name or "Pong" in args.env_name

    input_channels = 4 if is_atari else None
    model = DQN(num_actions=num_actions, input_channels=input_channels).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    if args.record_video:
        os.makedirs(args.output_dir, exist_ok=True)

    rewards = []
    for ep in range(args.episodes):
        obs, _ = test_env.reset(seed=args.seed + ep)

        if is_atari:
            state = test_preprocessor.reset(obs)
        else:
            state = obs

        done = False
        total_reward = 0
        frames = []

        while not done:
            if args.record_video:
                raw_frame = test_env.render()

                h, w = raw_frame.shape[:2]
                
                new_w = (w + 15) // 16 * 16
                new_h = (h + 15) // 16 * 16

                if new_w != w or new_h != h:
                    resized_frame = cv2.resize(raw_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    frames.append(resized_frame)
                else:
                    frames.append(raw_frame)
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            if is_atari:
                state = test_preprocessor.step(next_obs)
            else:
                state = next_obs

        rewards.append(int(total_reward))
        print(f"Environment steps: {args.env_steps}; seed: {args.seed + ep}; eval reward: {int(total_reward)}")

        if args.record_video and frames:
            env_tag = args.env_name.replace("/", "_").replace("-", "_")
            fname = f"{env_tag}_{args.env_steps}steps_seed{args.seed + ep}.mp4"
            out_path = os.path.join(args.output_dir, fname)
            with imageio.get_writer(out_path, fps=30) as video:
                for f in frames:
                    video.append_data(f)
            

    test_env.close()

    avg = np.mean(rewards)
    print(f"Average reward: {avg:.2f}")
    if not is_atari:
        score_pct = min(avg, 480) / 480 * 15
        print(f"Estimated Task 1 score: {score_pct:.2f}% / 15%")
    else:
        score_pct = min(avg, 19) / 40 * 20 + 21 / 40 * 20
        print(f"Estimated Task 2 score: {score_pct:.2f}% / 20%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,            help="Path to trained .pt model")
    parser.add_argument("--output-dir", type=str, default="./eval_videos",  help="Directory to save evaluation videos")
    parser.add_argument("--episodes",   type=int, default=20,               help="Number of evaluation episodes to run [Do NOT CHANGE]")
    parser.add_argument("--seed",       type=int, default=0,                help="Random seed for evaluation [Do NOT CHANGE]")
    parser.add_argument("--env_name",   type=str, required=True,            help="Gym env id, e.g. CartPole-v1 or ALE/Pong-v5")
    parser.add_argument("--env-steps",  type=int, default=0,                help="Training step count for this checkpoint (used in output and video filenames)")
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()
    evaluate(args)
