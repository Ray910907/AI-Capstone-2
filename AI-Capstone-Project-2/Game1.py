import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gymnasium.wrappers import RecordVideo
import os

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN Network
class DQN_net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# DQN Agent
class DQNAgent:
    def __init__(self, env, gamma=0.99, lr=1e-3, batch_size=64, eps_start=1.0, eps_end=0.02, eps_decay=5000, target_update=10, memory_capacity=10000):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.steps_done = 0

        self.n_actions = env.action_space.n
        self.state_dim = env.observation_space.shape[0]

        self.policy_net = DQN_net(self.state_dim, self.n_actions).to(device)
        self.target_net = DQN_net(self.state_dim, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayMemory(memory_capacity)

    def select_action(self, state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() < eps_threshold:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda d: not d, batch.done)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s, d in zip(batch.next_state, batch.done) if not d])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            '''
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
            '''
            next_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions).squeeze()

        expected_state_action_values = reward_batch + (self.gamma * next_state_values)

        loss = F.mse_loss(state_action_values.squeeze(), expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes=500):
        episode_rewards = []
        with tqdm(total=num_episodes, desc="Training", ncols=120) as pbar:
            for episode in range(num_episodes):
                state, _ = self.env.reset()
                state = torch.tensor([state], device=device, dtype=torch.float32)
                total_reward = 0

                for t in count():
                    action = self.select_action(state)
                    next_state, reward, terminated, truncated, _ = self.env.step(action.item())
                    done = terminated or truncated

                    # Reward shaping
                    x, _, theta, _ = next_state
                    x_norm = 1 - abs(x) / self.env.x_threshold
                    theta_norm = 1 - abs(theta) / self.env.theta_threshold_radians
                    reward = 0.5 * x_norm + 0.5 * theta_norm

                    total_reward += reward
                    
                    reward = torch.tensor([reward], device=device, dtype=torch.float32)
                    next_state_tensor = torch.tensor([next_state], device=device, dtype=torch.float32)

                    self.memory.push(state, action, reward, next_state_tensor, done)
                    state = next_state_tensor

                    self.optimize_model()

                    if done or t >= 199:
                        episode_rewards.append(total_reward)
                        pbar.set_postfix({'Episode': episode, 'Reward': total_reward})
                        pbar.update(1)
                        break

                if episode % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
        return episode_rewards

    def test(self, video_dir="./videos", max_steps=200):
        os.makedirs(video_dir, exist_ok=True)
        env = RecordVideo(
            gym.make("CartPole-v0", render_mode="rgb_array"),
            video_folder=video_dir,
            name_prefix="cartpole_test"
        )
        state, _ = env.reset()
        state = torch.tensor([state], device=device, dtype=torch.float32)
        total_reward = 0

        for _ in range(max_steps):
            with torch.no_grad():
                action = self.policy_net(state).max(1)[1].view(1, 1)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward
            state = torch.tensor([next_state], device=device, dtype=torch.float32)
            if done:
                break

        env.close()
        print(f"Test episode reward: {total_reward}")
        print(f"Video saved in: {video_dir}")


def main():
    env = gym.make("CartPole-v0")
    env = env.unwrapped

    agent = DQNAgent(env)

    rewards = agent.train(num_episodes=500)

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN on CartPole")
    plt.savefig("./videos/double_shaping.png")

    agent.test()


if __name__ == "__main__":
    main()
