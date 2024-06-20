""" 
    DQN agent implementation, without target network.

    Author: Pedro Mota
    Date: 19th Jun 2024
"""

import gymnasium as gym

from collections import namedtuple, deque
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class DeepQNetwork(nn.Module):
    def __init__(self, n_observations, n_actions, lr):
        super(DeepQNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

        self.optim = optim.Adam(self.parameters(), lr=lr)
        
        self.loss = nn.MSELoss()

        self.to(device)

    def forward(self, x): # Called with either one element or a batch
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class ReplayMemory(object):
    Transition = namedtuple('Transition', ('observation', 'action', 'reward', 'next_observation'))
    
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def store(self, *args):
        self.memory.append(ReplayMemory.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Agent():
    def __init__(
        self, 
        gamma, 
        n_observations, n_actions,
        lr, batch_size,
        epsilon, epsilon_decay, epsilon_min, 
        max_mem_size
    ):
        self.gamma = gamma 
        self.lr = lr 
        self.n_observations = n_observations
        self.batch_size = batch_size 
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q = DeepQNetwork(n_observations, n_actions, lr)
        self.replay_memory = ReplayMemory(capacity=max_mem_size)
    
    def get_action(self, observation):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.n_actions)
        with torch.no_grad():
            state = torch.tensor(np.array([observation])).to(device)
            actions = self.q(state)
            action = torch.argmax(actions).item()
            return action
    
    def store(self, observation, action, reward, new_observation):
        self.replay_memory.store(observation, action, reward, new_observation)
    
    def learn(self):
        if len(self.replay_memory) < self.batch_size: return 
            
        batch = self.replay_memory.sample(self.batch_size)
        observation_batch = torch.Tensor(np.array([transition.observation for transition in batch])).to(device)
        action_batch = [transition.action for transition in batch]
        reward_batch = torch.Tensor(np.array([transition.reward for transition in batch])).to(device)
        next_observation_batch = torch.Tensor(np.array([transition.next_observation for transition in batch])).to(device)

        q_obs = self.q(observation_batch).to(device)
        q_obs = q_obs[torch.arange(self.batch_size), action_batch]
        
        q_next_obs = self.q(next_observation_batch)
        q_target = reward_batch + self.gamma * torch.max(q_next_obs, dim=1)[0] # T.max returns the value as also the index

        loss = self.q.loss(q_obs, q_target).to(device)
        
        self.q.optim.zero_grad()
        loss.backward()        
        self.q.optim.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    
    agent = Agent(
        gamma          = 0.99, 
        n_observations = env.observation_space.shape[0], 
        n_actions      = env.action_space.n,
        lr             = 0.01, 
        batch_size     = 64,
        epsilon        = 1.0, 
        epsilon_decay  = 0.995, 
        epsilon_min    = 0.01, 
        max_mem_size   = 100000
    )

    EPISODES = 1000

    ep_rewards = np.zeros(shape=EPISODES)

    for ep in range(EPISODES):
        print(f"Episode {ep+1} starting...", end=" ")

        observation, _ = env.reset() # just want the id of the state
        done = False        
        while not done:
            action = agent.get_action(observation)
            new_observation, reward, terminated, truncated, info = env.step(action)
            ep_rewards[ep] += reward
            done = terminated or truncated
            agent.store(observation, action, reward, new_observation)
            agent.learn()
            observation = new_observation
 
        print(f"DONE.")
    
    plt.plot(ep_rewards)
    plt.title('Agent\'s return through the episodes')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.show()

    # Visualize trained agent
    env = gym.make('CartPole-v1', render_mode="human")
    obs, _ = env.reset(seed=42)
    done = False
    import time
    while not done:
        action = agent.get_action(obs)
        print(f"In state {obs},", end=" ")
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"took action {action}, to get to state {obs}, and got a reward of {reward};")
        time.sleep(0.1)

    env.close()