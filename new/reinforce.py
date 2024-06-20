""" 
    REINFORCE agent implementation.

    Author: Pedro Mota
    Date: 20th Jun 2024
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

class PolicyNetwork(nn.Module):
    def __init__(self, n_observations, n_actions, lr):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

        self.optim = optim.Adam(self.parameters(), lr=lr)
        
        #self.loss = nn.MSELoss()

        self.to(device)

    def forward(self, x): # called with either one element or a batch
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=-1) # why dim=-1?
        return x


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
        max_mem_size
    ):
        self.gamma = gamma 
        self.lr = lr 
        self.n_observations = n_observations
        self.batch_size = batch_size 
        self.n_actions = n_actions

        self.policy = PolicyNetwork(n_observations, n_actions, lr)
        self.replay_memory = ReplayMemory(capacity=max_mem_size)
    
    def get_action(self, observation):
        observation = torch.tensor(np.array([observation])).to(device)
        action_probs = self.policy(observation)
        return np.random.choice(env.action_space.n, p=action_probs)

    def store(self, observation, action, reward, new_observation):
        self.replay_memory.store(observation, action, reward, new_observation)
    
    def learn(self):
        if len(self.replay_memory) < self.batch_size: return 

        return            

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    
    agent = Agent(
        gamma          = 0.99, 
        n_observations = env.observation_space.shape[0], 
        n_actions      = env.action_space.n,
        lr             = 0.01, 
        batch_size     = 64,
        max_mem_size   = 100000
    )

    EPISODES = 100

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