"""
REINFORCE algorithm

Author: Pedro Mota
Date: 12 set 2022
"""

from cmath import log
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import time
import random

# For reproducibility
RANDOM_SEED = 42

tf.random.set_seed(RANDOM_SEED)
env = gym.make("MountainCar-v0")
np.random.seed(RANDOM_SEED)

print("Action Space: {}".format(env.action_space.n))
print("State space: {}".format(env.observation_space.shape))

# An episode a full game
train_episodes = 300
test_episodes = 100

GAMMA = 1

class Agent:
    def __init__(self):
        init = tf.keras.initializers.HeUniform()
        
        self.model = keras.Sequential()
        
        self.model.add(keras.layers.Dense(4, input_dim = 2, activation='relu', kernel_initializer=init))
        self.model.add(keras.layers.Dense(3, activation='softmax', kernel_initializer=init))
        
        self.model.compile(loss=tf.keras.losses.Huber(), 
                           optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001), 
                           metrics=['accuracy'])
        
        self.memory = []
    
    def train(self):
        discounted_returns = [self.memory[0][3]]
        for i, _, _, _, reward in enumerate(self.memory[1:]):
            discounted_returns.append(discounted_returns[i-1] + GAMMA ** i * reward)

        loss = 0
        for i, _, _, action_prob, reward in enumerate(self.memory):
            loss += -log(action_prob) * reward

        self.memory = []

    def get_action_dist(self, state):
        return self.model.predict(state.reshape((1, 2)))[0]

    def get_action(self, state):
        dist = self.get_action_dist(state) # Get action distribution for the state
        action = np.random.choice(env.action_space.n, p=dist) # Sample action from this distribution
        # dist[action] gives the prob of the chosen action
        action_prob = dist[action] 
        print(dist)
        return action, action_prob

    def observation(self, state, action, done, reward):        
        self.memory.append((state, action, reward))
        if done: 
           return

def main():
    agent = Agent()

    state = env.reset(seed=RANDOM_SEED)

    for ep in range(train_episodes):
        print(f"-------------- Episode {ep + 1} --------------")

        state = env.reset(seed=RANDOM_SEED)
        done = False

        while not done:        
            action = agent.get_action(state)

            new_state, reward, terminated, truncated, info = env.step(action)
            
            done = terminated | truncated
            
            agent.observation(state, action, new_state, done, reward)

            state = new_state
    
    env.close()

if __name__ == '__main__':
    main()
