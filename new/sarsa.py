""" 
    SARSA agent implementation.

    Author: Pedro Mota
    Date: 19th Jun 2024
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

EPISODES      = 1000 # 
ALPHA         = .1  # Learning rate
GAMMA         = 1    # Discount factor
EPSILON       = 1    # Epsilon-greedy policy
EPSILON_STEPS = 250   # Number of actions taken needed to perform a epsilon reduction
EPSILON_DECAY = 0.1  # Value to be subtracted from epsilon in each reduction

class Agent:
    def __init__(self, env):
        self.qtable = np.zeros(shape=(env.observation_space.n, env.action_space.n))
        self.n_actions = env.action_space.n

    def get_action(self, state):
        if np.random.rand() <= EPSILON: # Take Random Action with epsilon probability
            return np.random.choice(self.n_actions)
        # Otherwise, Take Best Estimated Action
        return np.argmax(self.qtable[state])        
        
    def observation(self, state, action, reward, new_state, next_action, done):
        # Update qtable
        self.qtable[state][action] += ALPHA * (reward + GAMMA * self.qtable[new_state][next_action] - self.qtable[state][action])

if __name__ == "__main__":
    env = gym.make("Taxi-v3")

    agent = Agent(env)

    ep_rewards = np.zeros(shape=EPISODES)

    for ep in range(EPISODES):
        print(f"Episode {ep+1} starting...", end=" ")

        obs, _ = env.reset(seed=42) # Just want the id of the state

        done = False

        action = agent.get_action(obs)

        while not done:
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            next_action = agent.get_action(next_obs)
            
            ep_rewards[ep] = ep_rewards[ep] + reward

            done = terminated or truncated
            
            agent.observation(obs, action, reward, next_obs, next_action, done)

            obs, action = next_obs, next_action
 
        # Decay epsilon in order to take less random actions as time goes
        if ep % 100 == 0: EPSILON -= EPSILON_DECAY      

        print(f"DONE.")
    
    plt.plot(ep_rewards)
    plt.title('Agent\'s return through the episodes')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.show()

    # Visualize trained agent
    env = gym.make("Taxi-v3", render_mode="human")
    obs, _ = env.reset(seed=42)
    done = False
    EPSILON = 0
    while not done:
        action = agent.get_action(obs)

        print(f"In state {obs},", end=" ")
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"took action {action}, to get to state {obs}, and got a reward of {reward};")

    env.close()
