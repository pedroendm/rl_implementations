""" 
    Q-Learning agent implementation.

    Author: Pedro Mota
    Date: 13th Set 2022
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

ALPHA         = .85  # Learning rate
GAMMA         = .9   # Discount factor
EPSILON       = 1    # Epsilon-greedy policy
EPSILON_STEPS = 100  # Number of actions taken needed to perform a epsilon reduction
EPSILON_DECAY = 0.1  # Value to be subtracted from epsilon in each reduction

class Agent:
    def __init__(self, env):
        self.qtable = np.zeros(shape=(env.observation_space.n, env.action_space.n)) # Rows <-> State ; Cols <-> Action
        self.n_actions = env.action_space.n

    def get_action(self, state):
        if np.random.rand() <= EPSILON: # Take Random Action with epsilon probability
            return np.random.choice(self.n_actions)
        else: # Otherwise, Take Best Estimated Action
            return np.argmax(self.qtable[state])        
        
    def observation(self, state, action, reward, new_state, done):
        # Update qtable
        self.qtable[state][action] += ALPHA * (reward + GAMMA * np.max(self.qtable[new_state]) - self.qtable[state][action])

if __name__ == "__main__":
    env = gym.make("Taxi-v3")

    agent = Agent(env)

    episodes = 1000

    ep_rewards = np.zeros(shape=episodes)

    for ep in range(episodes):
        state = env.reset() # Just want the id of the state

        print(f"Episode {ep+1} starting...", end=" ")

        done = False        

        while not done:
            action = agent.get_action(state)

            next_state, reward, done, info = env.step(action)

            ep_rewards[ep] = ep_rewards[ep] + reward

            #done = terminated | truncated
            
            agent.observation(state, action, reward, next_state, done)

            state = next_state
 
        # Decay epsilon in order to take less random actions as time goes
        if ep % 100 == 0: EPSILON -= EPSILON_DECAY      

        print(f"DONE.")
    
    plt.plot(ep_rewards)
    plt.title('Agent\'s return through the episodes')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.show()

    # Visualize trained agent
    state = env.reset()
    done = False

    while not done:
        env.render('human')
        action = agent.get_action(state)

        print(f"In state {state},", end=" ")
        state, reward, done, info = env.step(action)
        print(f"took action {action}, to get to state {state}, and got a reward of {reward};")

    env.close()