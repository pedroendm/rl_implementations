import numpy as np
import tensorflow as tf
from tensorflow import keras
import gym
import matplotlib.pyplot as plt
from replay_memory import ReplayMemory

class Agent:
    LEARNING_RATE           = 0.01
    REPLAY_MEMORY_MAX_SIZE  = 10000
    BATCH_SIZE              = 20
    ALPHA                   = .85  # Learning rate
    GAMMA                   = .9   # Discount factor
    EPSILON                 = 1    # Epsilon-greedy policy
    EPSILON_DECAY           = 0.01  # Value to be subtracted from epsilon in each reduction
    
    def __init__(self, input_dim, output_dim):
        self.qn = keras.Sequential([
            keras.layers.Dense(input_dim, activation='relu'),
            keras.layers.Dense(4, activation='relu'),
            keras.layers.Dense(output_dim, activation=None),
        ])
        
        self.qn.compile(
            optimizer = keras.optimizers.Adam(learning_rate = Agent.LEARNING_RATE),
            loss = 'mean_squared_error'
        )

        self.rm = ReplayMemory(max_size=Agent.REPLAY_MEMORY_MAX_SIZE)
        self.n_actions = output_dim

    def observation(self, state, action, reward, next_state, done):
        self.rm.store(state, action, reward, next_state)

    def get_action(self, state):
        if np.random.rand() <= Agent.EPSILON: # Take Random Action with epsilon probability
            return np.random.choice(self.n_actions)
        else: # Otherwise, Take Best Estimated Action
            qvs = self.qn.predict(state)   
            action = np.argmax(qvs)
            return action

    def learn(self):
        pass
        if len(self.rm) < Agent.BATCH_SIZE: return

        sample = self.rm.sample(Agent.BATCH_SIZE)
        
        states     = sample[:,0]
        actions    = sample[:,1]
        rewards    = sample[:,2]
        next_state = sample[:,3]
        
        qv  = self.qn.predict(states)

        Agent.EPSILON = Agent.EPSILON - Agent.EPSILON_DECAY

    def save_model(self, filename):
        self.qn.save(filename)
    def load_model(self, filename):
        self.qn = keras.load_model(filename)


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")

    agent = Agent(input_dim= , output_dim=4)

    episodes = 1200

    ep_rewards = np.zeros(shape=episodes)

    for ep in range(episodes):
        state = env.reset()
        print(f"Episode {ep+1} starting...", end=" ")

        done = False        

        while not done:
            action = agent.get_action(state)

            next_state, reward, done, info = env.step(action)
            
            ep_rewards[ep] = ep_rewards[ep] + reward

            agent.observation(state, action, reward, next_state, done)

            state = next_state
        
        agent.learn()
 
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
