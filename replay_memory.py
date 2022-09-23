import numpy as np

class ReplayMemory:
    def __init__(self, max_size):
        self.c = 0 # Counter
        self.max_size = max_size
        self.data = np.zeros((self.max_size, 4), dtype=np.float32) # (state, action, reward, next state)

    def store(self, state, action, reward, next_state):
        print(state, action, reward, next_state)
        idx = self.c % self.max_size
        self.data[idx] = (state, action, reward, next_state)
        self.c += 1
    
    def sample(self, batch_size):
        # Note that, if full, we can sample from the whole buffer.
        # But, if yet to be full, we need to sample only from index 0 to c, exclusive.
        idxs = np.random.choice(min(self.c, self.max_size), batch_size, replace=False)
        return self.data[idxs]

    def view(self):
        print(self.data)

    def __len__(self):
        return min(self.c, self.max_size)
    
if __name__ == "__main__":
    m = ReplayMemory(max_size=10)

    for i in range(20):
        s = np.random.choice(20)
        a = np.random.choice(4)
        r = np.random.choice(1000)
        ns = np.random.choice(20)

        m.store(s, a, r, ns)
    
    m.view()

    sample = m.sample(batch_size=10)
    print(sample)

    print(sample[:,0])