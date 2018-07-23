import numpy as np
import random
from collections import deque, namedtuple

class ExperienceReplay:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def add(self, state, action, reward, next_state, done): pass
    def sample(self): pass
    def __len__(self): pass


class ReplayBuffer(ExperienceReplay):
    def __init__(self, seed, batch_size=64, buffer_max=int(1e6)):
        super().__init__()
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_max)
        self.seed = random.seed(seed)
        self.device = device

        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])

    def add(self, state, action, rewar, next_state, done):
        self.buffer.append(self.experience(state, action, reward, next_state, done))

    def sample(self):
        experience = random.sample(self.buffer, k=self.batch_size)

        states = torch.from_numpy(np.vstack([ex.state for ex in experience if ex is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([ex.action for ex in experience if ex is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([ex.reward for ex in experience if ex is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([ex.next_state for ex in experience if ex is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([ex.done for ex in experience if ex is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def len(self):
        return len(self.buffer)
