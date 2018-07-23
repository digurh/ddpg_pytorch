import numpy as np
import random
import copy

import torch
import torch.optim as optim
import torch.nn.functional as F

from net import Actor, Critic
from experience_replay import ReplayBuffer
import ou_noise

class DDPGAgent:
    def __init__(self, state_size, action_size, seed, n_hidden_units=128, n_layers=3):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # actor network
        self.actor_local = Actor(state_size, action_size, seed)
        self.actor_target = Actor(state_size, action_size, seed)
        self.actor_opt = optim.Adam(self.actor_local.parameters, )
