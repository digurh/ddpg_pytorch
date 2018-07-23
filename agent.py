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

        # actor
        self.actor = Actor(state_size, action_size, seed)
        self.actor_target = Actor(state_size, action_size, seed)
        self.actor_opt = optim.Adam(self.actor.parameters, lr=1e-4)

        # critic
        self.critic = Critic(state_size, action_size, seed)
        self.critic_target = Critic(state_size, action_size, seed)
        self.critic_opt = optim.Adam(self.critic.parameters, lr=3e-4)

        # will add noise
        self.noise = OUNoise(action_size, seed)

        # experience replay
        self.replay = ReplayBuffer(batch_size, seed)

    
