import numpy as np
from utils import linear_params_init as init, forward_compute as forc

import torch
from torch import nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, n_hidden_units=128, n_layers=3):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.weights, self.bias, self.parameters = init(state_size, action_size, n_hidden_units, n_layers)

    def forward(self, X):
        return F.tanh(forc(X, str(len(self.weights.keys()))-1))

class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, n_hidden_units=128, n_layers=3):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.weights, self.bias = init()
