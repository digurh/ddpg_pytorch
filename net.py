import numpy as np
from utils import initialize_params

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init


class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, n_hidden_units=128, n_layers=3):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.weights, self.bias = initialize_params(state_size, action_size, n_hidden_units, n_layers)
        self.parameters = nn.ParameterList([self.weights[w] for w in self.weights.keys()] +
                                           [self.bias[b] for b in self.bias.keys()])

    def forward(self, X):
        last = str(len(self.weights.keys()))
        X = torch.mm(self.__forward(X, 1), self.weights[last] + self.bias[last])
        return F.tanh(X)

    def __forward(self, X, layer):
        if layer == len(self.weights.keys()) - 1:
            return X
        else:
            return self.__forward(F.relu(torch.mm(X, self.weights[str(layer)]) + self.bias[str(layer)]), layer+1)

class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, n_hidden_units=128, n_layers=3):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.weights, self.bias = initialize_params()
