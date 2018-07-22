import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init


def initialize_params(state_size, action_size, n_hidden_units=128, n_layers=3, initialize=None):
    if initialize is None: initialize = init.xavier_uniform_()
    layers = [n_hidden_units] * ((n_layers - 1) * 2)
    layers.insert(0, state_size)
    layers.append(action_size)
    layers = [(layers[2*i], layers[2*i+1]) for i in range(len(layers)//2)]

    weights, bias = {}, {}

    for i, (a, b) in enumerate(layers):
        weights[str(i)] = initialize(nn.Parameter(torch.FloatTensor(a, b)))
        bias[str(i)] = nn.Parameter(torch.zeros(b))

    return weights, bias


class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, n_hidden_units=128, n_layers=3):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.weights, self.bias = initialize_params(state_size, action_size, n_hidden_units, n_layers)
        self.parameters = nn.ParameterList([self.weights[w] for w in self.weights.keys()] +
                                           [self.bias[b] for b in self.bias.keys()])

    def forward(self, X):
        X = torch.mm(self.__forward(x, 1), self.weights['3'] + self.bias['3'])
        return F.tanh(X)

    def __forward(self, X, layer):
        if layer == len(self.weights.keys()) - 1: return X
        else:
            X = self.__forward(F.relu(torch.mm(X, self.weights[str(layer)]) + self.bias[str(layer)]), layer+1)

class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, n_hidden_units=128, n_layers=3):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.weights, self.bias = initialize_params()
