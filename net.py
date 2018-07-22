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
        self.weights, self.bias, self.parameters = self.params_init()


    def params_init(self, state_size, action_size, n_hidden_units):
        w_init = init.xavier_uniform_(nn.Parameter(torch.FloatTensor()))
        weights = {
            '1': w_init(state_size, n_hidden_units),
            '2': w_init(n_hidden_units+action_size, n_hidden_units),
            '3': w_init(n_hidden_units, n_hidden_units//2),
            '4': w_init(n_hidden_units//2, 1)
        }
        b_init = nn.Parameter(torch.zeros())
        bias = {
            '1': b_init(n_hidden_units),
            '2': b_init(n_hidden_units),
            '3': b_init(n_hidden_units//2),
            '4': b_init(1)
        }
        parameters = nn.ParameterList([weights[w] for w in self.weights.keys()] +
                                      [bias[b] for b in self.bias.keys()])
        return weights, bias, parameters

    def forward(self, state, action):
        Xs = F.leaky_relu(torch.mm(state, self.weights['1']) + self.bias['1'])
        X = torch.cat((Xs, action), dim=1)
        X = F.leaky_relu(torch.mm(X, self.weights['2']) + self.bias['2'])
        X = F.leaky_relu(torch.mm(X, self.weights['3']) + self.bias['3'])
        return torch.mm(X, self.weights['4']) + self.bias['4']
