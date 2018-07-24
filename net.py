import numpy as np
from utils import linear_params_init as initial, forward_compute as forc

import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, n_hidden_units=128, n_layers=3):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.weights, self.bias, self.parameters = initial(state_size, action_size, n_hidden_units, n_layers)

    def forward(self, X):
        return F.tanh(forc(X, len(self.weights.keys())-1, self.weights, self.bias))

class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, n_hidden_units=128, n_layers=3):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.weights, self.bias, self.parameters = self.params_init(state_size, action_size, n_hidden_units)

    def params_init(self, state_size, action_size, n_hidden_units):
        weights = {
            '1': init.xavier_uniform_(nn.Parameter(torch.FloatTensor(state_size, n_hidden_units))),
            '2': init.xavier_uniform_(nn.Parameter(torch.FloatTensor(n_hidden_units+action_size, n_hidden_units))),
            '3': init.xavier_uniform_(nn.Parameter(torch.FloatTensor(n_hidden_units, n_hidden_units//2))),
            '4': init.xavier_uniform_(nn.Parameter(torch.FloatTensor(n_hidden_units//2, 1)))
        }
        bias = {
            '1': nn.Parameter(torch.zeros(n_hidden_units)),
            '2': nn.Parameter(torch.zeros(n_hidden_units)),
            '3': nn.Parameter(torch.zeros(n_hidden_units//2)),
            '4': nn.Parameter(torch.zeros(1))
        }
        parameters = nn.ParameterList([weights[w] for w in weights.keys()] +
                                      [bias[b] for b in bias.keys()])
        return weights, bias, parameters

    def forward(self, state, action):
        Xs = F.leaky_relu(torch.mm(state, self.weights['1']) + self.bias['1'])
        X = torch.cat((Xs, action), dim=1)
        X = F.leaky_relu(torch.mm(X, self.weights['2']) + self.bias['2'])
        X = F.leaky_relu(torch.mm(X, self.weights['3']) + self.bias['3'])
        return torch.mm(X, self.weights['4']) + self.bias['4']
