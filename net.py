import numpy as np

import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions import Categorical as Cat

def initialize_params(self, state_size, action_size, n_hidden_units=128):
    self.seed = torch.manual_seed()
    weights = {
        '1': init.xavier_uniform_(nn.Parameter(torch.FloatTensor(state_size, n_hidden_units))),
        '2': init.xavier_uniform_(nn.Parameter(torch.FloatTensor(n_hidden_units, n_hidden_units))),
        '3': init.xavier_uniform_(nn.Parameter(torch.FloatTensor(n_hidden_units, action_size)))
    }
    bias = {
        '1': nn.Parameter(torch.zeros(n_hidden_units)),
        '2': nn.Parameter(torch.zeros(n_hidden_units)),
        '3': nn.Parameter(torch.zeros(action_size))
    }
    return weights, bias


class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, n_hidden_units=128, n_layers=3):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.weights, self.bias = initialize_params(state_size, action_size, n_hidden_units)
        self.parameters = nn.ParameterList([self.weights[w] for w in self.weights.keys()] +
                                           [self.bias[b] for b in self.bias.keys()])

    def forward(self, state):
        X = F.relu(torch.mm(state, self.weights['1']) + self.bias['1'])
        X = F.relu(torch.mm(X, self.weights['2'] + self.bias['2']))
        X = torch.mm(X, self.weights['3'] + self.bias['3'])
        return F.tanh(X)

class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, n_hidden_units=128, n_layers=3):
        super().__init__()
        self.seed = torch.manual_seed(seed)
