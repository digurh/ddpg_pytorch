import numpy as np

import torch
from torch import nn
import torch.nn.init as init


def linear_params_init(state_size, action_size, n_hidden_units=128, n_layers=3, initialize=None):
    if initialize is None: initialize = init.xavier_uniform_()
    layers = [n_hidden_units] * ((n_layers - 1) * 2)
    layers.insert(0, state_size)
    layers.append(action_size)
    layers = [(layers[2*i], layers[2*i+1]) for i in range(len(layers)//2)]

    weights, bias = {}, {}

    for i, (a, b) in enumerate(layers):
        weights[str(i)] = initialize(nn.Parameter(torch.FloatTensor(a, b)))
        bias[str(i)] = nn.Parameter(torch.zeros(b))

    parameters = nn.ParameterList([weights[w] for w in self.weights.keys()] +
                                  [bias[b] for b in self.bias.keys()])

    return weights, bias, parameters

def forward_compute(X, layer, weights, bias):
    if layer == 1:
        return F.relu(torch.mm(X, weights['1']) + bias['1'])
    else:
        X = forward_compute(F.relu(torch.mm(X, weights[str(layer)]) + bias[str(layer)]), layer-1)
        return torch.mm(X, weights[str(layer)] + bias[str(layer)])
