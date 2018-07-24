import gym
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from agent import DDPGAgent


env = gym.make('BipedalWalker-v2')
env.seed(0)
agent = DDPGAgent(env.observation_space.shape[0], env.action_space.shape[0], 0)
