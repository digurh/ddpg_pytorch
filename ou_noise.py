import numpy as np
import random
import copy


class OUNoise:
    def __init__(self, size, seed, mu=0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        '''
            Reset the internal state (= noise) to mean (mu)
        '''
        self.state = copy.copy(self.mu)

    def sample(self):
        '''
            Adds random noise to action: effect is to alter exploration and exploitation amounts
        '''
        s = self.state
        ds = self.theta * (self.mu - s) + self.sigma * np.array([random.random() for i in range(len(s))])
        self.state = s + ds
        return self.state
