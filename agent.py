import numpy as np
import random
import copy

import torch
import torch.optim as optim
import torch.nn.functional as F

from net import Actor, Critic
from experience_replay import ReplayBuffer
from ou_noise import OUNoise


GAMMA = 0.99
TAU = 1e-3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self, state_size, action_size, seed, n_hidden_units=128, n_layers=3):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # actor
        self.actor = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4)

        # critic
        self.critic = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-4, weight_decay=0.0001)

        # will add noise
        self.noise = OUNoise(action_size, seed)

        # experience replay
        self.replay = ReplayBuffer(seed)

    def act(self, state, noise=True):
        '''
            Returns actions taken.
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        if noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def step(self, state, action, reward, next_state, done):
        '''
            Save experiences into replay and sample if replay contains enough experiences
        '''
        self.replay.add(state, action, reward, next_state, done)

        if self.replay.len() > self.replay.batch_size:
            experiences = self.replay.sample()
            self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        '''
            Update policy and value parameters using given batch of experience tuples.
            Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
            where:
                actor_target(state) -> action
                critic_target(state, action) -> Q-value
            Params: experiences (Tuple[torch.Tensor]): tuple of (s, a, r, n_s, done) tuples
                    gamma (float): discount factor
        '''
        states, actions, rewards, next_states, dones = experiences
        # update critic:
        #   get predicted next state actions and Qvalues from targets
        next_actions = self.actor_target(next_states)
        next_Q_targets = self.critic_target(next_states, next_actions)
        #   get current state Qvalues
        Q_targets = rewards + (GAMMA * next_Q_targets * (1 - dones))
        #   compute citic loss
        Q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        #   minimize loss
        self.critic_opt.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_opt.step()

        # update actor:
        #   compute actor loss
        action_predictions = self.actor(states)
        actor_loss = -self.critic(states, action_predictions).mean()
        #   minimize actor loss
        self.actor_opt.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_opt.step()

        # update target networks
        self.soft_update(self.critic, self.critic_target, TAU)
        self.soft_update(self.actor, self.actor_target, TAU)

    def soft_update(self, local, target, tau):
        '''
            Soft update model parameters.
            θ_target = τ*θ_local + (1 - τ)*θ_target
            Params: local: PyTorch model (weights will be copied from)
                    target: PyTorch model (weights will be copied to)
                    tau (float): interpolation parameter
        '''
        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
