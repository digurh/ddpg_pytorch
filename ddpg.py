import gym
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from agent import DDPGAgent


env = gym.make('BipedalWalker-v2')
env.seed(0)
agent = DDPGAgent(env.observation_space.shape[0], env.action_space.shape[0], 0)

n_episodes = 2000
t_max = 1000

def train(n_episodes=2000, t_max=1000):
    score_deque = deque(maxlen=100)
    scores = []
    for ep in range(1, n_episodes+1):
        state = env.reset()
        agent.reset()
        score = 0
        for step in range(t_max):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action[0])
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break
        score_deque.append(score)
        scores.append(score)

        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(ep, np.mean(score_deque), score))
        if ep % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(ep, np.mean(score_deque)))
            # torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            # torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
    return scores

scores = train()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
