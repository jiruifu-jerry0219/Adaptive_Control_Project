# Import all necessary packages
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import seaborn

env = gym.make('RoboschoolInvertedDoublePendulum-v1')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

from dqn_agent import Agent

agent = Agent(state_size=8, action_size=4, seed=0)

# watch an untrained agent
state = env.reset()
for j in range(200):
    action = agent.act(state)
    env.render()
    state, reward, done, _ = env.step(action)
    if done:
        break

env.close()