from collections import deque

import numpy as np
import torch
from matplotlib import pyplot as plt


def discounted_rewards(rewards, gamma):
    # Calculate discounted rewards, going backwards from end
    G = torch.zeros_like(rewards)
    for b in range(rewards.shape[0]):
        returns = deque()
        R = 0
        for r in torch.flip(rewards, dims=(0,1))[b]:
            R = r + gamma * R
            returns.appendleft(R)
        G[b,:] = torch.tensor(returns)
    return G


def plot_performance(tour_lenghts : np.array):
    plt.plot(tour_lenghts)
    plt.xlabel("episodes")
    plt.ylabel("tour length")
    plt.show()
