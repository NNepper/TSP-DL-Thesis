from collections import deque

import torch


def discounted_rewards(rewards, gamma):
    """
    The discounted_rewards function takes in a list of rewards and a discount factor gamma.
    It returns the discounted reward for each step, where the first element is the reward at that step,
    the second element is gamma times that reward plus the next one, etc.  This function can be used to calculate
    the discounted return for an entire episode.

    :param rewards: Calculate the discounted rewards
    :param gamma: Calculate the discounted rewards
    :return: A tensor of the same shape as rewards, but with each element containing the discounted reward for that step
    """
    G = torch.zeros_like(rewards)
    for b in range(rewards.shape[0]):
        returns = deque()
        R = 0
        for r in torch.flip(rewards, dims=(0, 1))[b]:
            R = r + gamma * R
            returns.appendleft(R)
        G[b, :] = torch.tensor(returns)
    return G


