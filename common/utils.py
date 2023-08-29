from collections import deque

import torch
from scipy.spatial.distance import euclidean

from data.dataset import TSPDataset
from torch.utils.data import DataLoader


def load_dataset(filename, neighbors, batch_size=1, workers=4):    
    dataset = TSPDataset(
        filename=filename,
        batch_size=batch_size,
        neighbors=neighbors,
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=workers,
    )
    return dataset, dataloader


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


def compute_optimality_gap(graph, prediction, target):
    true_length = pred_length = 0
    for i in range(len(graph)):
        # Prediction
        pred_length += euclidean(
            graph[int(prediction[i])],
            graph[int(prediction[(i+1)%len(graph)])])

        # Target
        true_length += euclidean(
            graph[int(target[i])],
            graph[int(target[(i+1)%len(graph)])])
    return 100 * (pred_length - true_length) / true_length
