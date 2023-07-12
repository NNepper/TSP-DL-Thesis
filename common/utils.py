from collections import deque

import torch

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


def compute_optimality_gap(batch_graph, batch_pred_tour):
    length = torch.zeros(batch_graph.num_graphs)
    for i in range(batch_graph.num_graphs):
        nx_graph = to_networkx(batch_graph[i], edge_attrs=["edge_attr"])
        for j in range(batch_pred_tour.shape[1]):
            u = int(batch_pred_tour[i,j])
            v = int(batch_pred_tour[i, (j+1) % batch_pred_tour.shape[1]])
            length += nx_graph[u][v]["edge_attr"]
    return length
