import random
from collections import deque

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch_geometric.utils import to_networkx


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


def plot_performance(tour_lenghts: np.array):
    """
    The plot_performance function takes a numpy array of tour lengths and plots them.
        The x-axis is the number of episodes, while the y-axis is the length of each tour.

    :param tour_lenghts:np.array: Pass the array of tour lenghts to the function
    :return: A plot of the tour lengths for each episode
    """
    plt.plot(tour_lenghts)
    plt.xlabel("episodes")
    plt.ylabel("tour length")
    plt.show()

def draw_probs_graph(pyg_graph, probabilities, ax):
    """
    The draw_probs_graph function takes in a pyg_graph, probabilities, and an axis.
    It then draws the nodes of the graph according to their color and position attributes.
    The edges are drawn with weights corresponding to the probability values passed in.

    :param pyg_graph: Draw the graph
    :param probabilities: Set the edge weights
    :param ax: Plot the graph in a specific axis
    :return: A graph with the probabilities of each edge
    """
    graph_size = pyg_graph.x.shape[0]

    # draw nodes according to color and position attribute
    G_nx = to_networkx(pyg_graph, node_attrs=["x", "coordinates"])
    pos = nx.get_node_attributes(G_nx, "coordinates")
    nx.draw_networkx_nodes(
        G_nx, pos, ax=ax, node_size=100
    )
    labels_pos = {k: (v + np.array([0, 0.03])) for k, v in pos.items()}
    nx.draw_networkx_labels(
        G_nx, labels_pos, ax=ax
    )

    # set edges weights
    edge_weights = {(u,v) : {"probability" : float(probabilities[u,v])} for u in range(graph_size) for v in range(graph_size)}
    nx.set_edge_attributes(G_nx, edge_weights)
    probabilities = nx.get_edge_attributes(G_nx,'probability').values()
    options = {
        "edge_color": probabilities,
        "width": 1,
        "edge_cmap": plt.cm.Blues,
        "arrows": True,
    }
    nx.draw_networkx_edges(G_nx, pos, **options)


def sample_draw_probs_graph(batch, preds):
    """
    The sample_draw_probs_graph function takes a batch of data and the predictions for that batch,
    and plots the probabilities of each class for one randomly selected sample from that batch.
    The function returns a tuple containing the figure and axis objects used to plot this graph.

    :param batch: Get the image and label from the batch
    :param preds: Plot the predictions on top of the actual values
    :return: A figure and an axis object
    """
    fig, ax = plt.subplots()
    selected = random.randrange(len(batch))
    draw_probs_graph(batch[selected], preds[selected], ax)
    return fig, ax