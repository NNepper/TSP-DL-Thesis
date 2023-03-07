from collections import deque

import numpy as np
import torch
import random
from matplotlib import pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from itertools import product

def discounted_rewards(rewards, gamma):
    # Calculate discounted rewards, going backwards from end
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
    plt.plot(tour_lenghts)
    plt.xlabel("episodes")
    plt.ylabel("tour length")
    plt.show()

def draw_probs_graph(pyg_graph, probabilities, ax):
    """
    Draws the graph as a matplotlib plot.
    Depots are colored in red. Edges that have been
    traveresed
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
        "connectionstyle" : 'arc3, rad = 0.1',
    }
    nx.draw_networkx_edges(G_nx, pos, **options)


def sample_draw_probs_graph(batch, preds):
    fig, ax = plt.subplots()
    selected = random.randrange(len(batch))
    draw_probs_graph(batch[selected], preds[selected], ax)
    return fig, ax