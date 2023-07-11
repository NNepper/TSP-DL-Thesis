import random

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from torch_geometric.utils import to_networkx


def correlation_weight_distance(edges_probabilities, edges_distances):
    edges_correlation = np.correlate(edges_probabilities, edges_distances)
    sns.set_theme(style="ticks")
    f, ax = plt.subplots(figsize=(7, 5))
    sns.despine(f)

    sns.histplot(edges_correlation)


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

def draw_tour_graph(ax, nodes, tour, color="red"):
    # draw nodes
    for i in range(len(nodes)):
        ax.scatter(nodes[i][0], nodes[i][1], s=100, color="blue")
        ax.text(nodes[i][0], nodes[i][1] + 0.03, str(i), fontsize=10, ha="center")

    # draw tour solution
    for i in range(len(tour)-1):
        ax.plot(
            [nodes[int(tour[i])][0], nodes[int(tour[i+1])][0]],
            [nodes[int(tour[i])][1], nodes[int(tour[i+1])][1]],
            color=color,
            linewidth=2
        )
    ax.plot(
        [nodes[int(tour[-1])][0], nodes[int(tour[0])][0]],
        [nodes[int(tour[-1])][1], nodes[int(tour[0])][1]],
        color=color,
        linewidth=2
    )


def draw_probs_graph(ax, graph, probabilities):
    """
    The draw_probs_graph function takes in a list of nodes, probabilities, and an axis.
    It then draws the nodes of the graph according to their position attributes.
    The edges are drawn with weights corresponding to the probability values passed in.

    :param nodes: List of nodes
    :param probabilities: Set the edge weights
    :return: A graph with the probabilities of each edge
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    nodes = graph["nodes"]
    graph_size = len(nodes)

    # draw nodes according to position attribute
    for i in range(graph_size):
        ax.scatter(nodes[i][0], nodes[i][1], s=100, color="blue")
        ax.text(nodes[i][0], nodes[i][1] + 0.03, str(i), fontsize=10, ha="center")

    # set edges weights
    edge_weights = {(u, v): {"probability": float(probabilities[u, v])} for u in range(graph_size) for v in
                    range(graph_size)}

    # draw edges with weights
    for i in range(graph_size):
        for j in range(i+1, graph_size):
            if edge_weights.get((i, j)):
                weight = edge_weights[(i, j)]["probability"]
                ax.plot(
                    [nodes[i][0], nodes[j][0]],
                    [nodes[i][1], nodes[j][1]],
                    linewidth=1,
                    alpha=1,
                    color=plt.cm.Blues(weight)
                )

    return fig, ax


def sample_draw_probs_graph(batch, preds):
    """
    The sample_draw_probs_graph function takes a batch of data and the predictions for that batch,
    and plots the probabilities of each class for one randomly selected sample from that batch.
    The function returns a tuple containing the figure and axis objects used to plot this graph.

    :param batch: Get the image and label from the batch
    :param preds: Plot the predictions on top of the actual values
    :return: A figure and an axis object
    """
    selected = random.randrange(len(batch))
    return draw_probs_graph(batch[selected], preds[selected])


def draw_solution_graph(graph, predicted_tour):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.set_title("Predicted Tour")
    ax2.set_title("Optimal Tour")

    draw_tour_graph(ax1, graph["nodes"], predicted_tour, color="blue")
    draw_tour_graph(ax2, graph["nodes"], graph["tour_nodes"], color="red")

    return fig
