import random

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from common.utils import compute_optimality_gap

def correlation_weight_distance(edges_probabilities, edges_distances):
    edges_correlation = np.correlate(edges_probabilities, edges_distances)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.hist(edges_correlation, alpha=0.7, edgecolor='black', color='gray')


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
        ax.scatter(nodes[i][0], nodes[i][1], s=100, color="black")
        ax.text(nodes[i][0], nodes[i][1] + 0.03, str(i), fontsize=10, ha="center")

    # draw tour solution
    for i in range(len(tour)-1):
        ax.plot(
            [nodes[int(tour[i])][0], nodes[int(tour[i+1])][0]],
            [nodes[int(tour[i])][1], nodes[int(tour[i+1])][1]],
            color=color,
            linewidth=2
        )
    line, = ax.plot(
        [nodes[int(tour[-1])][0], nodes[int(tour[0])][0]],
        [nodes[int(tour[-1])][1], nodes[int(tour[0])][1]],
        color=color,
        linewidth=2
    )
    return line


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


def draw_probability_graph(graph, predicted_tour, probability):
    fig, ax = plt.subplots()
    plt.suptitle("Prediction of the TSP solution using Graph-to-Sequence")

    # sample random node
    sampled = random.randint(0, len(graph)-1)

    # draw nodes
    for i in range(len(graph)):
        color = "red" if sampled == i else "black"
        ax.scatter(graph[i][0], graph[i][1], s=100, color=color)
        ax.text(graph[i][0], graph[i][1] + 0.03, str(i), fontsize=10, ha="center")

    # plot partial tour until random node i
    idx = 0
    while predicted_tour[idx] != sampled:
        ax.plot(
            [graph[int(predicted_tour[idx])][0], graph[int(predicted_tour[idx+1])][0]],
            [graph[int(predicted_tour[idx])][1], graph[int(predicted_tour[idx+1])][1]],
            color="black",
            linewidth=2
        )
        idx += 1
    
    # plot edge probability over nodes
    for j in range(idx+1, len(graph)):
        weight = probability[idx, sampled, j]
        norm = Normalize(vmin=0, vmax=1)
        ax.plot(
            [graph[sampled][0], graph[j][0]],
            [graph[sampled][1], graph[j][1]],
            linewidth=1,
            alpha=1,
            color=plt.cm.Blues(norm(weight))
        )
    return fig


def draw_solution_graph(graph, true_tour, predicted_tour):
    #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))

    #ax1.set_title("Predicted Tour")
    #ax2.set_title("Optimal Tour")
    
    fig, ax = plt.subplots()
    plt.suptitle("Prediction of the TSP solution using Graph-to-Sequence")
    line_1 = draw_tour_graph(ax, graph, predicted_tour, color="royalblue")
    line_2 = draw_tour_graph(ax, graph, true_tour, color="red")

    # Optimality gap
    gap = compute_optimality_gap(graph, predicted_tour, true_tour)
    plt.title(f"Optimality Gap: {gap:.2f}%")

    # Add a legend
    ax.legend([line_1, line_2],["Prediction", "Target"])

    return fig
