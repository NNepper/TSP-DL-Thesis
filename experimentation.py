import pickle

import torch
import tqdm
from torch_geometric.loader import DataLoader

from common.utils import sample_draw_probs_graph
from model.Graph2Graph import Graph2Graph


def entropy_mixed_loss(batch_pi, batch_distances, opt_length):
    """
    The entropy_mixed_loss function takes in a batch of pi values, a batch of pairwise distances, and the optimal length
    for each tour. It then calculates the loss for each tour by first decoding the greedy path from pi and calculating its
    length. Then it subtracts this length from the optimal length to get an approximation of how far off we are. Finally,
    it adds up all log(pi) values along that path to calculate entropy (which is subtracted because we want higher entropy).

    :param batch_pi: Calculate the entropy of the tour
    :param batch_distances: Compute the greedy tour length
    :param opt_length: Calculate the loss
    :return: A loss for each batch
    """
    loss = torch.zeros(opt_length.shape)
    pairwise_distances = torch.split(batch_distances, batch_distances.shape[0] // opt_length.shape[0])
    for i, pred in enumerate(batch_pi):
        # Greedy Decoding of the tour
        curr_idx = 0
        pred_length = .0
        entropy_sum = .0
        pi_cpy = torch.clone(pred)
        for _ in range(len(opt_length) - 1):
            next_idx = torch.argmax(pi_cpy[curr_idx, :])
            pred_length += pairwise_distances[i][curr_idx, next_idx]

            entropy_sum += torch.log(pred[curr_idx, next_idx])
            pi_cpy[:, next_idx] = .0
            curr_idx = next_idx
        loss[i] = pred_length - opt_length[i] - entropy_sum
    return loss


def policy_gradient_loss(batch_pi, batch_distances, opt_length):
    """
    The policy_gradient_loss function takes in a batch of predicted tours and the optimal tour lengths,
    and returns a loss value. The loss is calculated by taking the negative log probability of each edge in
    the predicted tour multiplied by its distance, then subtracting that from the optimal length. This means
    that we are trying to maximize our likelihood of predicting edges with shorter distances.

    :param batch_pi: Calculate the loss
    :param batch_distances: Calculate the loss
    :param opt_length: Calculate the loss
    :return: The loss of the policy gradient
    """
    loss = torch.zeros(opt_length.shape)
    pairwise_distances = torch.split(batch_distances, batch_distances.shape[0] // opt_length.shape[0])
    for i, pred in enumerate(batch_pi):
        # Greedy Decoding of the tour
        curr_idx = 0
        pred_length = .0
        entropy_sum = .0
        for _ in range(len(opt_length) - 1):
            next_idx = torch.argmax(pred[curr_idx, :])
            loss[i] += torch.log(pred[curr_idx, next_idx]) * pairwise_distances[i][curr_idx, next_idx]
            curr_idx = next_idx
        loss[i] = pred_length - opt_length[i]
    return loss


def custom_loss(batch_pi, batch_distances, opt_length):
    """
    The custom_loss function takes in the predicted probability distribution over all possible routes,
    the pairwise distances between each node, and the optimal length of a route. It then calculates
    the expected length of a route based on the probabilities given by batch_pi. If this expected length is
    greater than or equal to opt_length, it returns 0; otherwise it returns (expected_length - opt_length).
    This loss function penalizes non-Hamiltonian tours by adding an additional penalty term to their cost.

    :param batch_pi: Calculate the expected length of a route
    :param batch_distances: Calculate the actual length of the route
    :param opt_length: Calculate the penalty for non-hamiltonian tours
    :return: The absolute difference between the expected length of a tour and the optimal length
    """
    expected_length = torch.zeros(batch_pi.shape[0])
    traversed_nodes = set()
    pairwise_distances = torch.split(batch_distances, batch_distances.shape[0] // opt_length.shape[0])
    for i, pred in enumerate(batch_pi):
        route = torch.argmax(pairwise_distances[i], dim=1)
        curr_idx = 0
        for next_idx in route:
            expected_length += pred[curr_idx, next_idx] * pairwise_distances[i][curr_idx, next_idx]
            traversed_nodes |= {int(next_idx)}
            curr_idx = next_idx
        if len(traversed_nodes) < pairwise_distances[i].shape[0]:  # Penalize non-Hamiltonian tour
            penalty = (len(traversed_nodes) - pairwise_distances[i].shape[0]) * opt_length
            return torch.abs((expected_length + penalty) - opt_length)
    return torch.abs(expected_length - opt_length)


if __name__ == '__main__':
    # Data importing
    with open('data/dataset_10.pkl', 'rb') as f:
        graphs, target, opt_length = pickle.load(f)
        dataLoader = DataLoader(graphs, batch_size=128)

    # Model Initialization
    model = Graph2Graph(graph_size=10, hidden_dim=124)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")

    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)

    plot_counter = 0
    for batch in tqdm.tqdm(dataLoader):
        total_loss = total_examples = prev_loss = 0
        for epoch in range(1, 100000):
            optimizer.zero_grad()

            X = torch.concat((batch.x, batch.coordinates), dim=1).to(torch.float32).to(device)
            pred = model.forward(X, batch.edge_index)

            loss = custom_loss(pred, batch.x, batch.y).sum()

            loss.backward()
            optimizer.step()
            total_loss += loss.sum()
            total_examples += pred.numel()
            if epoch % 50 == 0:
                print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
                if abs(total_loss - prev_loss) > 10e-6:
                    prev_loss = total_loss
                else:
                    print(loss)
                    fig, axs = sample_draw_probs_graph(batch, pred)
                    fig.savefig(f"{plot_counter}.png")
                    plot_counter += 1
                    break
