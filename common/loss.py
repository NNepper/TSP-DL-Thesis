import random

import torch


def cross_entropy(predictions, solutions):
    loss = torch.zeros(predictions.shape[0]).float().to(predictions.device, non_blocking=True)
    # Compute the true edges term
    for i, tour in enumerate(solutions):
        # Forward tour
        for j, u in enumerate(tour):
            v = tour[(j + 1) % len(tour)] 
            loss[i] -= torch.log(torch.clamp(predictions[i, u, v], min=1e-6))
            loss[i] -= torch.log(torch.clamp(predictions[i, v, u], min=1e-6))
    return loss


def cross_entropy_negative_sampling(predictions, solutions, n_neg=5):
    """
    The cross_entropy_negative_sampling function computes the cross entropy loss for a batch of tours.

    :param prediction: The probability of each edge in the predicted tour
    :param solutions: The optimal tour for each graph in the batch
    :param n_neg=5: Sample 5 negative edges for each edge in the optimal tour
    :return: The loss of each tour in the batch
    """
    loss = torch.zeros(predictions.shape[0]).float().to(predictions.device, non_blocking=True)

    # Compute the true edges term
    for i, tour in enumerate(solutions):
        # Forward tour
        for j, u in enumerate(tour):
            v = tour[(j + 1) % len(tour)]
            loss[i] -= torch.log(torch.clamp(predictions[i, u, v], min=1e-6))
            loss[i] -= torch.log(torch.clamp(predictions[i, v, u], min=1e-6))

    # Compute the false edges term (neg_sampling)
    for i, tour in enumerate(solutions):
        for j, u in enumerate(tour):
            for v in random.sample(range(0, len(tour)), n_neg):
                if v != tour[(j + 1) % len(tour)]:
                    loss[i] -= torch.log(1 - torch.clamp(predictions[i, u, v], min=1e-6, max=1 - 1e-6))

    return loss

def cross_entropy_full(predictions, solutions):
    loss = torch.zeros(predictions.shape[0]).float().to(predictions.device, non_blocking=True)

    # Compute the true edges term
    for i, tour in enumerate(solutions):
        # Forward tour
        for j, u in enumerate(tour):
            v = tour[(j + 1) % len(tour)]
            loss[i] -= torch.log(torch.clamp(predictions[i, u, v], min=1e-6))
            loss[i] -= torch.log(torch.clamp(predictions[i, v, u], min=1e-6))

    # Compute the false edges term (full)
    for i, tour in enumerate(solutions):
        for j, u in enumerate(tour):
            for v in range(len(tour)):
                if v != tour[(j + 1) % len(tour)]:
                    loss[i] -= torch.log(1 - torch.clamp(predictions[i, u, v], min=1e-6, max=1 - 1e-6))
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
