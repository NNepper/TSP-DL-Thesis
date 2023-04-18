import heapq
import numpy as np
import torch


class BFSearch:

    def __init__(self):
        ...

    def predict(self, batch_pi, batch_start):
        log_probs = torch.zeros(batch_pi.shape[0])
        tours = torch.zeros(batch_pi.shape[0], batch_pi.shape[1]).int()
        tours[:, 0] = batch_start

        # Greedy search procedure
        for i, pi in enumerate(batch_pi):
            curr = batch_start[i]
            visited_nodes = {int(batch_start[i])}
            all_nodes = set(range(pi.shape[0]))
            for j in range(1, pi.shape[0]):
                cands = list(all_nodes.difference(visited_nodes))
                next = torch.argmax(pi[curr, cands])
                log_probs[i] += torch.log(pi[curr, cands[next]])
                tours[i, j] = int(cands[next])
                curr = cands[next]
                visited_nodes |= {int(cands[next])}

        return tours, log_probs


class BeamSearch:

    def __init__(self, k: int):
        self.k = k  # Number of beam
        self.pq = heapq.heapify()

    def search(self, transition_matrix, start_node):
        return ...
