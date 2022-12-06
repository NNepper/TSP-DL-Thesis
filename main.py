import numpy as np
import torch

from agents import AgentMLP
from common.utils import plot_performance
from gym_vrp.envs import TSPEnv

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    # Environment
    env = TSPEnv(
        num_nodes=20,
        batch_size=1,
        num_draw=1,
        seed=69
    )

    # Agent
    agent = AgentMLP(
        graph_size=20,
        layer_number=10,
        layer_dim = 128,
        lr=1e-12,
        gamma=1
    )

    rewards = agent.train(env, 100000)

    plot_performance(rewards)
