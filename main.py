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
        layer_number=6,
        layer_dim = 256,
        lr=1e-2,
        gamma=0.99
    )
    rewards = agent.train(env, 1000)
    plot_performance(rewards)
