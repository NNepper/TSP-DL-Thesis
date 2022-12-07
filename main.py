import torch

from agents import AgentOverfit
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
    agent = AgentOverfit(
        graph_size=20,
        layer_number=8,
        layer_dim = 1024,
        lr=1e-4,
        gamma=1
    )
    env.render()

    best_sol, rewards = agent.train(env, 5000)

    plot_performance(rewards)
