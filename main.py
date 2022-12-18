import torch

from agents import AgentOverfit, AgentMLP
from common.utils import plot_performance
from gym_vrp.envs import TSPEnv


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
        layer_dim = 1024,
        lr=1e-6,
        gamma=1
    )
    env.render()

    best_sol, length, rewards = agent.train(env, 3000)

    plot_performance(rewards)
    best_sol.render()
