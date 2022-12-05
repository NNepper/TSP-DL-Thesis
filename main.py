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
        layer_number=6,
        layer_dim = 128,
        lr=1e-6,
        gamma=0.70
    )

    rewards = np.zeros(shape=(10,10))
    for i in range(100):
        rewards[i] = agent.train(env, 10)
        env.reset()
        print(rewards)

    plot_performance(np.mean(rewards, axis=1))
