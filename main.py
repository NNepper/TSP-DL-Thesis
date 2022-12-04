from agents import AgentMLP
from gym_vrp.envs import TSPEnv
import torch

# Environment
env = TSPEnv(
    num_nodes=10,
    batch_size=1,
    num_draw=1,
    seed=69
)
env.render()

# Agent
agent = AgentMLP(
    graph_size=10,
    layer_number=4,
    lr=1e-3,
)

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    agent.train(env, 1000)