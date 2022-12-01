from agents import BasicAgent
from gym_vrp.envs import TSPEnv

# Environment
env = TSPEnv(
    num_nodes=10,
    batch_size=1,
    num_draw=1,
    seed=69
)

# Agent
agent = BasicAgent(
    graph_size=10,
    layer_number=4,
    lr=1e-3,
)

if __name__ == '__main__':
    agent.train(env, 10000)