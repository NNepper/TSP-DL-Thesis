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
agent = BasicAgent()

# Training
agent.train(env)