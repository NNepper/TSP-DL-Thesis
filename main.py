from agents import AgentMLP
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
        layer_number=15,
        layer_dim = 2048,
        lr=1e-4,
        gamma=.99,
        baseline=True
    )
    env.render()

    best_sol, length, rewards = agent.train(env, 20000)

    plot_performance(rewards)
    best_sol.render()
