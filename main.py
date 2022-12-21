from agents import AgentVanilla
from common.utils import plot_performance
from gym_vrp.envs import TSPEnv
from model import PolicyFeedForward

if __name__ == '__main__':

    # Environment
    env = TSPEnv(
        num_nodes=20,
        batch_size=1,
        num_draw=1,
        seed=69
    )

    # Model
    PolicyNet = PolicyFeedForward(
        graph_size=env.num_nodes,
        layer_dim = 2048,
        layer_number=15,
    )

    # Agent
    agent = AgentVanilla(
        model = PolicyNet,
        lr=1e-4,
        gamma=.99,
        baseline=True
    )


    env.render()

    best_sol, length, rewards = agent.train(env, 50000)

    plot_performance(rewards)
    best_sol.render()
