import torch
import torch.nn as nn
from ray import tune, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch

from agents import AgentVanilla
from gym_vrp.envs import TSPEnv
from model import PolicyFeedForward


def train_ray(config):
    net = PolicyFeedForward(graph_size=config["graph_size"],
                            layer_dim=config["layer_dim"],
                            layer_number=config["layer_number"])

    agent = AgentVanilla(
        model=net,
        lr=config["lr"],
        gamma=config["gamma"],
        baseline=True
    )

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    _, length, rewards = agent.train(env, 10000)

    # report
    results = {"length":float(length), "rewards" : rewards}
    session.report(results)
    return results


# Environment
env = TSPEnv(
    num_nodes=20,
    batch_size=1,
    num_draw=1,
    seed=69
)
if __name__ == '__main__':
    search_space = {
        "graph_size": 20,
        "layer_dim": tune.choice([64, 128, 256, 512, 1024, 2048]),
        "layer_number": tune.choice([2, 4, 8, 16]),
        "gamma": tune.loguniform(0.55, 0.99),
        "lr": tune.loguniform(1e-8, 1e-3),
    }

    search_algo = OptunaSearch()

    tuner = tune.Tuner(
        train_ray,
        tune_config=tune.TuneConfig(
            metric="length",
            mode="min",
            num_samples=500,
        ),
        run_config=air.RunConfig(log_to_file=True),
        param_space=search_space
    )
    results = tuner.fit()

    best_results = results.get_best_result(metric="length", mode="min")

    print("Best trial config: {}".format(best_results.config))
    print("Best trial best length: {}".format(best_results.metrics["length"]))
