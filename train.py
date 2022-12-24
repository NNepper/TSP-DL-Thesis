import torch
import torch.nn as nn
from ray import tune, air
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

    _, length, rewards = agent.train(env, 5000)


# Environment
env = TSPEnv(
    num_nodes=5,
    batch_size=1,
    num_draw=1,
    seed=69
)

search_space = {
    "graph_size": 5,
    "layer_dim": tune.choice([64, 128, 256, 512]),
    "layer_number": tune.choice([2, 4, 8, 16]),
    "gamma": 1,
    "lr": tune.loguniform(1e-8, 1e-3),
}


def tune_model(env, model, search_space, experiment_name):
    # Directory of the results
    dir = f"/results/{model}_{experiment_name}"

    # Hyperparameters Tuning
    search_algo = OptunaSearch()
    trainable_with_resources = tune.with_resources(train_ray, {"cpu": 2})
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(
            search_algo=search_algo,
            metric="rewards",
            mode="max",
            num_samples=50,
        ),
        run_config=air.RunConfig(log_to_file=True),
        param_space=search_space
    )
    results = tuner.fit()
    best_config = results.get_best_result(metric="rewards", mode="max").config

    # Train with best Parameters
    net = PolicyFeedForward(config=best_config)
    agent = AgentVanilla(
        model=net,
        config=best_config
    )
    _, length, rewards = agent.train(env, 5000)

    # Save to dir
    agent.save(dir)

