import argparse

import torch

from agents import AgentVanilla
from common.utils import plot_performance
from gym_vrp.envs import TSPEnv
from model import PolicyFeedForward

# Argument
parser = argparse.ArgumentParser(description='TSP Solver using Vanilla Policy Gradient')
parser.add_argument('--batch-size', type=int, default=1, help='input batch size for training (default: 1)')
parser.add_argument('--num_nodes', type=int, default=10, help='number fo nodes in the graphs (default: 10)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=69, help='random seed (default: 69)')
parser.add_argument('--layer_size', type=int, default=128, help='number of unit per dense layer')
parser.add_argument('--layer_number', type=int, default=4, help='number of layer')
parser.add_argument('--lr', type=float, default=.001, help='learning rate')
parser.add_argument('-gamma', type=float, default=1, help='discounting factor of the reward')
parser.add_argument('--directory', type=str, default="./results", help='path where model is or will be saved')

config = parser.parse_args()
config.cuda = config.cuda and torch.cuda.is_available()
config.tuning = False

# Environment
env = TSPEnv(
    num_nodes=config.num_nodes,
    batch_size=config.batch_size,
    num_draw=1,
    seed=config.seed
)

# Model
PolicyNet = PolicyFeedForward(vars(config))

# Agent
agent = AgentVanilla(config=vars(config), model=PolicyNet)

best_sol, length, rewards = agent.train(env, config.epochs)
agent.save()

plot_performance(rewards)
best_sol.render()
