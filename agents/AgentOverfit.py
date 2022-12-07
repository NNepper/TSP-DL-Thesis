# Libs
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from common import discounted_rewards


class PolicyNetOverfit(nn.Module):
    def __init__(self, input_dim, layer_dim, layer_number, output_dim):
        super().__init__()

        # Dense-Layer
        self.dense = nn.Sequential(
            nn.Linear(input_dim, layer_dim),
            nn.ReLU(),
            nn.Linear(layer_dim, layer_dim),
            nn.ReLU()
        )

        for _ in range(layer_number - 2):
            self.dense.append(nn.Linear(layer_dim, layer_dim))
            self.dense.append(nn.ReLU())


        self.dense.append(nn.Linear(layer_dim, output_dim))

        # Output softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        output = self.dense(input)
        probs = self.softmax(output)
        return probs


class AgentOverfit:
    def __init__(self,
                 graph_size: int,
                 layer_number: int,
                 layer_dim: int,
                 seed: int = 88,
                 gamma: float = 0.99,
                 lr=.001):

        super().__init__()
        self.gamma = gamma

        # Torch configuration
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        # Policy model
        input_dim = graph_size * 2
        self.model = PolicyNetOverfit(
            input_dim=input_dim,
            output_dim=(graph_size-1)*graph_size,
            layer_dim=layer_dim,
            layer_number=layer_number
        ).to(self.device)

        # Optimization
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def predict(self, env):
        done = False
        t = 0

        # Trajectory
        rewards = torch.zeros(env.batch_size, env.num_nodes - 1, dtype=torch.float)
        log_probs = torch.zeros(env.batch_size, env.num_nodes - 1, dtype=torch.float)
        tour = []

        # Current state
        state = torch.tensor(env.get_state(), dtype=torch.float, device=self.device)

        # Forming the input vector
        input = torch.hstack((
            state[:, :, 0],
            state[:, :, 1],
        ))

        policy = self.model(input)

        for i in range(env.num_nodes-1):
            mask = torch.flatten(torch.tensor(np.array([env.generate_mask()] * (env.num_nodes-1))))
            policy = policy.masked_fill(mask.bool(), float('1e-8'))

            # Sample the action
            sampler = Categorical(policy[:,i*env.num_nodes:(i*env.num_nodes)+env.num_nodes])
            selected = sampler.sample().unsqueeze(0)

            # Compute loss
            _, loss, done, _ = env.step(selected)

            # Store result
            rewards[:, t] += loss
            log_probs[:, t] = sampler.log_prob(selected)
            tour.append(selected)
            t += 1

        return tour, log_probs, rewards

    def train(self, env, epochs: int = 100):
        """
        The train function is responsible for training the model.
        It takes an environment and number of epochs as arguments.

        :param self: Access the class attributes
        :param env:TSPEnv: Pass the environment to the model
        :param epochs:int=100: Specify the number of epochs to train for
        :return: The loss, which is the negative of the reward
        """
        G_list = []
        best_length = float("inf")
        best_env = copy.deepcopy(env)

        for i in range(epochs):
            env.restart()
            self.model.train()

            # Predict tour
            tour, log_probs, rewards = self.predict(env)

            # Normalized rewards
            rewards_b = rewards

            # Compute Discounted rewards for the trajectory
            G = discounted_rewards(rewards_b, self.gamma)

            # Back-propagate the policy loss for each timestep
            self.optimizer.zero_grad()
            policy_loss = torch.sum(-log_probs * G)
            policy_loss.backward()

            self.optimizer.step()


            # report
            tour_length = torch.sum(rewards)*-1
            if tour_length < best_length:
                best_length = tour_length
                best_env = copy.deepcopy(env)
            G_list.append(tour_length)


            if i % 100 == 0 and i>0:
                print('Trajectory {}\tAverage Score: {:.2f}'.format(i, np.mean(G_list[-100:-1])))
        return best_env, G_list
