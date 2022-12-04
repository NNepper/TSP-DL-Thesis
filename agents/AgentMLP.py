# Libs
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from collections import deque

from gym_vrp.envs import TSPEnv
from common import discounted_rewards


class PolicyNetMLP(nn.Module):
    def __init__(self, input_dim, layer_dim, layer_number, output_dim):
        super().__init__()

        # Dense-Layer
        self.dense = nn.Sequential(
            nn.Linear(input_dim, layer_dim),
            nn.ReLU())
        for _ in range(layer_number - 2):
            self.dense.append(nn.Linear(layer_dim, layer_dim))
            self.dense.append(nn.ReLU())
        self.dense.append(nn.Linear(layer_dim, output_dim))
        self.dense.append(nn.ReLU())

        # Output softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, mask):
        output = self.dense(input)
        output_masked = output.masked_fill(mask.bool(), float('-1e2'))
        probs = self.softmax(output_masked)
        return probs


class AgentMLP:
    def __init__(self,
                 graph_size: int,
                 layer_number: int,
                 seed: int = 88,
                 gamma: float = 0.50,
                 lr=.001):

        super().__init__()
        self.gamma = gamma

        # Torch configuration
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

        # Policy model
        input_dim = (graph_size * 2) + graph_size + 1
        self.model = PolicyNetMLP(
            input_dim=input_dim,
            output_dim=graph_size,
            layer_dim=input_dim,
            layer_number=layer_number
        )

        # Optimization
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def predict(self, env):
        state = torch.tensor(env.get_state(), dtype=torch.float, device=self.device)

        # Trajectory
        done = False
        rewards = torch.zeros(env.batch_size, env.num_nodes - 1, dtype=torch.float, device=self.device)
        log_probs = torch.zeros(env.batch_size, env.num_nodes - 1, dtype=torch.float, device=self.device)
        tour = [env.depots]
        t = 0
        while not done:
            # Mask already visited nodes
            mask = torch.tensor(env.generate_mask(), dtype=torch.float, device=self.device)

            # Forming the input vector
            input = torch.hstack((
                torch.tensor(env.depots),
                state[:, :, 0],
                state[:, :, 1],
                mask
            ))

            # Find probabilities
            policy = self.model(input, mask)

            # Sample the action
            sampler = Categorical(policy)
            selected = sampler.sample().unsqueeze(1)

            # Compute loss
            _, loss, done, _ = env.step(selected)
            env.render()

            # Store result
            # TODO: Make code more efficient with batch operations through big Tensor
            state = torch.tensor(env.get_state(), dtype=torch.float, device=self.device)
            rewards[:, t] += loss
            log_probs[:, t] = torch.gather(torch.log(policy), dim=1, index=selected).squeeze()
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
        for i in range(epochs):
            self.model.train()
            env.reset()

            # Predict tour
            tour, log_probs, rewards = self.predict(env)

            # Compute Reward for the trajectory
            G = discounted_rewards(rewards, self.gamma)

            # Compute Policy loss
            policy_loss = -log_probs * G

            # Back-propagate the policy loss
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            # report
            if i % 1000 == 0:
                print(f"epoch {i}: loss={policy_loss}")
