# Libs
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from gym_vrp.envs import TSPEnv


class MLP(nn.Module):
    def __init__(self, input_dim: int,
                 output_dim: int,
                 layer_dim: int,
                 layer_number: int,
                 gamma: float = 0.50,
                 device=torch.device("cuda:0")):
        super().__init__()
        self.device = device
        self.gamma = gamma

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

    def forward(self, env: TSPEnv):
        state = torch.tensor(env.get_state(), dtype=torch.float, device=self.device)

        # Trajectory
        done = False
        rewards = []
        log_probs = []
        while not done:
            # Mask already visited nodes
            mask = torch.tensor(env.generate_mask(), dtype=torch.int, device=self.device)

            # Forming the input vector
            input = torch.hstack((
                torch.tensor(env.depots),
                state[:, :, 0].detach().clone(),
                state[:, :, 1].detach().clone(),
                mask
            ))

            # Find probabilities
            output = self.dense(input)
            probs = self.softmax(output.masked_fill(mask, float("-inf")))

            # Sample the action
            sampler = Categorical(probs)
            selected = sampler.sample()

            # Compute loss
            _, loss, done, _ = env.step(selected.unsqueeze(1))

            # Store result
            state = torch.tensor(env.get_state(), dtype=torch.float, device=self.device)
            rewards.append(loss)
            log_probs.append(torch.log(probs))

        # Compute Reward for the trajectory
        discounts = [self.gamma ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        return policy_loss


class BasicAgent:
    def __init__(self, graph_size: int, layer_number: int, lr: float, seed: int = 69):
        # Torch configuration
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

        # Model
        input_dim = (graph_size * 2) + graph_size + 1
        self.model = MLP(
            input_dim=input_dim,
            layer_dim=input_dim,  # TODO: change afterward
            layer_number=layer_number,
            output_dim=graph_size,
            device=self.device
        ).to(self.device)

        # Optimization
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, env: TSPEnv, epochs: int = 100):
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
            loss = self.model(env)

            # back-prop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # report
            print(f"epoch {i}: loss={loss}")
