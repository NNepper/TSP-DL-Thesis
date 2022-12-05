# Libs
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

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
        output_masked = output.masked_fill(mask.bool(), float('-1e8'))
        probs = self.softmax(output_masked)
        return probs


class AgentMLP:
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
            layer_dim=layer_dim,
            layer_number=layer_number
        )

        # Optimization
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def predict(self, env):

        # Trajectory
        done = False
        rewards = torch.zeros(env.batch_size, env.num_nodes - 1, dtype=torch.float, device=self.device)
        log_probs = torch.zeros(env.batch_size, env.num_nodes - 1, dtype=torch.float, device=self.device)
        tour = [env.depots]
        t = 0
        while not done:
            # Current state
            state = torch.tensor(env.get_state(), dtype=torch.float, device=self.device)

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

        for i in range(epochs):
            self.model.train()
            env.restart()

            # Predict tour
            tour, log_probs, rewards = self.predict(env)

            # Compute Discounted rewards for the trajectory
            G = discounted_rewards(rewards, self.gamma)

            # Back-propagate the policy loss for each timestep
            self.optimizer.zero_grad()
            for b in range(env.batch_size):
                for t in range(env.num_nodes - 1):
                    policy_loss_t = -log_probs[b,t] * G[b,t]
                    policy_loss_t.backward(retain_graph=True)
            self.optimizer.step()

            # report
            G_list.append(G[0,0])
            if i % 100 == 0:
                #best_sol.render()
                print(f"epoch n°{i}: {G[0,0]}")
        return G_list
