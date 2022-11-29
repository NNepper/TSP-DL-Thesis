# Libs
import random

import torch
import torch.distributions as distri
import torch.nn as nn
import torch.optim as optim

from gym_vrp.envs import TSPEnv


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, layer_dim: int, layer_number: int):
        # Dense-Layer
        self.dense = nn.Sequential(
            nn.Linear(input_dim, layer_dim),
            nn.ReLU())
        for _ in range(layer_number - 2):
            self.dense.append(nn.Linear(layer_dim, layer_dim))
            self.dense.append(nn.ReLU())
        self.dense.append(nn.Linear(layer_dim, output_dim))

        # Probability output
        self.softmax = nn.Softmax(output_dim, output_dim)

    def forward(self, env: TSPEnv, input):
        state = torch.tensor(env.get_state(), dtype=torch.float, device=self.device)

        # Forming the input vector
        input = torch.hstack([
            env.depots[0],
            torch.Tensor(input[0]),
            torch.Tensor(input[1]),
            torch.zeros(env.num_nodes),
        ])

        done = False
        acc_loss = torch.zeros((state.shape[0],), device=self.device)
        while not done:
            # Find probabilities
            output = self.dense(input)
            probs = self.softmax(output)
            selected = distri.Categorical(probs).sample()

            # Compute loss
            _, loss, done, _ = env.step(selected)

            # Store result
            state = torch.tensor(env.get_state(), dtype=torch.float, device=self.device)
            # TODO: Loss

            # Prepare input for next timestep
            # TODO: Update and pass to next node
        return acc_loss


class BasicAgent:
    def __init__(self, graph_size: int, layer_dim: int, layer_number: int, lr: float, seed: int = 69):
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
            layer_dim=layer_dim,
            layer_number=layer_number
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