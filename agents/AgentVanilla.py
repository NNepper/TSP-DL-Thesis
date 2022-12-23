# Libs
import copy

import torch
import torch.optim as optim
from torch.distributions import Categorical

from common import discounted_rewards


class AgentVanilla:
    def __init__(self,
                 model,
                 seed: int = 88,
                 gamma: float = 0.99,
                 baseline=True,
                 lr=.001):

        super().__init__()
        self.gamma = gamma
        self.baseline = baseline

        # Torch configuration
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

        # Policy model
        self.model = model.to(self.device)

        # Optimization
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, maximize=True)

    def predict(self, env):

        # Trajectory
        done = False
        rewards = torch.zeros(env.batch_size, env.num_nodes - 1, dtype=torch.float)
        log_probs = torch.zeros(env.batch_size, env.num_nodes - 1, dtype=torch.float)
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
                state[:, :, 2:env.num_nodes+2].flatten(1),
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
        G = torch.zeros(epochs, env.num_nodes - 1)
        self.model.train()

        best_sol = env
        best_length = float("inf")

        for i in range(epochs):
            env.restart()

            # Predict tour
            tour, log_probs, rewards = self.predict(env)

            # Compute Discounted rewards for the trajectory
            G[i, :] = discounted_rewards(rewards, self.gamma)

            # Back-propagate the policy loss for each timestep
            self.optimizer.zero_grad()
            policy_loss = (log_probs * G[i, :]).mean()
            policy_loss.backward()
            self.optimizer.step()

            # report
            length = -torch.sum(rewards)
            if length < best_length:
                best_length = length
                best_sol = copy.deepcopy(env)

            """if i % 100 == 0 and i != 0:
                print(
                    f'Trajectory {i}\tMean rewards: {G[i - 100:i, 0].mean()}\tBest length:{best_length}')"""

        return best_sol, best_length, G[:, 0]
