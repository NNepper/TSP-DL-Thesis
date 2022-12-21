# Libs
import copy
import random

import torch
import torch.optim as optim
from torch.distributions import Categorical

from common import discounted_rewards


class AgentPPO:
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
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        # Policy model
        self.model = model.to(self.device)

        # Optimization
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, maximize=True, weight_decay=0.01)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)

    def predict(self, env):

        # Trajectory
        done = False
        rewards = torch.zeros(env.batch_size, env.num_nodes, dtype=torch.float)
        probs = torch.zeros(env.batch_size, env.num_nodes, env.num_nodes, dtype=torch.float)
        tour = torch.zeros(env.batch_size, env.num_nodes)
        tour[:, 0] = torch.tensor(env.depots, dtype=torch.int32)
        t = 1
        while not done:
            # Current state
            state = torch.tensor(env.get_state(), dtype=torch.float, device=self.device)

            # Mask already visited nodes
            mask = torch.tensor(env.generate_mask(), dtype=torch.float, device=self.device)

            # Forming the input vector
            input = torch.hstack((
                torch.tensor(env.depots),
                state[:, :, 2:22].flatten(1),
                mask
            ))

            # Find probabilities
            policy = self.model(input, mask)

            # Sample the action
            sampler = Categorical(policy)
            selected = sampler.sample().unsqueeze(1).int()

            # Compute loss
            _, loss, done, _ = env.step(selected)

            # Store result
            rewards[:, t] += loss
            probs[:, t, :] = policy
            tour[:, t] = selected
            t += 1

        return tour, probs[:, 1:], rewards[:, 1:]

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
        b = torch.zeros(env.num_nodes - 1)
        policy_old = torch.ones(env.batch_size, env.num_nodes - 1, env.num_nodes) / env.num_nodes

        for i in range(epochs):
            env.restart()

            # Predict tour
            print(i)
            tour, policy, rewards = self.predict(env)

            # Compute Discounted rewards for the trajectory
            G[i, :] = discounted_rewards(rewards, self.gamma)

            # Discounted with Baseline
            advantage_t = G[i, :] - b

            # Back-propagate the policy loss for each timestep
            self.optimizer.zero_grad()
            policy_loss = torch.sum(clipped_loss(policy, advantage_t, policy_old, tour[:, 1:], 0.2))
            policy_loss.backward()

            self.optimizer.step()
            # self.scheduler.step()

            # Update old policy
            policy_old = policy.detach().clone()

            # report
            length = -torch.sum(rewards)
            if length < best_length:
                best_length = length
                best_sol = copy.deepcopy(env)
                b = G[i, :]

            if i % 10 == 0:
                print(
                    f'Trajectory {i}\tBaseline: {str(b[0])}\tBest length:{best_length}')

        return best_sol, best_length, G[:, 0]


def clipped_loss(policy, advantage, policy_old, selected, eps):
    # TODO: Handle the fact that some action probability may be zero due to the masking
    ratio = torch.gather(policy, index=selected.unsqueeze(2).long(), dim=2).squeeze(2) / \
            torch.gather(policy_old, index=selected.unsqueeze(2).long(), dim=2).squeeze(2)
    vanilla = ratio * advantage
    clipped = torch.clamp(ratio, min=1-eps, max=1+eps)* advantage

    return torch.min(vanilla, clipped)