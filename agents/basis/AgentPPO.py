# Libs
import copy
import random

import torch
import torch.optim as optim
from torch.distributions import Categorical

from agents import Agent
from common import discounted_rewards


class AgentPPO(Agent):
    def __init__(self, model, config):
        # Load from memory if already defined
        loaded_model, loaded_config = super().__init__(config["directory"])
        if loaded_config is not None:
            config = loaded_config
            model = loaded_model

        self.config = config
        self.gamma = config["gamma"]
        self.seed = config["seed"]
        self.lr = config["lr"]

        # Torch configuration
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed(config["seed"])
        torch.backends.cudnn.deterministic = True

        # Policy model
        self.model = model.to(self.device)

        # Optimization
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, maximize=True)

    def predict(self, env):
        # Trajectory
        rewards = torch.zeros(env.batch_size, env.num_nodes, dtype=torch.float)
        log_probs = torch.zeros(env.batch_size, env.num_nodes, env.num_nodes, dtype=torch.float)
        tour = torch.zeros(env.batch_size, env.num_nodes)
        tour[:, 0] = torch.tensor(env.depots, dtype=torch.int32)
        for t in range(env.num_nodes - 1):
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
            _, loss, _, _ = env.step(selected)

            # Store result
            rewards[:, t] += loss
            log_probs[:, t] = torch.log(policy)
            tour[:, t] = selected

        return tour, log_probs[:, 1:], rewards[:, 1:]

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
            tour, policy, rewards = self.predict(env)

            # Compute Discounted rewards for the trajectory
            G[i, :] = discounted_rewards(rewards, self.gamma)

            # Discounted with Baseline
            advantage_t = G[i, :] - b

            # Back-propagate the policy loss for each timestep
            self.optimizer.zero_grad()
            policy_loss = clipped_loss(policy, advantage_t, policy_old, tour[:, 1:], 0.2).sum()
            policy_loss.backward()
            self.optimizer.step()

            # Update old policy
            policy_old = policy.detach().clone()

            # Update Baseline (Greedy)
            length = -torch.sum(rewards)
            if length < best_length:
                best_length = length
                best_sol = copy.deepcopy(env)
                b = G[i, :]

            # report
            self.log.info(f"epoch:[{i}/{epochs}] - G_0:{G[i, 0]}")
            self.tensorboard_writer.add_scalar("Reward", G[i, 0])
            self.tensorboard_writer.add_scalar("Epoch", i)

        return best_sol, best_length, G[:, 0]


def clipped_loss(pi, advantage, pi_old, selected, eps):
    # Normalize the Policy
    # TODO: Adapt overall model to give a really small probability and avoid clippin inf in the loss

    pi_clip = torch.clamp(pi, min=10e-8)
    pi_old_clip = torch.clamp(pi_old, min=10e-8)

    ratio = torch.gather(pi_clip, index=selected.unsqueeze(2).long(), dim=2).squeeze(2) \
            / torch.gather(pi_old_clip, index=selected.unsqueeze(2).long(), dim=2).squeeze(2)
    vanilla = ratio * advantage
    clipped = torch.clamp(ratio, min=1 - eps, max=1 + eps) * advantage

    return torch.min(vanilla, clipped)
