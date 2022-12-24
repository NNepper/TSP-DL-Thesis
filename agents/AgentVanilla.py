# Libs

import torch
import torch.optim as optim
from torch.distributions import Categorical

from agents import Agent
from common import discounted_rewards


class AgentVanilla(Agent):
    def __init__(self,
                 config,
                 model):
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
        self.device = torch.device("cuda:0" if config["cuda"] else 'cpu')

        # Policy model
        self.model = model.to(self.device)

        # Optimization
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, maximize=True)

    def predict(self, env):
        # Trajectory
        done = False
        rewards = torch.zeros(env.batch_size, env.num_nodes - 1, dtype=torch.float)
        log_probs = torch.zeros(env.batch_size, env.num_nodes - 1, dtype=torch.float)
        tour = [env.depots]
        for t in range(env.num_nodes - 1):
            # Current state
            state = torch.tensor(env.get_state(), dtype=torch.float, device=self.device)

            # Mask already visited nodes
            mask = torch.tensor(env.generate_mask(), dtype=torch.float, device=self.device)

            # Forming the input vector

            input = torch.hstack((
                torch.tensor(env.depots),
                state[:, :, 2:env.num_nodes + 2].flatten(1),
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
            self.log.info(f"epoch:[{i}/{epochs}] - G_0:{G[i,0]}")
            self.tensorboard_writer.add_scalar("Reward", G[i,0])
            self.tensorboard_writer.add_scalar("Epoch", i)
        self.tensorboard_writer.flush()
        return best_sol, best_length, G[:, 0]