# Libs
import torch
import torch.optim as optim

from agents.basis import Agent


class AgentVanilla(Agent):
    def __init__(self,
                 config,
                 model):
        # Load from memory if already defined
        loaded_model, loaded_config = super().__init__(config["directory"])
        if loaded_config is not None and not config["tune"]:
            config = loaded_config
            model = loaded_model

        self.config = config
        self.tune = config["tune"]
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

        # TODO: Implementing prediction

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

        # TODO: Implementing train

        return best_sol, best_length, G[:, 0]
