import logging
import os
import pickle

import torch
from torch.utils.tensorboard import SummaryWriter


class Agent:
    def __init__(self, directory):
        self.directory = directory
        self.log = logging.getLogger(__name__)
        self.tensorboard_writer = SummaryWriter(f"{directory}/tensorboard.pt",
                                                filename_suffix="")
        if os.path.exists(f"{directory}/model.pt"):
            return load(directory)
    def save(self):
        # save NeuralNet
        model_dir = f"{self.directory}/model.pt"
        torch.save(self.model, model_dir)

        # Serialize objects
        agent_dir = f"{self.directory}/agent.pt"
        parameters = self.config
        with open(agent_dir, "wb") as agent_f:
            pickle.dump(parameters, file=agent_f)
        self.log.debug("Successfully saved the agent")

def load(directory):
    model = torch.load(f"{directory}/model.pt")
    with open(f"{directory}/agent.pt", "rb") as agent_f:
        config = pickle.load(agent_f)
    return model, config
