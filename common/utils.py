import numpy as np
import torch

def discounted_rewards(rewards, gamma):
    # Calculate discounted rewards, going backwards from end
    discounts = torch.tensor([gamma ** i for i in reversed(range(rewards.shape[1]))]).repeat(rewards.shape[0], 1)
    G = torch.cumsum(rewards * discounts)
    return G
