import torch
from torch import nn
from torchrl.modules import NoisyLinear
import torch.nn.functional as F
import numpy as np


class FeedForwardNN(nn.Module):
    """Class for actor and critic networks."""

    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, obs):
        """Forward pass on network"""
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        return output


class NoisyFeedForwardNN(nn.Module):
    """Class for actor and critic networks with noisy network"""

    def __init__(self, in_dim, out_dim):
        super(NoisyFeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.noisy_layer = NoisyLinear(64, out_dim)

    def forward(self, obs):
        """Forward pass on network"""
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.noisy_layer(activation2)

        return output
