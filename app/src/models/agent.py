from typing import Any
import numpy as np

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class AgentAC(nn.Module):
    def __init__(self, envs):
        super(AgentAC, self).__init__()
        self.obs_dim = np.array(envs.single_observation_space.shape).prod() # intup state
        self.action_dim = envs.single_action_space.n                       # output action

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,1), std=1.0) #std 1 for some reason
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            # std small so initially all the parameters have similar probabilities of being chosen
            layer_init(nn.Linear(64, self.action_dim), std=0.01) 
        )


def layer_init(layer: Any, std: float = np.sqrt(2), bias_const: float = 0.0) -> Any:
    """Initialize the values of a layer

    Args:
        layer (Any): The layer
        std (float, optional): Standar deviation. Defaults to np.sqrt(2).
        bias_const (float, optional): The bias. Defaults to 0.0.

    Returns:
        Any: The updated layer
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer