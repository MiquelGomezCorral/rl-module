import numpy as np
from typing import Any

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class AgentAC(nn.Module):
    def __init__(self, envs):
        super(AgentAC, self).__init__()
        self.state_dim = np.array(envs.single_observation_space.shape).prod() # input state
        self.action_dim = envs.single_action_space.n                       # output action

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,128)),
            nn.Tanh(),
            layer_init(nn.Linear(128,128)),
            nn.Tanh(),
            layer_init(nn.Linear(128,128)),
            nn.Tanh(),
            layer_init(nn.Linear(128,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,1), std=1.0) #std 1 for some reason
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,128)),
            nn.Tanh(),
            layer_init(nn.Linear(128,128)),
            nn.Tanh(),
            layer_init(nn.Linear(128,128)),
            nn.Tanh(),
            layer_init(nn.Linear(128,64)),
            nn.Tanh(),
            # std small so initially all the parameters have similar probabilities of being chosen
            layer_init(nn.Linear(64, self.action_dim), std=0.01) 
        )

    def get_value(self, state: np.ndarray):
        """Get the critic value for state 

        Args:
            x (np.array): The state to evaluate

        Returns:
            tensor: The evaluated tensor
        """
        return self.critic(state)
    
    def get_action_value(self, state, action=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the action, log-probability, entropy, and state value for a given state.

        Args:
            state (torch.Tensor or np.ndarray): The input state(s) for the policy and value network.
            action (torch.Tensor, optional): Pre-selected action. If None, an action is sampled from the policy.

        Returns:
            tuple:
                action (torch.Tensor): Selected or sampled action(s).
                log_prob (torch.Tensor): Log-probability of the action(s) under the current policy.
                entropy (torch.Tensor): Entropy of the policy distribution for exploration measurement.
                value (torch.Tensor): Critic value estimate for the given state(s).
        """
        logits = self.actor(state)
        # Softmax like operation
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(state)



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