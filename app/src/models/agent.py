import numpy as np
from typing import Any

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class AgentAC(nn.Module):
    def __init__(
        self, state_space: tuple, action_space: int,
        hidden_actor: list[int] = [64, 128, 256, 256, 256, 128, 64],
        hidden_critic: list[int] = [64, 128, 256, 256, 256, 128, 64]
    ):
        super(AgentAC, self).__init__()
        self.state_space = np.array(state_space).prod() # input state
        self.action_space = action_space                # output action

        self.hidden_actor = hidden_actor   # output action
        self.hidden_critic = hidden_critic # output action

        self.critic = self._build_mlp(self.state_space, 1, hidden_critic, out_std=1.0)
        self.actor = self._build_mlp(self.state_space, self.action_space, hidden_actor, out_std=0.01)


    def _build_mlp(self, in_dim: int, out_dim: int, hidden_sizes: list[int], out_std: float):
        """Builds the sequential layers for a submodel

        Args:
            in_dim (int): In dimension.
            out_dim (int): Out dimension.
            hidden_sizes (list[int]): Size of the inner layers
            out_std (float): last layer std

        Returns:
            nn.Sequential: The sencuantialized layers.
        """
        layers = []
        # In size for in layer
        prev = in_dim
        for h in hidden_sizes:
            layers += [layer_init(nn.Linear(prev, h)), nn.Tanh()]
            prev = h
        # Out size for out layer
        layers.append(layer_init(nn.Linear(prev, out_dim), std=out_std))
        return nn.Sequential(*layers)

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