import numpy as np
from typing import Any

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from src.utils import build_mlp

# ========================================================================================
#                                    AUX FUNCTIONS
# ========================================================================================
class ACAgent(nn.Module):
    """Actor-Critic agent for reinforcement learning.

    This agent supports fully-connected (MLP) architectures. 
    It outputs both action logits (actor) and state-value estimates (critic).

    Args:
        state_space (tuple): Shape of the observation/state space.
        action_space (int): Number of discrete actions.
        hidden_actor (list[int], optional): Sizes of hidden layers for the actor MLP. 
            Defaults to [64, 128, 256, 256, 256, 128, 64].
        hidden_critic (list[int], optional): Sizes of hidden layers for the critic MLP.
            Defaults to [64, 128, 256, 256, 256, 128, 64].
    """
    def __init__(
        self, state_space: tuple, action_space: int,
        hidden_actor: list[int] = [64, 128, 256, 256, 256, 128, 64],
        hidden_critic: list[int] = [64, 128, 256, 256, 256, 128, 64],
    ):
        super(ACAgent, self).__init__()
        self.state_space = np.array(state_space).prod() # input state
        self.action_space = action_space                # output action

        self.hidden_actor = hidden_actor   # output action
        self.hidden_critic = hidden_critic # output action

        self.actor = build_mlp(self.state_space, self.action_space, hidden_actor, out_std=0.01)
        self.critic = build_mlp(self.state_space, 1, hidden_critic, out_std=1.0)


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


