import numpy as np
from typing import Any

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


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
        self, state_space: tuple, action_space: int, continuous: bool = False,
        hidden_actor: list[int] = [64, 128, 256, 256, 256, 128, 64],
        hidden_critic: list[int] = [64, 128, 256, 256, 256, 128, 64],
    ):
        super(ACAgent, self).__init__()
        
        self.state_space = np.array(state_space).prod() # input state
        self.action_space = action_space                # output action

        self.hidden_actor = hidden_actor   # output action
        self.hidden_critic = hidden_critic # output action

        self.actor = build_mlp(self.state_space, self.action_space, hidden_actor, out_std=0.01, continuous=continuous)
        self.critic = build_mlp(self.state_space, 1, hidden_critic, out_std=1.0, continuous=False) # do not change the last layer

        self.continuous = continuous
        if self.continuous:
            # actor log standard deviation, each is independent from each other
            self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_space)))


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
        if self.continuous:
            # Sample normal distributions per each output action
            action_mean = self.actor(state)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
        else:
            # Get a discrete action with a Softmax like operation 
            logits = self.actor(state)
            probs = Categorical(logits=logits)


        # To get the probs when actions has been already taken
        action = probs.sample() if action is None else action

        # Sum the logs probs bc independent if continuous or something 
        log_prob, entropy = probs.log_prob(action), probs.entropy()
        if self.continuous:
            log_prob, entropy = log_prob.sum(1), entropy.sum(1)

        return action, log_prob, entropy, self.critic(state)



# =================================================================================
#                                    MODEL lAYERS
# =================================================================================
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

def build_mlp(in_dim: int, out_dim: int, hidden_sizes: list[int], out_std: float, continuous: bool = False):
    """Builds the sequential layers for a submodel.

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
    if continuous:
        layers.append(layer_init(nn.Linear(prev, np.prod(out_dim)), std=out_std))
    else:
        layers.append(layer_init(nn.Linear(prev, out_dim), std=out_std))

    return nn.Sequential(*layers)