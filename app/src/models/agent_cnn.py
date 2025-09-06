import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.distributions import Categorical, Normal

from src.models.agent import ACAgent, build_mlp, layer_init


class ACAgentCNN(ACAgent):
    """Actor-Critic agent with a CNN encoder for image-based RL environments.

    The CNN extracts features from raw pixels, which are then fed into
    separate MLPs for the actor (policy) and critic (value function).

    Args:
        state_space (tuple): Shape of the input observation (C, H, W).
        action_space (int): Number of discrete actions.
        hidden_actor (list[int], optional): Hidden layer sizes for actor MLP.
        hidden_critic (list[int], optional): Hidden layer sizes for critic MLP.
        cnn_input_channels (int, optional): Input channels for the CNN (e.g., 3 for RGB).
        cnn_feature_dim (int, optional): Output feature dimension from CNN encoder.
    """
    def __init__(
        self, state_space: tuple, action_space: int, continuous: bool = False,
        hidden_actor: list[int] = [256, 128, 64],
        hidden_critic: list[int] = [256, 128, 64],
        cnn_layers: list[dict] = [
            {'out': 32, 'k': 8, 's': 4, 'p': 0},
            {'out': 64, 'k': 4, 's': 2, 'p': 0},
            {'out': 64, 'k': 3, 's': 1, 'p': 0},
        ],
        cnn_input_channels: int = 4, cnn_feature_dim: int = 512,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        super(ACAgentCNN, self).__init__(state_space, action_space, continuous, hidden_actor, hidden_critic, device)

        if len(self.base_state_space) == 3:
            input_shape = (cnn_input_channels, self.base_state_space[1], self.base_state_space[2])  
        else: 
            input_shape = (cnn_input_channels, self.base_state_space[0], self.base_state_space[1])

        self.cnn_layers = cnn_layers
        self.cnn = build_cnn(input_shape, cnn_feature_dim, cnn_layers=cnn_layers, activation=nn.ReLU)


        self.cnn_input_channels = cnn_input_channels
        self.cnn_feature_dim = cnn_feature_dim


        self.actor = build_mlp(self.cnn_feature_dim, self.action_space, hidden_actor, out_std=0.01, continuous=continuous)
        self.critic = build_mlp(self.cnn_feature_dim, 1, hidden_critic, out_std=1.0, continuous=False) # do not change the last layer

        if self.continuous:
            self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_space)))



    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the input state.

        If convolutional mode is enabled, applies the CNN encoder 
        and normalizes pixel values to [0, 1]. Otherwise, this should 
        be replaced with identity or flattening logic.

        Args:
            x (torch.Tensor): Input state. Shape is either 
                (B, C, H, W) for images or (B, state_dim) for vectors.

        Returns:
            torch.Tensor: Encoded feature vector of shape (B, feature_dim).
        """
        # ensure tensor on model device
        device = next(self.parameters()).device
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(device)

        # convert HWC -> CHW if needed (gym returns H,W,C)
        if x.ndim == 4 and x.shape[1] != self.cnn_input_channels:
            x = x.permute(0, 3, 1, 2)

        x = x.float()

        # quick sanity checks
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise RuntimeError("NaN or Inf in observation tensor (forward_features)")

        return self.cnn(x / 255.0)  # normalize pixels
    
    def get_value(self, state: torch.Tensor):
        """
        Compute the critic value for a given state.

        Args:
            state (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Estimated state-value (B, 1).
        """
        features = self.forward_features(state)
        return self.critic(features)

    def get_action_value(self, state: torch.Tensor, action=None):
        """
        Sample an action and compute log-prob, entropy, and value estimate.

        Args:
            state (torch.Tensor): Input image tensor of shape (B, C, H, W).
            action (torch.Tensor, optional): Predefined action. If None, an action is sampled.

        Returns:
            tuple:
                - action (torch.Tensor): Chosen action.
                - log_prob (torch.Tensor): Log probability of the action.
                - entropy (torch.Tensor): Policy entropy for exploration.
                - value (torch.Tensor): Critic value estimate.
        """
        features = self.forward_features(state)

        if self.continuous:
            # mean from actor, std from learnable param
            action_mean = self.actor(features)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
        else:
            logits = self.actor(features)
            probs = Categorical(logits=logits)

        # To get the probs when actions has been already taken
        action = probs.sample() if action is None else action

        # Sum the logs probs bc independent if continuous or something 
        log_prob, entropy = probs.log_prob(action), probs.entropy()
        if self.continuous:
            log_prob, entropy = log_prob.sum(1), entropy.sum(1)

        return action, log_prob, entropy, self.critic(features)

def build_cnn(
    input_shape: tuple,
    feature_dim: int,
    cnn_layers: list = None,
    activation=nn.ReLU,
    flatten: bool = True,
):
    """
    Build a CNN encoder from a conv_layers description and return nn.Sequential.
    input_shape: (C, H, W)
    conv_layers: list of dicts or tuples, each: {'out':int, 'k':int, 's':int, 'p':int(optional)}
                 or tuples: (out, kernel, stride, padding)
    feature_dim: final linear output dim after convs
    """
    C, H, W = input_shape

    layers = []
    in_ch = C
    for spec in cnn_layers:
        if isinstance(spec, dict):
            out_ch = spec['out']; k = spec['k']; s = spec['s']; p = spec.get('p', 0)
        else:
            out_ch, k, s, p = spec

        layers += [layer_init(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p))]
        layers += [activation()]
        in_ch = out_ch

    seq = nn.Sequential(*layers)

    # compute flattened size by running a dummy through seq
    with torch.no_grad():
        dummy = torch.zeros(1, C, H, W)
        conv_out = seq(dummy).view(1, -1).shape[1]

    mlp_head = nn.Sequential(
        layer_init(nn.Linear(conv_out, feature_dim)),
        activation()
    ) if flatten else nn.Identity()

    return nn.Sequential(seq, nn.Flatten(), mlp_head)

       # self.cnn = nn.Sequential(
        #     layer_init(nn.Conv2d(cnn_input_channels, 32, kernel_size=8, stride=4)),
        #     nn.ReLU(),
        #     layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
        #     nn.ReLU(),
        #     layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
        #     nn.ReLU(),

        #     nn.Flatten(),
        #     layer_init(nn.Linear(64*7*7, cnn_feature_dim)),
        #     nn.ReLU(),
        # )