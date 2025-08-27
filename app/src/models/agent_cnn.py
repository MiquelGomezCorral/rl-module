import numpy as np

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from src.utils import build_mlp

# ========================================================================================
#                                           AGENT
# ========================================================================================
class ACAgentCNN(nn.Module):
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
        self, state_space: tuple, action_space: int,
        hidden_actor: list[int] = [64, 128, 256, 256, 256, 128, 64],
        hidden_critic: list[int] = [64, 128, 256, 256, 256, 128, 64],
        cnn_input_channels: int = 3, cnn_feature_dim: int = 256
    ):
        super(ACAgentCNN, self).__init__()
        self.state_space = np.array(state_space).prod() # input state
        self.action_space = action_space                # output action

        self.hidden_actor = hidden_actor   # output action
        self.hidden_critic = hidden_critic # output action

        self.cnn = CNNEncoder(cnn_input_channels, cnn_feature_dim)
        self.cnn_input_channels = cnn_input_channels
        self.cnn_feature_dim = cnn_feature_dim

        self.actor = build_mlp(self.cnn_feature_dim, self.action_space, hidden_actor, out_std=0.01)
        self.critic = build_mlp(self.cnn_feature_dim, 1, hidden_critic, out_std=1.0)


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
        return self.cnn(x / 255.0)  # normalize pixels


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




class CNNEncoder(nn.Module):
    def __init__(self, input_channels: int = 3, feature_dim: int =256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # dummy input to compute output size after convs
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 84, 84)  # adjust to your frame size
            n_flat = self.conv(dummy).shape[1]

        self.fc = nn.Linear(n_flat, feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        x = self.relu(x)
        return x