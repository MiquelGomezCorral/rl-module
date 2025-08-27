import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.models.agent import ACAgent, build_mlp


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
        self, state_space: tuple, action_space: int,
        hidden_actor: list[int] = [64, 128, 256, 256, 256, 128, 64],
        hidden_critic: list[int] = [64, 128, 256, 256, 256, 128, 64],
        cnn_input_channels: int = 3, cnn_feature_dim: int = 256
    ):
        super(ACAgentCNN, self).__init__(state_space, action_space, hidden_actor, hidden_critic)

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
        logits = self.actor(features)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(features)


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
            # nn.Flatten()
        )
        # adaptive pool -> always outputs (B, 64, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # final linear from conv channels -> feature_dim
        self.fc = nn.Linear(64, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W], float
        x = self.conv(x)
        x = self.global_pool(x)      # [B, 64, 1, 1]
        x = x.view(x.size(0), -1)    # [B, 64]
        x = self.fc(x)               # [B, feature_dim]
        return F.relu(x)