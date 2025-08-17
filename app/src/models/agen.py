import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()