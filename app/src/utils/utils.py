
import random
import torch
import numpy as np

from src.config import Configuration


def set_seed(seed: int, torch_deterministic: bool = None):
    """Set the seed for all modules

    Args:
        seed (int): The new seed
        torch_deterministic (bool, optional): If the module torch should behaeve deterministaclly. Defaults to None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch_deterministic is not None:
        torch.backends.cudnn.deterministic = torch_deterministic


def get_device(CONFIG: Configuration) -> torch.device:
    """Return the correct device acording to configuration and hardware

    Args:
        CONFIG (Configuration): Configuration

    Returns:
        torch.device: Selected device
    """
    return torch.device("cuda" if torch.cuda.is_available() and CONFIG.cuda else "cpu")