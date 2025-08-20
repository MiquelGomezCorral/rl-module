import os
import random
import torch
import numpy as np

from src.models.agent import AgentAC
from src.config import Configuration
# =================================================================================
#                                    GENERAL
# =================================================================================

def set_seed(seed: int, torch_deterministic: bool = None):
    """Set the seed for all modules

    Args:
        seed (int): The new seed
        torch_deterministic (bool, optional): If the module torch should behave deterministically. Defaults to None.
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

# =================================================================================
#                                    MODEL
# =================================================================================
def save_model(agent: AgentAC, CONFIG: Configuration) -> None:
    version = 0
    model_path = os.path.join(CONFIG.models_path, f"{CONFIG.exp_name}-v{version}.pt")
    while os.path.exists(model_path):
        version += 1
        model_path = os.path.join(CONFIG.models_path, f"{CONFIG.exp_name}-v{version}.pt")
    
    os.makedirs(CONFIG.models_path, exist_ok=True)
    torch.save(agent.state_dict(), model_path)
    print(f" - Model saved to {model_path}")
    

    