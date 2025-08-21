import os
import torch
import random
import numpy as np
from time import time
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

from src.models.agent import AgentAC
from src.models.env_management import get_envs, get_shape_from_envs
from src.config import Configuration

from maikol_utils.file_utils import list_dir_files

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
#                                    EXTERNAL
# =================================================================================
def start_wandb_tensorboard(CONFIG: Configuration) -> SummaryWriter:
    """Manage wandb and tensorbard configuration and setup"""
    run_name = f"{CONFIG.gym_id}__{CONFIG.exp_name}__{CONFIG.seed}__{int(time())}"
    if CONFIG.track_run:
        import wandb

        wandb.init(
            project=CONFIG.wandb_project_name,
            entity=CONFIG.wandb_entity,
            sync_tensorboard=True,          # To sync with tensorboard
            config=asdict(CONFIG),
            name=run_name,
            monitor_gym= not CONFIG.record_video,
            save_code=True,
            dir=CONFIG.wandb_path
        )

    # ===================== TensorBoard =====================
    writer = SummaryWriter(os.path.join(CONFIG.runs_path, run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(CONFIG).items()])),
    )

    return writer