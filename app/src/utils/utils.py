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
#                                    MODEL
# =================================================================================
def save_agent(CONFIG: Configuration, agent: AgentAC) -> None:
    """Saves the agent in memory.

    It assigns it a version. If the version is already created check for
    higher version values

    Args:
        agent (AgentAC): Trained  agent
        CONFIG (Configuration): Configuration
    """
    version = 0
    model_path = os.path.join(CONFIG.models_path, f"{CONFIG.exp_name}-v{version}.pt")
    while os.path.exists(model_path):
        version += 1
        model_path = os.path.join(CONFIG.models_path, f"{CONFIG.exp_name}-v{version}.pt")
    
    os.makedirs(CONFIG.models_path, exist_ok=True)
    torch.save(agent.state_dict(), model_path)
    print(f" - Model saved to {model_path}")
    

def load_agent(CONFIG: Configuration, agent: AgentAC = None) -> AgentAC:
    """Load and agent from memory.

    If config has no version specified, it will look for the newest.

    Args:
        CONFIG (Configuration): Configuration
        agent (AgentAC, optional): Already created model with envs. Defaults to None.

    Raises:
        RuntimeError: Failed to load state_dic

    Returns:
        AgentAC: Loaded model
    """
    if agent is None:
        envs = get_envs(CONFIG)
        agent = AgentAC(*get_shape_from_envs(envs))

    if CONFIG.model_version is None:
        trained_agents, _ = list_dir_files(CONFIG.models_path)
        # From the model path, get the name (no extension), split by version and keep the number
        agent_name = os.path.basename(trained_agents[-1])
        CONFIG.model_version = os.path.splitext(agent_name)[0].split("-v")[-1]

    agent_path = os.path.join(CONFIG.models_path, f"{CONFIG.exp_name}-v{CONFIG.model_version}.pt")
    try:
        print(f" - Loading agent at {agent_path}")
        state = torch.load(agent_path, map_location=CONFIG.device)
        agent.load_state_dict(state)
        agent.to(CONFIG.device)
        agent.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load state_dict from {agent_path}: {e}") from e
    
    return agent

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