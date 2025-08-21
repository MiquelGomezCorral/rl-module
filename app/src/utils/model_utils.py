import os
import torch
from typing import Any

from src.models.agent import AgentAC
from src.models.env_management import get_envs, get_shape_from_envs
from src.config import Configuration

from maikol_utils.file_utils import list_dir_files, make_dirs

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
    
    make_dirs(CONFIG.models_path)
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
        state = torch.load(agent_path, map_location=CONFIG.device, weights_only=False)

        if 'agent_state_dict' in state: # if it comes from a checkpoint
            print(f" - Loading from checkpoint")
            state = state['agent_state_dict']

        agent.load_state_dict(state)
        agent.to(CONFIG.device)
        agent.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load state_dict from {agent_path}: {e}") from e
    
    return agent


# =================================================================================
#                                    MODEL CHECKPOINTS
# =================================================================================
def save_checkpoint(
    CONFIG: Configuration, 
    agent: AgentAC, 
    optimizer: Any, 
    update: int, 
):
    """Save a checkpoint in a rotatory way.
    
    Keep the last CONFIG.keep_last_k. If all filled:
    rotate: v(k-1) → v(k-2), ..., v1→v0, drop v0.
    Last is always k-1

    Args:
        CONFIG (Configuration): Configuration
        agent (AgentAC): Agent class
        optimizer (Any): Optimizer
        update (int): Update step
    """
    # rotate: v(k-1) → v(k-2), ..., v1→v0, drop v0
    for v in range(1, CONFIG.keep_last_k):
        src = os.path.join(CONFIG.checkpoint_path, f"{CONFIG.exp_name}-v{v}.pt")
        dst = os.path.join(CONFIG.checkpoint_path, f"{CONFIG.exp_name}-v{v-1}.pt")
        if os.path.exists(src):
            os.replace(src, dst)

    checkpoint_path = os.path.join(CONFIG.checkpoint_path, f"{CONFIG.exp_name}-v{CONFIG.keep_last_k-1}.pt")
    
    torch.save({
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "update": update,
        "config": CONFIG,
    }, checkpoint_path)


def load_checkpoint(
    CONFIG: Configuration, 
    agent: AgentAC, 
    optimizer: Any
) -> int:
    """Load the last checkpoint agent

    Args:
        CONFIG (Configuration): Configuration
        agent (AgentAC): Agent class
        optimizer (Any): Optimizer

    Returns:
        int: Update step in which the trainnig process was was.
    """
    checkpoints, n_files = list_dir_files(CONFIG.checkpoint_path)
    # From the model path, get the name (no extension), split by version and keep the number
    if n_files == 0:
        print(" - No checkpoint found.")
        return 0  # Start from scratch
    
    agent_name = os.path.basename(checkpoints[-1])
    check_number = os.path.splitext(agent_name)[0].split("-v")[-1]
    checkpoint_path = os.path.join(CONFIG.checkpoint_path, f"{CONFIG.exp_name}-v{check_number}.pt")
    
    if not os.path.exists(checkpoint_path):
        print(" - No checkpoint found.")
        return 0  # Start from scratch
    
    checkpoint = torch.load(checkpoint_path, map_location=CONFIG.device, weights_only=False)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f" - Checkpoint loaded from {checkpoint_path}")

    return checkpoint["update"] # start from this update