import torch
from typing import Any
import gymnasium as gym

from src.config import Configuration


def get_shape_from_envs(envs: gym.Env) -> tuple:
    """Given the vectorized envs object, return the shape

    Args:
        envs (gym.Env): Vectorized envs

    Returns:
        tuple: 
            - State shape
            - Actions shape
    """
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space

    # Handle Dict observation spaces (e.g., VizDoom 'screen')
    if isinstance(obs_space, gym.spaces.Dict):
        # pick the 'screen' key for CNN input
        obs_shape = obs_space['screen'].shape
    else:
        obs_shape = obs_space.shape

    # Handle Discrete or Box action spaces
    if hasattr(act_space, "n"):
        action_shape = act_space.n
    else:
        action_shape = act_space.shape[0]

    return obs_shape, action_shape

def handle_states(CONFIG: Configuration, obs: Any) -> torch.Tensor:
    """Convert environment observations to a PyTorch tensor on the correct device.

    Handles both raw arrays and dict observations (e.g., ViZDoom) by extracting
    the 'screen' key if present. Moves the resulting tensor to the device
    specified in the configuration.

    Args:
        CONFIG (Configuration): Configuration object containing device info.
        obs (Any): Observation from the environment, can be a numpy array or a dict.

    Returns:
        torch.Tensor: Observation tensor ready for model input, on CONFIG.device.
    """
    if isinstance(obs, dict):
        obs = obs['screen']  # use the image part for CNN input
    obs = torch.as_tensor(obs, dtype=torch.float32, device=CONFIG.device)

    if CONFIG.convolutional:
        if obs.ndim == 4 and obs.shape[-1] in (1, 3):
            obs = obs.permute(0, 3, 1, 2)  # -> [N,C,H,W]
        else:
            raise ValueError(f"handle_states expected batched NHWC image, got {obs.shape}")

    return obs


def get_envs(CONFIG: Configuration, evaluating: bool = False) -> gym.vector.SyncVectorEnv:
    """
    Create and return a vectorized environment for training.

    This function generates multiple synchronized environments using `SyncVectorEnv`
    according to the number of environments specified in the configuration. It ensures
    that the action space is discrete (for compatibility with current agents).

    Args:
        CONFIG (Configuration): Configuration object containing environment settings, 
            including `gym_id` (str) and `n_envs` (int).

    Returns:
        gym.vector.SyncVectorEnv: A vectorized environment instance suitable for training
            with multiple parallel environments.
    """

    envs =  gym.vector.SyncVectorEnv([
        get_env_trunk(CONFIG, idx, evaluating)
        for idx in range(CONFIG.n_envs)
    ])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete actions space is supported"

    return envs


def create_env(CONFIG: Configuration, idx: int, evaluating: bool = False) -> gym.Env:
    """Create an environment from configuration

    Args:
        CONFIG (Configuration): Configuration

    Returns:
        gym.Env: Created env for CONFIG.gym_id env
    """
    env = gym.make(
        CONFIG.env_id, 
        render_mode = "rgb_array" if CONFIG.record_video else None
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    if CONFIG.record_video and (evaluating or idx == 0): # capture only the first 4 videos
        env = gym.wrappers.RecordVideo(
            env, 
            video_folder=CONFIG.videos_path, 
            fps=CONFIG.fps,
            name_prefix=f"env{idx}"
        )

    # NOTE: This seeds are different from the code seeds and are different for different envs
    seed = CONFIG.seed + idx 
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env

def get_env_trunk(CONFIG: Configuration, idx: int, evaluating: bool = False) -> callable:
    """Given a configuration reuturns a functions that can
    create environments?

    Args:
        CONFIG (Configuration): Configuration

    Returns:
        callable: Function that creates envs
    """
    return lambda: create_env(CONFIG, idx, evaluating)
