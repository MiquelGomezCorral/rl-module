import torch
from typing import Any
import gymnasium as gym

from src.config import Configuration
from src.custom_envs import Env2048

import ale_py
gym.register_envs(ale_py)


CUSTOM_MODELS_LIST = {
    "Env2048": Env2048,
}
def make_env_with_customs(CONFIG: Configuration) -> gym.Env:
    """Instanciate the correct env depending if it a custom one or a gym built one

    Args:
        CONFIG (Configuration): Configuration

    Returns:
        gym.Env: Gym environment
    """
    render_mode = "rgb_array" if CONFIG.record_video else None
    
    return (
        CUSTOM_MODELS_LIST[CONFIG.env_id](render_mode = render_mode)
        if CONFIG.env_id in CUSTOM_MODELS_LIST else
        gym.make(CONFIG.env_id, render_mode = render_mode)
    )

def get_shape_from_envs(envs: gym.Env) -> tuple:
    """Given the vectorized envs object, return the shape

    Args:
        envs (gym.Env): Vectorized envs

    Returns:
        tuple: 
            - State shape
            - Actions shape
            - Continuous or not bool
    """
    continuous = isinstance(envs.single_action_space, gym.spaces.Box)
    return (
        envs.single_observation_space.shape, 
        envs.single_action_space.shape if continuous else envs.single_action_space.n, 
        continuous
    )

def handle_states(CONFIG: Configuration, states: Any) -> torch.Tensor:
    """Convert environment observations to a PyTorch tensor on the correct device.

    Handles both raw arrays and dict observations (e.g., ViZDoom) by extracting
    the 'screen' key if present. Moves the resulting tensor to the device
    specified in the configuration.

    Args:
        CONFIG (Configuration): Configuration object containing device info.
        state (Any): Observation from the environment, can be a numpy array or a dict.

    Returns:
        torch.Tensor: Observation tensor ready for model input, on CONFIG.device.
    """
    return torch.Tensor(states).to(CONFIG.device)


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
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete actions space is supported"

    return envs


def create_env(CONFIG: Configuration, idx: int, evaluating: bool = False) -> gym.Env:
    """Create an environment from configuration

    Args:
        CONFIG (Configuration): Configuration

    Returns:
        gym.Env: Created env for CONFIG.gym_id env
    """
    env = make_env_with_customs(CONFIG)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    if CONFIG.record_video and (evaluating or idx == 0): # capture only the first 4 videos
        env = gym.wrappers.RecordVideo(
            env, 
            video_folder=CONFIG.videos_path, 
            fps=CONFIG.fps,
            name_prefix=f"env{idx}{'-eval' if evaluating else ''}"
        )

    if CONFIG.convolutional:
        env = gym.wrappers.GrayscaleObservation(env) # Convert to grayscale if colors are not important
        env = gym.wrappers.ResizeObservation(env, shape=(84, 84)) # Reshape the image
        env = gym.wrappers.FrameStackObservation(env, 4) # Last for frames

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
