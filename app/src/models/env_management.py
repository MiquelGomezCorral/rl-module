import gymnasium as gym

from src.config import Configuration


def get_envs(CONFIG): 
    envs =  gym.vector.SyncVectorEnv([
        get_env_trunk(CONFIG, idx)
        for idx in range(CONFIG.n_envs)
    ])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete actions space is supported"

    return envs


def create_env(CONFIG: Configuration, idx: int) -> gym.Env:
    """Create an environment from configuration

    Args:
        CONFIG (Configuration): Configuration

    Returns:
        gym.Env: Created env for CONFIG.gym_id env
    """
    env = gym.make(
        CONFIG.gym_id, 
        render_mode = "rgb_array" if CONFIG.record_video else None
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if CONFIG.record_video and idx < 4: # capture only the firts 4 videos
        env = gym.wrappers.RecordVideo(
            env, 
            video_folder=CONFIG.VIDEOS, 
            fps=CONFIG.fps
        )

    # NOTE: This seeds are different from the code seeds and are different for different envs
    seed = CONFIG.seed + idx 
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env

def get_env_trunk(CONFIG: Configuration, idx: int) -> callable:
    """Given a configuration reuturns a functions that can
    create environments?

    Args:
        CONFIG (Configuration): Configuration

    Returns:
        callable: Function that creates envs
    """
    return lambda: create_env(CONFIG, idx)
