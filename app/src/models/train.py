import numpy as np
import gymnasium as gym

from src.models.env_management import get_env_trunk
from src.config import Configuration


def train_ppo(CONFIG: Configuration) -> None:
    """Train a PPO model for the Configuration.gym_id env

    Args:
        CONFIG (Configuration): Configuration for the training
    """
    # ================== ENV MANAGEMENT ==================
    # envs = create_env(CONFIG)
    envs = gym.vector.SyncVectorEnv([
        get_env_trunk(CONFIG, idx)
        for idx in range(CONFIG.n_envs)
    ])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete actions space is supported"
    print(f"{envs.single_action_space.shape = }")
    print(f"{envs.single_observation_space.shape = }")

    # ================== RUN ==================
    states, info = envs.reset()
    for episode in range(10):
        done_envs = np.zeros(CONFIG.n_envs, dtype=bool)
        for step in range(200):
            action = envs.action_space.sample()
            states, rewards, terms, truncs, infos = envs.step(action)

            done_envs |= terms | truncs 
            if done_envs.all():
                break

        if 'episode' in infos:
            print(f"Episode {episode:2} ends at step {step:3} with rewards {infos['episode']['r']}")

        
    envs.close()



