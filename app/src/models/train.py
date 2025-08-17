import time
import numpy as np
import gymnasium as gym

import torch
import torch.optim as optim

from src.config import Configuration
from src.models.agent import AgentAC
from src.models.env_management import get_envs


def train_ppo(CONFIG: Configuration) -> None:
    """Train a PPO model for the Configuration.gym_id env

    Args:
        CONFIG (Configuration): Configuration for the training
    """
    # ======================================================
    #                   ENV MANAGEMENT
    # ======================================================
    # envs = create_env(CONFIG)
    envs = get_envs(CONFIG)
    # ======================================================
    #                   AGENT & VARS
    # ======================================================
    # ================== AGENT ==================
    agent = AgentAC(envs).to(CONFIG.device)
    optimizer = optim.Adam(agent.parameters(), lr=CONFIG.learning_rate, eps=CONFIG.eps)

    print(agent.obs_dim)
    print(agent.action_dim)
    # ================== VARS ==================
    # Store setup
    obs      = torch.zeros((CONFIG.n_steps, CONFIG.n_envs) + (agent.obs_dim,)   ).to(CONFIG.device)
    actions  = torch.zeros((CONFIG.n_steps, CONFIG.n_envs) + (agent.action_dim,)).to(CONFIG.device)
    logprobs = torch.zeros((CONFIG.n_steps, CONFIG.n_envs)).to(CONFIG.device)
    rewards  = torch.zeros((CONFIG.n_steps, CONFIG.n_envs)).to(CONFIG.device)
    dones    = torch.zeros((CONFIG.n_steps, CONFIG.n_envs)).to(CONFIG.device)
    values   = torch.zeros((CONFIG.n_steps, CONFIG.n_envs)).to(CONFIG.device)

    # ================== OTHERS ==================
    global_step = 0
    start_time  = time.time()
    next_obs    = torch.Tensor(envs.reset()[0]).to(CONFIG.device)
    next_dones  = torch.zeros(CONFIG.n_envs).to(CONFIG.device)
    num_updates = CONFIG.total_timesteps // CONFIG.batch_size

    # ================== TRAININ LOOP ==================
    print(agent.get_value(next_obs))
    print(agent.get_action_value(next_obs))



#
def evaluate_model(model, CONIFG: Configuration):
    ... 
    # states, info = envs.reset()
    # for episode in range(10):
    #     done_envs = np.zeros(CONFIG.n_envs, dtype=bool)
    #     for step in range(200):
    #         action = envs.action_space.sample()
    #         states, rewards, terms, truncs, infos = envs.step(action)

    #         done_envs |= terms | truncs 
    #         if done_envs.all():
    #             break

    #     if 'episode' in infos:
    #         print(f"Episode {episode:2} ends at step {step:3} with rewards {infos['episode']['r']}")

        
    # envs.close()