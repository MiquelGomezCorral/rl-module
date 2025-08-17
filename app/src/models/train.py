import time
import numpy as np
import gymnasium as gym

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.config import Configuration
from src.models.agent import AgentAC
from src.models.env_management import get_envs


def train_ppo(CONFIG: Configuration, writer: SummaryWriter) -> None:
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

    print(f"Observation dimension: {agent.obs_dim}. Action dimensions: {agent.action_dim}")
    # ================== VARS ==================
    # Store setup
    obs      = torch.zeros((CONFIG.n_steps, CONFIG.n_envs) + (agent.obs_dim,)   ).to(CONFIG.device)
    # actions  = torch.zeros((CONFIG.n_steps, CONFIG.n_envs) + (agent.action_dim,)).to(CONFIG.device)
    actions  = torch.zeros((CONFIG.n_steps, CONFIG.n_envs)).to(CONFIG.device)
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
    # print(agent.get_value(next_obs))
    # print(agent.get_action_value(next_obs))

    # Episodes?
    for update in range(1, num_updates + 1):
        # 1. Annealing the rate if instructed to do so.
        if CONFIG.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates # 1 beginning decreases -> 0
            lr_now = frac * CONFIG.learning_rate
            optimizer.param_groups[0]["lr"] = lr_now

        # 2. Collect observations
        for step in range(CONFIG.n_steps):
            # 2.1 Updating variables
            global_step += CONFIG.n_envs
            obs[step] = next_obs
            dones[step] = next_dones

            # 2.2 Getting the model actions / predictions
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob   
        
            # 2.3 Acting in the environment
            next_obs, reward, term, trunc, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(CONFIG.device).view(-1)
            next_obs   = torch.Tensor(next_obs).to(CONFIG.device)
            next_dones = torch.Tensor(term | trunc).to(CONFIG.device)

            if global_step % 1_000 == 0:
                for k, v in info.items():
                    if k != "episode": continue
                    print(f"{global_step = } | {step = }")
                    print(f" - episodic return {v['r']} | mean {v['r'].mean().item(): .2f}")
                    print(f" - episodic length {v['l']} | mean {v['l'].mean().item(): .2f}")
                    writer.add_scalar("charts/episodic_return", v['r'].mean().item(), global_step)
                    writer.add_scalar("charts/episodic_length", v['l'].mean().item(), global_step)
                    break



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