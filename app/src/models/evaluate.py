import torch
import numpy as np
from tqdm import tqdm
from time import time

from maikol_utils.print_utils import print_separator
from maikol_utils.time_tracker import print_time

from src.config import Configuration
from src.models.agent import AgentAC
from src.models.env_management import get_envs


def evaluate_agent(agent: AgentAC, CONFIG: Configuration) -> tuple[float, float]:
    """
    Evaluate the agent for n_eval_episodes and return average reward and std of reward.

    Args:
        model (AgentAC): Trained agent.
        CONFIG (Configuration): Configuration object.
        n_eval_episodes (int): Number of episodes to evaluate.

    Returns:
        mean_reward (float): Mean episodic reward.
        std_reward (float): Std of episodic reward.
    """
    print_separator(f"EVALUATING PPO AGENT '{CONFIG.exp_name}'", sep_type="START")
    print_separator("CONFIGURATION", sep_type="LONG")
    # ================================================================
    #                       ENV MANAGEMENT
    # ================================================================
    print(f" - Creating envs...")
    envs = get_envs(CONFIG)

    # ================================================================
    #                               VARS
    # ================================================================
    print(f" - Creating vars...")
    episode_rewards = []
    states = torch.Tensor(envs.reset()[0]).to(CONFIG.device)
    dones = torch.zeros(CONFIG.n_envs, dtype=bool)
    start_time = time()

    # ================================================================
    #                       EVALUATING LOOP
    # ================================================================
    print_separator("EVALUATING", sep_type="SUPER")
    print(f" - Evaluating for {CONFIG.n_eval_episodes}.")

    for episode in tqdm(range(1, CONFIG.n_eval_episodes + 1)):
        with torch.no_grad():
            actions, _, _, _ = agent.get_action_value(states)

        next_states, rewards, terms, truncs, infos = envs.step(actions.cpu().numpy())
        states = torch.Tensor(next_states).to(CONFIG.device)
        dones = terms | truncs

        episode_rewards.append(np.array(rewards))
        if all(dones): # stop early if all envs finished an episode
            break
    

    # ================================================================
    #                            DONE
    # ================================================================
    print_separator("RESUME", sep_type="LONG")
    envs.close()
    # If no rewards collected -> return NaNs (safe)
    if len(episode_rewards) == 0:
        mean_reward, std_reward = float("nan"), float("nan")
    else:
        # episode_rewards: list of (T, n_envs) per-step reward arrays
        rewards_arr = np.stack(episode_rewards, axis=0)      
        per_env_returns = rewards_arr.sum(axis=0)         
        mean_reward = float(np.mean(per_env_returns))
        std_reward  = float(np.std(per_env_returns))

    print(f" - Mean rewards: {mean_reward:.4f}+-{std_reward:.4f}")
    print_time(time() - start_time, prefix=" - ")

    return mean_reward, std_reward