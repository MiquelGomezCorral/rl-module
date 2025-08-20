import time
import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
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
    next_done   = torch.zeros(CONFIG.n_envs).to(CONFIG.device)
    num_updates = CONFIG.total_timesteps // CONFIG.batch_size

    # ================== TRAINING LOOP ==================
    # print(agent.get_value(next_obs))
    # print(agent.get_action_value(next_obs))

    # Episodes?
    for update in range(1, num_updates + 1):
        # 1. Annealing the rate if config says so.
        if CONFIG.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates # 1 beginning decreases -> 0
            lr_now = frac * CONFIG.learning_rate
            optimizer.param_groups[0]["lr"] = lr_now

        # 2. Collect observations
        for step in range(CONFIG.n_steps):
            # 2.1 Updating variables
            global_step += CONFIG.n_envs
            obs[step]   = next_obs
            dones[step] = next_done

            # 2.2 Getting the model actions / predictions
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob   
        
            # 2.3 Acting in the environment
            next_obs, reward, term, trunc, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(CONFIG.device).view(-1)
            next_obs      = torch.Tensor(next_obs).to(CONFIG.device)
            next_done     = torch.Tensor(term | trunc).to(CONFIG.device)

            if global_step % 1_000 == 0:
                for k, v in info.items():
                    if k != "episode": continue
                    print(f"{global_step = } | {step = }")
                    print(f" - episodic return {v['r']} | mean {v['r'].mean().item(): .2f}")
                    print(f" - episodic length {v['l']} | mean {v['l'].mean().item(): .2f}")
                    writer.add_scalar("charts/episodic_return", v['r'].mean().item(), global_step)
                    writer.add_scalar("charts/episodic_length", v['l'].mean().item(), global_step)
                    break

        # 3. Bootstrap reward if not done (GAE thing)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if CONFIG.gae:
                advantages = torch.zeros_like(rewards).to(CONFIG.device)
                last_gae_lam = 0
                for t in reversed(range(CONFIG.n_steps)):
                    if t == CONFIG.n_steps - 1:
                        next_non_terminal = 1.0 - next_done
                        next_values = next_value
                    else: 
                        next_non_terminal = 1.0 - dones[t + 1]
                        next_values = values[t + 1]
                    
                    delta = rewards[t] = CONFIG.gamma * next_values * next_non_terminal - values[t]
                    advantages[t] = last_gae_lam = ( # Yeah this is correct
                        delta + CONFIG.gamma * CONFIG.gae_lambda * next_non_terminal * last_gae_lam
                    )
                returns =  advantages + values 

            else:
                returns = torch.zeros_like(rewards).to(CONFIG.device)
                for t in reversed(range(CONFIG.n_steps)):
                    if t == CONFIG.n_steps - 1:
                        next_non_terminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        next_non_terminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    
                    returns[t] = rewards[t] + CONFIG.gamma * next_non_terminal * next_return
                
                advantages = returns - values

        # 4. flatten the batch
        b_obs        = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs   = logprobs.reshape(-1)
        b_actions    = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns    = returns.reshape(-1)
        b_values     = values.reshape(-1)

        # 5. Minibaches
        b_ids = np.arange(CONFIG.batch_size)
        clip_fracs = []
        for epoch in range(CONFIG.update_epochs):
            np.random.shuffle(b_ids)
            for start in range(0, CONFIG.batch_size, CONFIG.mini_batch_size):
                # 5.1 prepare baches
                end = start + CONFIG.mini_batch_size
                mb_ids = b_ids[start:end] # Mini batch indices

                # 5.2 Train beggins
                _, new_log_probs, entropy, new_values = agent.get_action_value(
                    b_obs[mb_ids], b_actions.long()[mb_ids]
                )
                log_ratio = new_log_probs - b_logprobs[mb_ids]
                ratio = log_ratio.exp()

                # Debug
                with torch.no_grad():
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fracs += [((ratio - 1.0).abs() > CONFIG.clip_coef).float().mean()]

                # 5.3 Advantage normalization
                mb_advantages = b_advantages[mb_ids]
                if CONFIG.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # 5.4 Clipping the values to the 1 +- 0.2
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CONFIG.clip_coef, 1 + CONFIG.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # 5.5 Value loss
                new_value = new_values.view(-1)
                if CONFIG.clip_vloss:
                    v_loss_unclipped = (new_value - b_returns[mb_ids]) ** 2
                    v_clipped = b_values[mb_ids] + torch.clamp(
                        new_value - b_values[mb_ids],
                        -CONFIG.clip_coef,
                        CONFIG.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_ids]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_unclipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else: # Mean square error between predicted and returns
                    v_loss = 0.5 * ((new_value - b_returns[mb_ids]) ** 2).mean()

                # 5.6 Entropy
                entropy_loss = entropy.mean()

                # 5.7 Total loss
                loss = pg_loss - CONFIG.entropy_coef * entropy_loss + v_loss * CONFIG.vf_coef
                
                # 5.8 Backpropagation
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), CONFIG.max_grad_norm)
                optimizer.step()
            # END FOR START
            # Early stop batch level
            if CONFIG.target_kl is not None and approx_kl > CONFIG.target_kl:
                break
        # END FOR 
        
        # 6 Explained variance
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            
         # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean([cf.cpu().item() for cf in clip_fracs]), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()


    end_time = time.time()
    print(f"Total time {end_time - start_time:.4f}s")


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