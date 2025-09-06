import time
import numpy as np
from tqdm import tqdm

from maikol_utils.print_utils import print_separator, print_color
from maikol_utils.time_tracker import print_time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.config import Configuration
from src.models import get_envs, handle_states, get_shape_from_envs
from src.utils import save_agent, load_checkpoint, save_checkpoint, get_agent_from_config


def train_ppo(CONFIG: Configuration, writer: SummaryWriter) -> None:
    """Train a PPO model for the Configuration.gym_id env

    Args:
        CONFIG (Configuration): Configuration for the training
    """
    print_separator("CONFIGURATION", sep_type="LONG")
    # ================================================================
    #                       ENV MANAGEMENT
    # ================================================================
    print(f" - Creating envs...")
    # envs = create_env(CONFIG)
    envs = get_envs(CONFIG)
    state_shape, action_shape, continuous = get_shape_from_envs(envs)
    # ================================================================
    #                           AGENT & VARS
    # ================================================================
    print(f" - Creating agent and vars...")
    # ================== AGENT ==================
    agent = get_agent_from_config(CONFIG, envs)
    optimizer = optim.Adam(agent.parameters(), lr=CONFIG.learning_rate, eps=CONFIG.eps)

    # ================== CHECK CHECKPOINT ==================
    start_update = 1
    if CONFIG.use_checkpoint:
        loaded_agent, start_update = load_checkpoint(CONFIG, optimizer) 
        if loaded_agent is not None:
            agent = loaded_agent

    print(" - Env features")
    print(
        f"   - Observation dim: {print_color(agent.state_space, color='green', print_text=False)} \n"
        f"   - Action dim:      {print_color(agent.action_space, color='green', print_text=False)} "
        f"({print_color('continuous' if continuous else 'discrete', color='green', print_text=False)})"
    )
    # ================== OTHERS ==================
    global_step = 0
    start_time  = time.time()
    next_done   = torch.zeros(CONFIG.n_envs).to(CONFIG.device)
    num_updates = CONFIG.total_timesteps // CONFIG.batch_size
    next_states = handle_states(CONFIG, envs.reset()[0]) 

    # ================== VARS ==================
    # Store setup
    states = torch.zeros((CONFIG.n_steps, CONFIG.n_envs) + state_shape).to(CONFIG.device)
    if continuous:
        actions  = torch.zeros((CONFIG.n_steps, CONFIG.n_envs) + action_shape).to(CONFIG.device)
    else: 
        actions  = torch.zeros((CONFIG.n_steps, CONFIG.n_envs)).to(CONFIG.device)
    logprobs = torch.zeros((CONFIG.n_steps, CONFIG.n_envs)).to(CONFIG.device)
    rewards  = torch.zeros((CONFIG.n_steps, CONFIG.n_envs)).to(CONFIG.device)
    dones    = torch.zeros((CONFIG.n_steps, CONFIG.n_envs)).to(CONFIG.device)
    values   = torch.zeros((CONFIG.n_steps, CONFIG.n_envs)).to(CONFIG.device)


    # ================================================================
    #                       TRAINING LOOP
    # ================================================================
    print_separator("TRAINING", sep_type="SUPER")
    print(f" - Training for {CONFIG.total_timesteps} time steps and {CONFIG.batch_size} as batch size. {num_updates} updates in total.")
    if start_update != 1:
        print(f" - Starting at update: {start_update}.")
    # Episodes?
    for update in tqdm(range(start_update, num_updates + 1)):
        # 1. Annealing the rate if config says so.
        if CONFIG.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates # 1 beginning decreases -> 0
            lr_now = frac * CONFIG.learning_rate
            optimizer.param_groups[0]["lr"] = lr_now

        # 2. Collect observations
        for step in range(CONFIG.n_steps):
            # 2.1 Updating variables
            global_step += CONFIG.n_envs
            states[step] = next_states
            dones[step] = next_done

            # 2.2 Getting the model actions / predictions
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_value(next_states)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob   
        
            # 2.3 Acting in the environment
            next_states, reward, term, trunc, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(CONFIG.device).view(-1)
            next_states = handle_states(CONFIG, next_states) 
            next_done = torch.Tensor(term | trunc).to(CONFIG.device)

            if global_step % 1_000 == 0:
                for k, v in info.items():
                    if k != "episode": continue
                    writer.add_scalar("charts/episodic_return", v['r'].mean().item(), global_step)
                    writer.add_scalar("charts/episodic_length", v['l'].mean().item(), global_step)
                    break

        # 3. Bootstrap reward if not done (GAE thing)
        with torch.no_grad():
            next_value = agent.get_value(next_states).reshape(1, -1)
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
                    
                    delta = rewards[t] + CONFIG.gamma * next_values * next_non_terminal - values[t]
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
        b_states = states.reshape((-1,) + state_shape)
        b_logprobs = logprobs.reshape(-1)
        if continuous:
            b_actions = actions.reshape((-1,) + action_shape)
        else:
            b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns    = returns.reshape(-1)
        b_values     = values.reshape(-1)

        # 5. Minibaches
        # b_ids = np.arange(CONFIG.batch_size)
        b_ids = np.arange(b_states.shape[0])
        clip_fracs = []
        for epoch in range(CONFIG.update_epochs):
            np.random.shuffle(b_ids)
            for start in range(0, CONFIG.batch_size, CONFIG.mini_batch_size):
                # 5.1 prepare baches
                end = start + CONFIG.mini_batch_size
                mb_ids = b_ids[start:end] # Mini batch indices

                # 5.2 Train begins
                b_state = b_states[mb_ids]
                b_act = b_actions[mb_ids] 
                if not agent.continuous:
                    b_act = b_act.long() # convert into integer for discrete actions

                _, new_log_probs, entropy, new_values = agent.get_action_value(b_state, b_act)

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
                    v_loss_max = torch.max(v_loss_clipped, v_loss_unclipped)
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

        # SAVE CHECKPOINT
        save_checkpoint(CONFIG, agent, optimizer, update)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean([cf.cpu().item() for cf in clip_fracs]), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    # END FOR UPDATES

    # ================================================================
    #                            DONE
    # ================================================================
    envs.close()
    writer.close()

    print_separator("RESUME", sep_type="LONG")
    save_agent(CONFIG, agent)
    tot_time = time.time() - start_time
    print_time(tot_time, prefix=" - ")


    return agent
