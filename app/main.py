import os
import argparse


from src.models import train_ppo, evaluate_agent
from src.utils import set_seed, get_device, start_wandb_tensorboard, load_agent
from src.config import Configuration, args_to_config

from maikol_utils.file_utils import clear_directories, make_dirs
from maikol_utils.print_utils import print_separator


def ppo_train(args: argparse.Namespace):
    # ==============================================================================
    #                                   CONFIGURATION
    # ==============================================================================
    CONFIG: Configuration = args_to_config(args)

    CONFIG.batch_size = int(CONFIG.n_envs * CONFIG.n_steps) 
    CONFIG.mini_batch_size = int(CONFIG.batch_size // CONFIG.n_mini_batches) 
    CONFIG.device = get_device(CONFIG)
    set_seed(CONFIG.seed)
    
    # ===================== FILES MANAGEMENT =====================
    if CONFIG.remove_old_video and CONFIG.record_video:
        clear_directories(CONFIG.videos_path)
    make_dirs([CONFIG.videos_path, CONFIG.checkpoint_path])
    

    # ===================== WANDB =====================
    writer = start_wandb_tensorboard(CONFIG)

    # ==============================================================================
    #                                   TRAIN EVALUATE 
    # ==============================================================================
    agent = train_ppo(CONFIG, writer)
    mean_reward, std_reward = evaluate_agent(agent, CONFIG)

    clear_directories(CONFIG.TEMP)
    print_separator("DONE!", sep_type="START")


def ppo_eval(args: argparse.Namespace):
    # ==============================================================================
    #                                   CONFIGURATION
    # ==============================================================================
    CONFIG: Configuration = args_to_config(args)

    CONFIG.device = get_device(CONFIG)
    set_seed(CONFIG.seed)
    
    # ===================== FILES MANAGEMENT =====================
    if CONFIG.remove_old_video and CONFIG.record_video:
        clear_directories(CONFIG.videos_path)
    make_dirs(CONFIG.videos_path)

    # ==============================================================================
    #                                   TRAIN EVALUATE 
    # ==============================================================================
    agent = load_agent(CONFIG)
    mean_reward, std_reward = evaluate_agent(agent, CONFIG)

    clear_directories(CONFIG.TEMP)
    print_separator("DONE!", sep_type="START")



def parse_args_config():
    parser = argparse.ArgumentParser(
        prog="PPO Implementation following HF RL Course and https://www.youtube.com/watch?v=MEt6rrxH8W4&ab_channel=Weights%26Biases tutorial",
        description="PPO implementation with customizable parameters for it's use"
    )
    subparsers = parser.add_subparsers(dest="function", required=True)

    # ======================================================================================
    #                                       PPO TRAIN
    # ======================================================================================
    # ===================== GENERAL =====================
    p_ppo_train = subparsers.add_parser("ppo-train", help="Train a ppo model")
    p_ppo_train.set_defaults(func=ppo_train)

    p_ppo_train.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p_ppo_train.add_argument(
        "-exp", "--exp_name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="The name of the experiment."
    )
    p_ppo_train.add_argument(
        "-ne", "--n_envs", type=int, default=4,
        help="Total number of sub envs for the experiment"
    )
    p_ppo_train.add_argument(
        "-ch", "--use_checkpoint", action='store_true', default=False,
        help="if toggled (-ch), try to resume the training from a saved checkpoint."
    )
    p_ppo_train.add_argument(
        "-lk", "--keep_last_k", type=int, default=2,
        help="When saving checkpoints, keep the last k agents."
    )
     

    p_ppo_train.add_argument(
        "-v", "--record_video", action='store_true', default=False,
        help="if toggled (-v), videos of the execution of the model will be recorded and saved."
    )
    p_ppo_train.add_argument(
        "-rv", "--remove_old_video", action='store_true', default=False,
        help="if toggled (-rv), videos of the execution of the model will be recorded and saved."
    )
    

    # ===================== ENV =====================
    p_ppo_train.add_argument(
        "-id", "--gym_id", type=str, default="CartPole-v1",
        help="The id of the gym environment"
    )
    
    p_ppo_train.add_argument(
        "-lr", "--learning_rate", type=float, default=1e-4,
        help="The leraning rate of the optimizer"
    )
    p_ppo_train.add_argument(
        "-alr", "--anneal_lr", action='store_false', default=True,
        help="Toggle leaning rate annealing for policy and value networks. (-alr store false)"
    )
    p_ppo_train.add_argument(
        "-gae", "--gae", action='store_false', default=True,
        help="Toggle use of General Advantage Estimator (GAE) for advantage computation. (-gae store false)"
    )
    p_ppo_train.add_argument(
        "-gae_l", "--gae_lambda", type=float, default=0.98,
        help="The lambda for the general advantage estimator"
    )
    p_ppo_train.add_argument(
        "-g", "--gamma", type=float, default=0.999,
        help="The discount factor gamma"
    )
    p_ppo_train.add_argument(
        "-ts", "--total_timesteps", type=int, default=1_000_000,
        help="Total timesteps of the experiments"
    )
    p_ppo_train.add_argument(
        "-ns", "--n_steps", type=int, default=1024,
        help="The number of steps to run in each environment per polcy rollout."
    )
    p_ppo_train.add_argument(
        "-b", "--n_mini_batches", type=int, default=16,
        help="The number mini batches."
    )
    p_ppo_train.add_argument(
        "-ue", "--update_epochs", type=int, default=16,
        help="The K epochs to update the policy."
    )
    p_ppo_train.add_argument(
        "-na", "--norm_adv", action="store_false", default=True,
        help="Toggle advantages normalization (-na store false)."
    )
    p_ppo_train.add_argument(
        "-cc", "--clip_coef", type=float, default=0.2,
        help="The surrogate clipping coefficient"
    )
    p_ppo_train.add_argument(
        "-cv", "--clip_vloss", action="store_false", default=True,
        help="Toggle whether or not to use a clipped loss for the value function. (na store false)."
    )
    p_ppo_train.add_argument(
        "-ec", "--entropy_coef", type=float, default=0.01,
        help="Coefficient of the entropy."
    )
    p_ppo_train.add_argument(
        "-vc", "--vf_coef", type=float, default=0.5,
        help="Coefficient of the value function."
    )
    p_ppo_train.add_argument(
        "-mgn", "--max_grad_norm", type=float, default=0.5,
        help="The maximum norm for the gradient clipping."
    )
    p_ppo_train.add_argument(
        "-tkl", "--target_kl", type=float, default=0.015,
        help="The target KL divergence threshold."
    )

    # ===================== GPU =====================
    p_ppo_train.add_argument(
        "--torch_deterministic", action='store_false', default=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`"
    )
    p_ppo_train.add_argument(
        "--cuda",  action='store_false', default=True,
        help="if toggled, cuda will be enabled by default"
    )

    # ===================== WANDB =====================
    p_ppo_train.add_argument(
        "-tr", "--track_run",  action='store_true', default=False,
        help="Track the run at wanb."
    )
    p_ppo_train.add_argument(
        "-wm", "--wandb_project_name", type=str, default="RL",
        help="Name of the wandb's project."
    )
    p_ppo_train.add_argument(
        "-we", "--wandb_entity", type=str, default=None,
        help="Name of the ntity (team) of wandb's project."
    )

    # ======================================================================================
    #                                       EVALUATE
    # ======================================================================================
    p_ppo_eval = subparsers.add_parser("ppo-eval", help="Evaluate a ppo model")
    p_ppo_eval.set_defaults(func=ppo_eval)

    p_ppo_eval.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p_ppo_eval.add_argument(
        "-exp", "--exp_name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="The name of the experiment."
    )

    p_ppo_eval.add_argument(
        "-id", "--gym_id", type=str, default="CartPole-v1",
        help="The id of the gym environment"
    )
    p_ppo_eval.add_argument(
        "-v", "--record_video", action='store_true', default=False,
        help="if toggled (-v), videos of the execution of the model will be recorded and saved."
    )
    p_ppo_eval.add_argument(
        "-rv", "--remove_old_video", action='store_true', default=False,
        help="if toggled (-rv), videos of the execution of the model will be recorded and saved."
    )
    

    p_ppo_eval.add_argument(
        "-ne", "--n_envs", type=int, default=4,
        help="Total number of sub envs for the experiment"
    )

    p_ppo_eval.add_argument(
        "-ep", "--n_eval_episodes", type=int, default=25,
        help="Total number episodes to evaluate the agent"
    )
    p_ppo_eval.add_argument(
        "-mv", "--model_version", type=int, default=None,
        help="The model version to load"
    )
    # ======================================================================================
    #                                       CALL
    # ======================================================================================
    args = parser.parse_args()
    args.func(args)

    

if __name__ == "__main__":
    parse_args_config()

