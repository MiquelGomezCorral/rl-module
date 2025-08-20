import os
import time
import argparse
from dataclasses import asdict

from torch.utils.tensorboard import SummaryWriter

from src.models import train_ppo
from src.utils import set_seed, get_device
from src.config import Configuration, args_to_config

from maikol_utils.file_utils import clear_directories


def main(CONFIG: Configuration, writer: SummaryWriter):
    train_ppo(CONFIG, writer)



def parse_args_config():
    parser = argparse.ArgumentParser(
        prog="PPO Implementation following HF RL and https://www.youtube.com/watch?v=MEt6rrxH8W4&ab_channel=Weights%26Biases tutorial",
        description="PPO implementation with customizable parameters for it's use"
    )

    # ===================== GENERAL =====================
    parser.add_argument(
        "-exp", "--exp_name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="The name of the experiment."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "-v", "--record_video", action='store_true', default=False,
        help="if toggled (-v), videos of the execution of the model will be recorded and saved."
    )
    parser.add_argument(
        "-rv", "--remove_old_video", action='store_true', default=False,
        help="if toggled (-rv), videos of the execution of the model will be recorded and saved."
    )
    

    # ===================== ENV =====================
    parser.add_argument(
        "-id", "--gym_id", type=str, default="CartPole-v1",
        help="The id of the gym environment"
    )
    parser.add_argument(
        "-n", "--n_envs", type=int, default=4,
        help="Total number of sub envs for the experiment"
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=2.5e-4,
        help="The leraning rate of the optimizer"
    )
    parser.add_argument(
        "-alr", "--anneal_lr", action='store_false', default=True,
        help="Toggle leaning rate annealing for policy and value networks. (-alr store false)"
    )
    parser.add_argument(
        "-gae", "--gae", action='store_false', default=True,
        help="Toggle use of General Advantage Estimator (GAE) for advantage computation. (-gae store false)"
    )
    parser.add_argument(
        "-gae_l", "--gae_lambda", type=float, default=0.95,
        help="The lambda for the general advantage estimator"
    )
    parser.add_argument(
        "-g", "--gamma", type=float, default=0.99,
        help="The discount factor gamma"
    )
    parser.add_argument(
        "-ts", "--total_timesteps", type=int, default=50_000,
        help="Total timesteps of the experiments"
    )
    parser.add_argument(
        "-ns", "--n_steps", type=int, default=128,
        help="The number of steps to run in each environment per polcy rollout."
    )
    parser.add_argument(
        "-b", "--n_mini_batches", type=int, default=4,
        help="The number mini batches."
    )
    parser.add_argument(
        "-ue", "--update_epochs", type=int, default=4,
        help="The K epochs to update the policy."
    )
    parser.add_argument(
        "-na", "--norm_adv", action="store_false", default=True,
        help="Toggle advantages normalization (-na store false)."
    )
    parser.add_argument(
        "-cc", "--clip_coef", type=float, default=0.2,
        help="The surrogate clipping coefficient"
    )
    parser.add_argument(
        "-cv", "--clip_vloss", action="store_false", default=True,
        help="Toggle whether or not to use a clipped loss for the value function. (na store false)."
    )
    parser.add_argument(
        "-ec", "--entropy_coef", type=float, default=0.01,
        help="Coefficient of the entropy."
    )
    parser.add_argument(
        "-vc", "--vf_coef", type=float, default=0.5,
        help="Coefficient of the value function."
    )
    parser.add_argument(
        "-mgn", "--max_grad_norm", type=float, default=0.5,
        help="The maximum norm for the gradient clipping."
    )
    parser.add_argument(
        "-tkl", "--target_kl", type=float, default=0.015,
        help="The target KL divergence threshold."
    )

    # ===================== GPU =====================
    parser.add_argument(
        "--torch_deterministic", action='store_false', default=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`"
    )
    parser.add_argument(
        "--cuda",  action='store_false', default=True,
        help="if toggled, cuda will be enabled by default"
    )

    # ===================== WANDB =====================
    parser.add_argument(
        "-tr", "--track_run",  action='store_true', default=False,
        help="Track the run at wanb."
    )
    parser.add_argument(
        "-wm", "--wandb_project_name", type=str, default="RL",
        help="Name of the wandb's project."
    )
    parser.add_argument(
        "-we", "--wandb_entity", type=str, default=None,
        help="Name of the ntity (team) of wandb's project."
    )
    args = parser.parse_args()
    CONFIG = args_to_config(args)

    CONFIG.batch_size = int(CONFIG.n_envs * CONFIG.n_steps) 
    CONFIG.mini_batch_size = int(CONFIG.batch_size // CONFIG.n_mini_batches) 
    
    return CONFIG
    

if __name__ == "__main__":
    CONFIG = parse_args_config()

    if CONFIG.remove_old_video and CONFIG.record_video:
        clear_directories(CONFIG.VIDEOS)
        
    os.makedirs(CONFIG.videos_path, exist_ok=True)

    # ===================== WANDB =====================
    run_name = f"{CONFIG.gym_id}__{CONFIG.exp_name}__{CONFIG.seed}__{int(time.time())}"
    if CONFIG.track_run:
        import wandb

        wandb.init(
            project=CONFIG.wandb_project_name,
            entity=CONFIG.wandb_entity,
            sync_tensorboard=True,          # To sync with tensorboard
            config=asdict(CONFIG),
            name=run_name,
            monitor_gym= not CONFIG.record_video,
            save_code=True,
            dir=CONFIG.wandb_path
        )

    # ===================== TensorBoard =====================
    writer = SummaryWriter(os.path.join(CONFIG.runs_path, run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(CONFIG).items()])),
    )

    set_seed(CONFIG.seed)
    CONFIG.device = get_device(CONFIG)

    main(CONFIG, writer)