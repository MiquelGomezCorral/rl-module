import os
import time
import argparse
from dataclasses import asdict

from torch.utils.tensorboard import SummaryWriter

from src.models import train_ppo
from src.utils import set_seed, get_device
from src.config import Configuration, args_to_config



def main(CONFIG: Configuration):
    train_ppo(CONFIG)



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

    # ===================== ENV =====================
    parser.add_argument(
        "-id", "--gym_id", type=str, default="CartPole-v1",
        help="The id of the gym environment"
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=2.5e-4,
        help="The leraning rate of the optimizer"
    )
    parser.add_argument(
        "-ts", "--total_timesteps", type=int, default=25_000,
        help="Total timesteps of the experiments"
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
        "-n", "--wandb_project_name", type=str, default="RL",
        help="Name of the wandb's project."
    )
    parser.add_argument(
        "-we", "--wandb_entity", type=str, default=None,
        help="Name of the ntity (team) of wandb's project."
    )
    args = parser.parse_args()
    
    return args_to_config(args)
    

if __name__ == "__main__":
    CONFIG = parse_args_config()

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

    main(CONFIG)