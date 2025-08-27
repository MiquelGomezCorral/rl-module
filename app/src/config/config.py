import os
import dataclasses
from dataclasses import dataclass

import torch 
from argparse import Namespace

@dataclass 
class Configuration:
    # ================== Variables ==================
    exp_name:        str = "base_name"
    model_version:   int = None
    use_checkpoint: bool = False
    keep_last_k:     int = 2
    seed:            int = 42
    record_video:   bool = False
    remove_old_video: bool = False
    fps:             int = 30
    n_eval_episodes: int = 25

    gym_id:          str = "CartPole-v1"
    n_envs:          int = 4
    eps:           float = 1e-5 # DO NOT TOUCH! is for the algorithm, is not even in the args ;)
    learning_rate: float = 2.5e-4
    anneal_lr:      bool = True
    gae:            bool = True
    gae_lambda:    float = 0.98
    gamma:         float = 0.999
    total_timesteps: int = 1_000_000
    n_steps:         int = 128
    batch_size:      int = 512 # n_envs * num_steps. Will be initialized at 'parse_args_config'
    n_mini_batches:  int = 4   
    mini_batch_size: int = 128 # batch_size / n_mini_batches. Will be initialized at 'parse_args_config'
    update_epochs:   int = 4
    norm_adv:        bool = True
    clip_coef:      float = 0.2
    clip_vloss:      bool = True
    entropy_coef:   float = 0.01
    vf_coef:        float = 0.5
    max_grad_norm:  float = 0.5
    target_kl:      float = 0.015

    torch_deterministic: bool = True
    cuda:                bool = True
    device:      torch.device = None

    track_run:         bool = False
    wandb_project_name: str = "RL"
    wandb_entity:       str = None

    # ================== Paths ==================
    TEMP:   str = "../temp"
    VIDEOS: str = "../videos"
    MODELS: str = "../models"

    runs_path:    str = os.path.join(TEMP, "runs")
    wandb_path:   str = os.path.join(TEMP, "wandb")
    videos_path:  str = os.path.join(VIDEOS, exp_name)
    models_path:  str = os.path.join(MODELS, exp_name)
    checkpoint_path: str = os.path.join(TEMP, exp_name)

    def __post_init__(self):
        self.videos_path = os.path.join(self.VIDEOS,  self.exp_name)
        self.models_path = os.path.join(self.MODELS,  self.exp_name)
        self.checkpoint_path = os.path.join(self.TEMP,  self.exp_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cuda else "cpu")


def args_to_config(args: Namespace) -> Configuration:
    """Creates a configuration object from the args parser
    If any field is not included in the Configuration class
    i will be skiped

    Args:
        args (_type_): The arguments

    Returns:
        Configuration: The Configuration object
    """
    fields = {f.name for f in dataclasses.fields(Configuration)}
    filtered = {k: v for k, v in vars(args).items() if k in fields}
    return Configuration(**filtered)

