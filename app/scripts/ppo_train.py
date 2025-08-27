from src.models import train_ppo, evaluate_agent
from src.utils import set_seed, start_wandb_tensorboard
from src.config import Configuration
from src.models import train_ppo, evaluate_agent

from maikol_utils.file_utils import clear_directories, make_dirs
from maikol_utils.print_utils import print_separator

def ppo_train(CONFIG: Configuration):
    """Train a ppo AC model and manage files and folder for videos.

    After training the agent save it and evaluate it.

    Args:
        CONFIG (Configuration): Configuration
    """
    print_separator(f"TRAIN PPO {CONFIG.exp_name}", sep_type="START")

    CONFIG.batch_size = int(CONFIG.n_envs * CONFIG.n_steps) 
    CONFIG.mini_batch_size = int(CONFIG.batch_size // CONFIG.n_mini_batches) 
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