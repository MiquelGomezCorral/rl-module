"""PPO models evaluation

Manage files to evaluate PPO models
"""

from src.models import evaluate_agent
from src.utils import load_agent, set_seed

from src.config import Configuration

from maikol_utils.file_utils import clear_directories, make_dirs
from maikol_utils.print_utils import print_separator


def ppo_eval(CONFIG: Configuration):
    """Load a trained PPO AC model to evaluate it.

    Optionally record videos of the agent.

    Args:
        CONFIG (Configuration): Configuration.
    """
    print_separator(f"EVALUATE PPO {CONFIG.exp_name}", sep_type="START")

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
