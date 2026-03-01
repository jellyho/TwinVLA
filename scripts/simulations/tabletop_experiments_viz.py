import os
import sys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import draccus
import imageio
import numpy as np
import tqdm

from PIL import Image
import torch

from twinvla.model.twinvla import TwinVLA
import tabletop
from dm_env import StepType as st

from robot_utils import (
    DATE_TIME,
    DATE,
    set_seed_everywhere,
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["MUJOCO_GL"] = "egl"

def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, folder=None):
    """Saves an MP4 replay of an episode."""
    if folder is None:
        rollout_dir = f"./rollouts/{DATE}"
    else:
        rollout_dir = f"./rollouts/{DATE}/{folder}/videos"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


@dataclass
class Config:
    checkpoint: Union[str, Path] = ""
    task_name: str = "aloha_dish_drainer"  # Task name
    action_space: str = "ee_6d_pos"        
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 5                   # Number of rollouts per task
    action_len: int = 16
    benchmark: bool = True
    unnorm_key: str = "ee_6d_pos"

    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    seed: int = 48  
    cfg: float = 1.0
    reverse: bool = False

@draccus.wrap()
def eval_tabletop(cfg: Config) -> None:
    """
    Evaluates the model on tabletop tasks and saves rollout videos.
    """
    set_seed_everywhere(cfg.seed)
    unnorm_key = cfg.unnorm_key

    model = TwinVLA(pretrained_path=cfg.checkpoint)
    ##################

    env = tabletop.env(cfg.task_name, cfg.action_space)
    highest_rewards = []
    episode_returns = []
    for rollout_id in tqdm.tqdm(range(cfg.num_trials_per_task)):
        ts = env.reset()
        if cfg.benchmark:
            ts = env.task.benchmark_init(env.physics, rollout_id)
        action_counter = 0
        replay_images = []
        rewards = []
        token_list = []
        with torch.inference_mode():
            while True:
                obs = ts.observation
                replay_images.append(obs['images']['back'])
                front_img = obs['images']['back']
                right_wrist_img = obs['images']['wrist_right']
                left_wrist_img = obs['images']['wrist_left']
                proprio = obs['ee_6d_pos']
                
                if action_counter == 0:
                    actions, token = model.predict_action(
                        unnorm_key=unnorm_key, 
                        instruction=obs['language_instruction'],
                        image=front_img,
                        image_wrist_r=right_wrist_img,
                        image_wrist_l=left_wrist_img,
                        proprio=proprio,
                        output_action_token=True
                    )
                    token_list.append(token)

                action = actions[action_counter]
                ts = env.step(action)
                rewards.append(ts.reward)
                action_counter += 1
                if action_counter == cfg.action_len:
                    action_counter = 0
                if ts.reward == env.task.max_reward or ts.step_type==st.LAST:
                    break

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        env_max_reward = env.task.max_reward
        
        save_rollout_video(
            replay_images, rollout_id, success=episode_highest_reward==env_max_reward, task_description=cfg.task_name, folder=f"{cfg.benchmark}-{cfg.checkpoint.split('/')[-1]}-{cfg.task_name}-chunk{cfg.action_len}-{cfg.seed}-{cfg.cfg}"
        )
        replay_images.clear()
        token_list = np.array(token_list)
        np.save(f'table_traj_{cfg.task_name}.npy', token_list)

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    # std_return = np.std(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / cfg.num_trials_per_task
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{cfg.num_trials_per_task} = {more_or_equal_r_rate*100}%\n'
    
    log_dir = Path('rollouts') / DATE / f"{cfg.benchmark}-{cfg.checkpoint.split('/')[-1]}-{cfg.task_name}-chunk{cfg.action_len}-{cfg.seed}-{cfg.cfg}"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_file = log_dir / "summary.txt"
    with summary_file.open("w") as f:
        f.write(summary_str)

if __name__ == "__main__":
    eval_tabletop()