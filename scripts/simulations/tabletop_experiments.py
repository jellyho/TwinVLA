import os
import tempfile
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import draccus
import numpy as np
import tqdm

from PIL import Image
import torch
import wandb
import imageio
from twinvla.model.twinvla import TwinVLA
import tabletop
from dm_env import StepType as st
from robot_utils import (
    DATE_TIME,
    DATE,
    set_seed_everywhere,
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Config:
    checkpoint: Union[str, Path] = ""
    task_name: str = "aloha_dish_drainer"  # Task name
    action_space: str = "ee_6d_pos"        
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 5                   # Number of rollouts per task
    action_len: int = 20
    benchmark: bool = True
    unnorm_key: str = "ee_6d_pos"

    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    seed: int = 48  
    cfg: float = 1.0
    reverse: bool = False
    f32: bool = False
    save_video: bool = False                          # Whether to save rollout videos and upload to wandb

    wandb_entity: Optional[str] = None
    wandb_project: str = "Tabletop Evaluation"

@draccus.wrap()
def eval_tabletop(cfg: Config) -> None:
    """
    Evaluates the model on tabletop tasks.
    """
    set_seed_everywhere(cfg.seed)

    wandb_name = f"{cfg.checkpoint.split('/')[-1]}-{cfg.task_name}-cfg{cfg.cfg}-B{cfg.benchmark}"

    wandb_mode = "disabled" if cfg.wandb_entity is None else "online"
    wandb.init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        name=wandb_name,
        mode=wandb_mode
    )

    unnorm_key = cfg.unnorm_key
    if cfg.f32:
        dtype = torch.float32
    else:
        dtype = torch.bfloat16
    model = TwinVLA(pretrained_path=cfg.checkpoint, dtype=dtype)
    ##################

    env = tabletop.env(cfg.task_name, cfg.action_space)

    # Create log directory only when wandb is disabled (local saving mode)
    log_dir = None
    if cfg.wandb_entity is None:
        log_dir = Path('rollouts') / DATE / f"{cfg.benchmark}-{cfg.checkpoint.split('/')[-1]}-{cfg.task_name}-chunk{cfg.action_len}-{cfg.seed}-{cfg.cfg}"
        log_dir.mkdir(parents=True, exist_ok=True)

    highest_rewards = []
    episode_returns = []
    for rollout_id in tqdm.tqdm(range(cfg.num_trials_per_task)):
        ts = env.reset()
        if cfg.benchmark:
            ts = env.task.benchmark_init(env.physics, rollout_id)
        action_counter = 0
        replay_images = []
        rewards = []

        with torch.inference_mode():
            while True:
                obs = ts.observation
                replay_images.append(obs['images']['back'])
                front_img = obs['images']['back']
                right_wrist_img = obs['images']['wrist_right']
                left_wrist_img = obs['images']['wrist_left']
                proprio = obs['ee_6d_pos']
                
                if action_counter == 0:
                    actions = model.predict_action(
                        unnorm_key=unnorm_key, 
                        instruction=obs['language_instruction'],
                        image=front_img,
                        image_wrist_r=right_wrist_img,
                        image_wrist_l=left_wrist_img,
                        proprio=proprio
                    )

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

        success_rate = np.mean(np.array(highest_rewards) == env_max_reward)

        # Save rollout video if enabled
        if cfg.save_video and len(replay_images) > 0:
            if cfg.wandb_entity is not None:
                # Wandb is active: write to a temp file and upload to wandb
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp_path = tmp.name
                imageio.mimsave(tmp_path, replay_images, fps=30)
                wandb.log(
                    {
                        f'rollout_video': wandb.Video(tmp_path, fps=30, format="mp4"),
                        f'success_rate': success_rate,
                    },
                    step=rollout_id
                )
                os.remove(tmp_path)
            else:
                # Wandb is disabled: save video locally
                video_path = log_dir / f"rollout_{rollout_id}.mp4"
                imageio.mimsave(str(video_path), replay_images, fps=30)
        else:
            wandb.log(
                {f'success_rate': success_rate},
                step=rollout_id
            )
        replay_images.clear()

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / cfg.num_trials_per_task
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{cfg.num_trials_per_task} = {more_or_equal_r_rate*100}%\n'
    
    print(summary_str)

    if log_dir is not None:
        # Save summary locally only when wandb is disabled
        summary_file = log_dir / "summary.txt"
        with summary_file.open("w") as f:
            f.write(summary_str)

    wandb.log(
        {f'success_rate': success_rate},
        step=0
    )

if __name__ == "__main__":
    eval_tabletop()