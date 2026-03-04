import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from twinvla.datasets.lerobot.config import LeRobotConfig

class InfiniteShuffleSampler(Sampler):
    """
    Infinite sampler that shuffles data every epoch.
    """
    def __init__(self, data_source, generator=None):
        super().__init__(data_source)
        self.data_source = data_source
        self.generator = generator

    def __iter__(self):
        n = len(self.data_source)
        # Infinite loop
        while True:
            # Generate a new random permutation every time using torch.randperm
            # yield from returns each index of the generated permutation one by one
            yield from torch.randperm(n, generator=self.generator).tolist()

    def __len__(self):
        # Formally return the length of one epoch
        return len(self.data_source)


# This needs to be customized per lerobot dataset
class LeRobotDatasetForTwinVLA(Dataset):
    def __init__(
        self,
        repo_id: str,
        preprocess_fn = None,
        future_action_window_size = 19,
    ):
        self.preprocess_fn = preprocess_fn
        self.action_chunk = future_action_window_size + 1
        self.config = LeRobotConfig[repo_id]
        self.dataset_meta = LeRobotDatasetMetadata(repo_id)
        self.dataset = LeRobotDataset(
            repo_id,
            delta_timestamps={
                self.config['action']: [t / self.dataset_meta.fps for t in range(self.action_chunk)]
            },
        )
        self.dataset_name = repo_id.split('/')[-1]
        self.dataset_statistics = self.parse_stats(self.dataset_meta.stats)

        ## prepare constants for normalization
        self.prop_low = torch.from_numpy(self.dataset_statistics[self.dataset_name]['proprio']['q01'])
        self.prop_high = torch.from_numpy(self.dataset_statistics[self.dataset_name]['proprio']['q99'])
        self.prop_mask = torch.from_numpy(self.dataset_statistics[self.dataset_name]['proprio']['mask'])
        self.act_low = torch.from_numpy(self.dataset_statistics[self.dataset_name]['action']['q01'])
        self.act_high = torch.from_numpy(self.dataset_statistics[self.dataset_name]['action']['q99'])
        self.act_mask = torch.from_numpy(self.dataset_statistics[self.dataset_name]['action']['mask'])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # 1. Get sample corresponding to idx from basic LeRobotDataset.
        sample = self.dataset[idx]
        # 2. Transform data by calling existing vla_transform function.
        transformed_sample = self.vla_transform(sample)
        return transformed_sample
    
    def parse_stats(self, stat):
        new_stat = {}
        new_stat['action'] = stat[self.config['action']]
        new_stat['action']['mask'] = np.array(self.config['mask'])
        new_stat['proprio'] = stat[self.config['proprio']]
        new_stat['proprio']['mask'] = np.array(self.config['mask'])
        return {self.dataset_name : new_stat}

    def vla_transform(self, lerobot_sample):
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        proprio = lerobot_sample[self.config['proprio']] # D
        action = lerobot_sample[self.config['action']] # L, D
        language_instruction = lerobot_sample[self.config['language_instruction']]

        # Image shape should be (1, H, W, C)
        img_primary = lerobot_sample[self.config['image_primary']].permute(1, 2, 0).unsqueeze(0) * 255.0 
        img_wrist_left = lerobot_sample[self.config['image_wrist_l']].permute(1, 2, 0).unsqueeze(0) * 255.0
        img_wrist_right = lerobot_sample[self.config['image_wrist_r']].permute(1, 2, 0).unsqueeze(0) * 255.0

        output_dict = {}
        proprio, action = self.normalize_state_and_action(proprio.unsqueeze(0), action)
        output_dict['proprio'] = proprio # 1, D
        output_dict['action'] = action # L, D

        ## This process_inputs_fn should generate labels if needed
        inputs = self.preprocess_fn(img_primary, img_wrist_right, img_wrist_left, language_instruction, action)
        output_dict.update(inputs)

        return output_dict

    def normalize_state_and_action(self, state, action):
        with torch.no_grad():
            state = torch.where(
                self.prop_mask,
                (state - self.prop_low) * 2 / (self.prop_high - self.prop_low + 1e-6) - 1,
                state
            )
            action = torch.where(
                self.act_mask,
                (action - self.act_low) * 2 / (self.act_high - self.act_low + 1e-6) - 1,
                action
            )
        return state, action