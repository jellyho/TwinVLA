"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset
import scipy.spatial.transform as spt

from twinvla.datasets.rlds import make_interleaved_dataset
from twinvla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights, hz_dict
from twinvla.datasets.rlds.utils.data_utils import NormalizationType
from twinvla.datasets.hz_interpolation_utils import interpolate_action
from twinvla.model.tokenizers import FASTTokenizer

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque
import time

@dataclass
class RLDSBatchTransform:
    process_inputs_fn: Any = None
    # processor: Any = None
    # tokenizer: Any = None
    window_size: int = 1
    single_arm: bool = False
    chunk_hz: bool = True
    hz_interpolate: int = None
    interpolate_gripper: bool = False  # Flag to control gripper interpolation
    action_len: int = None
    knowledge_insulation: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name = rlds_batch["dataset_name"]
        hz = hz_dict[dataset_name.decode()]
        proprio = torch.Tensor(rlds_batch['observation']['proprio'].copy())
        
        if self.chunk_hz:
            action = torch.Tensor(np.array(rlds_batch["action"]))[:hz, :]
        else:
            action = torch.Tensor(np.array(rlds_batch["action"]))
            if self.hz_interpolate is not None:
                action = interpolate_action(
                    proprio[0:1], action, hz, self.hz_interpolate, self.action_len, self.interpolate_gripper
                )

        if isinstance(rlds_batch["task"]["language_instruction"], (bytes, bytearray)):
            language_instruction = rlds_batch["task"]["language_instruction"].decode().lower()
        elif isinstance(rlds_batch["task"]["language_instruction"], np.ndarray):
            # For RoboTwin dataset. Randomly sample one instruction from the array.
            idx = np.random.randint(0, len(rlds_batch["task"]["language_instruction"]))
            language_instruction = rlds_batch["task"]["language_instruction"][idx].decode().lower()
        else:
            language_instruction = rlds_batch["task"]["language_instruction"].lower()
        img_primary = rlds_batch["observation"]["image_primary"]
        if not self.single_arm:
            img_wrist_left = rlds_batch["observation"]["image_secondary"]
            img_wrist_right = rlds_batch["observation"]["image_wrist"]
        else:
            img_wrist = rlds_batch["observation"]["image_secondary"]

        output_dict = {}
        output_dict['dataset_name'] = dataset_name
        output_dict['action'] = action
        output_dict['proprio'] = proprio
        
        ## This process_inputs_fn should generate labels if needed
        if self.single_arm:
            inputs = self.process_inputs_fn(img_primary, img_wrist, language_instruction, action)
        else:
            inputs = self.process_inputs_fn(img_primary, img_wrist_right, img_wrist_left, language_instruction, action)
        if self.knowledge_insulation:
            # Find modal_ids 6
            input_ids = inputs['input_ids']
            labels = input_ids.clone().detach()
            labels[inputs['label_ids']==0] = IGNORE_INDEX
            inputs['labels'] = labels
        else:
            labels = None

        output_dict.update(inputs)

        return output_dict
    
@dataclass
class RLDSBatchIdentity: # Euijin Jeong, 250415
    single_arm: bool = True
    chunk_hz: bool = False
    hz_interpolate: int = 30
    interpolate_gripper: bool = False  # Flag to control gripper interpolation
    action_len: int = None
    
    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Identity function for RLDS batch to the format expected by the OpenVLA collator/models."""
        output_dict = {}
        
        dataset_name = rlds_batch["dataset_name"]
        hz = hz_dict[dataset_name.decode()] if isinstance(dataset_name, bytes) else hz_dict[dataset_name]
        proprio = torch.Tensor(rlds_batch['observation']['proprio'].copy())
        
        if self.chunk_hz:
            action = torch.Tensor(np.array(rlds_batch["action"]))[:hz, :]
        else:
            action = torch.Tensor(np.array(rlds_batch["action"]))
            if self.hz_interpolate is not None:
                action = interpolate_action(
                    proprio[0:1], action, hz, self.hz_interpolate, self.action_len, self.interpolate_gripper
                )
        
        if "image_primary" in rlds_batch["observation"]:
            img_primary = rlds_batch["observation"]["image_primary"]
            
            if not self.single_arm and "image_secondary" in rlds_batch["observation"] and "image_wrist" in rlds_batch["observation"]:
                img_wrist_left = rlds_batch["observation"]["image_secondary"]
                img_wrist_right = rlds_batch["observation"]["image_wrist"]
                output_dict["pixel_values_primary"] = torch.Tensor(np.array(img_primary))
                output_dict["pixel_values_wrist_left"] = torch.Tensor(np.array(img_wrist_left))
                output_dict["pixel_values_wrist_right"] = torch.Tensor(np.array(img_wrist_right))
            elif "image_secondary" in rlds_batch["observation"]:
                img_wrist = rlds_batch["observation"]["image_secondary"]
                output_dict["pixel_values_primary"] = torch.Tensor(np.array(img_primary))
                output_dict["pixel_values_wrist"] = torch.Tensor(np.array(img_wrist))
            else:
                output_dict["pixel_values"] = torch.Tensor(np.array(img_primary))
        
        output_dict['dataset_name'] = dataset_name
        output_dict['action'] = action
        output_dict['proprio'] = proprio
        output_dict['language_instruction'] = rlds_batch["task"]["language_instruction"].decode().lower()
        
        return output_dict

class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        batch_size: int = 8,
        use_state_input: bool = True,
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        window_size = 1,
        future_action_window_size=0,
        enable_autotune=False,
        num_parallel_calls=100,
        quantile_norm=False,
        global_normalization=False,
        dataset_statistics_path=None,
        collate_fn=None,
        single_arm=False,
        num_workers=8
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform
        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]
        if global_normalization:
            assert quantile_norm and global_normalization, "Only quantile normalization is supported for global normalization (Euijin, 250420)."

        # fmt: off
        #INFO dataset kwargs
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary", "secondary", "wrist"),
            load_depth=False,
            load_proprio=use_state_input,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.NORMAL if not quantile_norm else NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=window_size,                        # If we wanted to feed / predict more than one step
                future_action_window_size=future_action_window_size,                        # For action chunking
                skip_unlabeled=True,    
                subsample_length=None                  # Skip trajectories without language labels
            ),
            frame_transform_kwargs=dict(
                resize_size={}, 
                num_parallel_calls=16,        # They used 100!!! how?                  # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
            enable_autotune=enable_autotune,
            global_normalization=global_normalization,
            dataset_statistics_path=dataset_statistics_path,
            single_arm=single_arm
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_brightness=[0.3],
                random_contrast=[0.6, 1.4],
                random_saturation=[0.5, 1.5],
                random_hue=[0.05],
                augment_order=[
                    # "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)
        self.data_iter = self.dataset.as_numpy_iterator()

        num_workers = 8
        prefetch_batches = 4 * num_workers

        self.num_workers = num_workers
        self.prefetch_queue = deque(maxlen=prefetch_batches)
        self._executor = ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix='data_transformer')
        
        self._shutdown_event = threading.Event()
        self._producer_finished = threading.Event()

        self._producer_thread = threading.Thread(target=self._producer_loop, daemon=True)
        self._producer_thread.start()

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def batch_transform_and_collate(self, list_samples):
        transformed_list = []
        for sample in list_samples:
            transformed_list.append(self.batch_transform(sample))
        return self.collate_fn(transformed_list)

    def __iter__(self):
        while not (self._producer_finished.is_set() and len(self.prefetch_queue) == 0):
            try:
                future = self.prefetch_queue.popleft()
                yield future.result()
            except IndexError:
                time.sleep(0.01)
            except Exception as e:
                print(f"[ERROR] in batch_transform: {e}")
                continue

    def _producer_loop(self):
        try:
            sample_list = []
            while not self._shutdown_event.is_set():
                for sample in self.dataset.as_numpy_iterator():
                    if len(self.prefetch_queue) == self.prefetch_queue.maxlen:
                        time.sleep(0.1)
                        continue

                    if len(sample_list) < self.batch_size:
                        sample_list.append(sample)
                    
                    elif len(sample_list) == self.batch_size:
                        future = self._executor.submit(self.batch_transform_and_collate, sample_list)
                        self.prefetch_queue.append(future)
                        sample_list = []
        except StopIteration:
            pass
        except Exception as e:
            print(f"[ERROR] Producer thread failed: {e}")
        finally:
            self._producer_finished.set()

    def close(self):
        print("Shutting down RLDSDataset...")
        self._shutdown_event.set()
        if self._producer_thread.is_alive():
            self._producer_thread.join(timeout=5)
        self._executor.shutdown(wait=True)
        print("RLDSDataset shutdown complete.")

    def __del__(self):
        if not self._shutdown_event.is_set():
            self.close()

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")