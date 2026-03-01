"""
data_utils.py

Additional RLDS-specific data utilities.
"""
from dataclasses import dataclass
from fileinput import filename
from typing import Callable, Dict, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
import hashlib
import json
import os
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Literal

import dlimp as dl
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from copy import deepcopy
# abs_r6 convertsion
import tensorflow_graphics.geometry.transformation as tfg_transformation
from huggingface_hub import hf_hub_download


def tree_map(fn: Callable, tree: Dict) -> Dict:
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_merge(*trees: Dict) -> Dict:
    merged = {}
    for tree in trees:
        for k, v in tree.items():
            if isinstance(v, dict):
                merged[k] = tree_merge(merged.get(k, {}), v)
            else:
                merged[k] = v
    return merged


def to_padding(tensor: tf.Tensor) -> tf.Tensor:
    if tf.debugging.is_numeric_tensor(tensor):
        return tf.zeros_like(tensor)
    elif tensor.dtype == tf.string:
        return tf.fill(tf.shape(tensor), "")
    else:
        raise ValueError(f"Cannot generate padding for tensor of type {tensor.dtype}.")


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# === State / Action Processing Primitives ===


# ruff: noqa: B023
def normalize_action_and_proprio(traj: Dict, metadata: Dict, normalization_type: NormalizationType):
    """Normalizes the action and proprio fields of a trajectory using the given metadata."""
    keys_to_normalize = {"action": "action", "proprio": "observation/proprio"}

    if normalization_type == NormalizationType.NORMAL:
        for key, traj_key in keys_to_normalize.items():
            mask = metadata[key].get("mask", tf.ones_like(metadata[key]["mean"], dtype=tf.bool))
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: tf.where(mask, (x - metadata[key]["mean"]) / (metadata[key]["std"] + 1e-8), x),
            )

        return traj

    elif normalization_type in [NormalizationType.BOUNDS, NormalizationType.BOUNDS_Q99]:
        for key, traj_key in keys_to_normalize.items():
            if normalization_type == NormalizationType.BOUNDS:
                low = metadata[key]["min"]
                high = metadata[key]["max"]
            elif normalization_type == NormalizationType.BOUNDS_Q99:
                low = metadata[key]["q01"]
                high = metadata[key]["q99"]
            mask = metadata[key].get("mask", tf.ones_like(metadata[key]["min"], dtype=tf.bool))
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: tf.where(
                    mask,
                    tf.clip_by_value(2 * (x - low) / (high - low + 1e-6) - 1, -1, 1),
                    x,
                ),
            )

            # # commented out to handle jaco_play & cmu_stretch virtual rotation generation (Euijin Jeong, 20250502)
            # # Note (Moo Jin): Map unused action dimensions (i.e., dimensions where min == max) to all 0s.
            # zeros_mask = metadata[key]["min"] == metadata[key]["max"]
            # traj = dl.transforms.selective_tree_map(
            #     traj, match=lambda k, _: k == traj_key, map_fn=lambda x: tf.where(zeros_mask, 0.0, x)
            # )

        return traj

    raise ValueError(f"Unknown Normalization Type {normalization_type}")


def binarize_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """
    Converts gripper actions from continuous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open (near 1.0) or fully closed (near 0.0). As it
    transitions between the two, it sometimes passes through a few intermediate values. We relabel those intermediate
    values based on the state that is reached _after_ those intermediate values.

    In the edge case that the trajectory ends with an intermediate value, we give up on binarizing and relabel that
    chunk of intermediate values as the last action in the trajectory.

    The `scan_fn` implements the following logic:
        new_actions = np.empty_like(actions)
        carry = actions[-1]
        for i in reversed(range(actions.shape[0])):
            if in_between_mask[i]:
                carry = carry
            else:
                carry = float(open_mask[i])
            new_actions[i] = carry
    """
    open_mask, closed_mask = actions > 0.95, actions < 0.05
    in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))
    is_open_float = tf.where(open_mask, tf.ones_like(actions), tf.zeros_like(actions)) # -1 for closed, 1 for open

    def scan_fn(carry, i):
        return tf.cond(in_between_mask[i], lambda: tf.cast(carry, tf.float32), lambda: is_open_float[i])

    return tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), actions[-1], reverse=True)


def invert_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    return -actions


def rel2abs_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """
    Converts relative gripper actions (+1 for closing, -1 for opening) to absolute actions (-1 = closed; 1 = open).

    Assumes that the first relative gripper is not redundant (i.e. close when already closed)!
    """
    # Note =>> -1 for closing, 1 for opening, 0 for no change
    opening_mask, closing_mask = actions < -0.1, actions > 0.1
    thresholded_actions = tf.where(opening_mask, 1, tf.where(closing_mask, -1, 0))

    def scan_fn(carry, i):
        return tf.cond(thresholded_actions[i] == 0, lambda: carry, lambda: thresholded_actions[i])

    # If no relative grasp, assumes open for whole trajectory
    start = -1 * thresholded_actions[tf.argmax(thresholded_actions != 0, axis=0)]
    start = tf.cond(start == 0, lambda: 1, lambda: start)

    # Note =>> -1 for closed, 1 for open
    new_actions = tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), start)
    new_actions = tf.cast(new_actions, tf.float32)

    return new_actions

def zero_to_minus_one_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """
    Converts gripper actions from 0 (closed) to -1, while 1 (open) remains unchanged.
    """
    # Note =>> -1 for closed, 1 for open
    closed_mask = tf.abs(actions) < 1e-4
    new_actions = tf.where(closed_mask, -1.0, actions)
    return new_actions

def aloha_binarize_gripper_actions(
    actions: tf.Tensor,
    initial_open_thresh: float = -0.5,
    open_thresh: float = 0.05,
    close_thresh: float = 0.05,
) -> tf.Tensor:
    """
    Convert continuous ALOHA gripper commands in the range [0, 1]
    to a binary {-1, +1} signal.

    Logic
    -----
    1. diff = actions[t + 1] − actions[t] (length T − 1)
    2. State-transition rules
       • diff ≥ open_thresh  ⇒ state = +1 ("opening" → OPEN)  
       • diff ≤ −close_thresh ⇒ state = −1 ("closing" → CLOSED)  
       • otherwise       ⇒ state unchanged
    3. Initial state is +1 if actions[0] >= initial_open_thresh, else −1.

    Parameters
    ----------
    actions : tf.Tensor
        1-D tensor of shape (T,) with values in [0, 1].
    open_thresh : float, default 0.003
        Positive delta that triggers an open event.
    close_thresh : float, default 0.003
        Negative delta that triggers a close event.
    initial_open_thresh : float, default 0.1
        Threshold for determining if initial gripper state is open.

    Returns
    -------
    tf.Tensor
        1-D tensor of shape (T,) containing −1 (closed) or +1 (open).
    """
    actions = tf.cast(actions, tf.float32)
    diffs   = actions[1:] - actions[:-1]            # Δ sequence, length T-1

    # Initial state: treat >=initial_open_thresh as open, else closed
    init_state = tf.where(actions[0] >= initial_open_thresh, 1.0, -1.0)

    def scan_fn(state, delta):
        """
        State-update rule executed sequentially for each delta.
        """
        state = tf.where(delta >=  open_thresh,  1.0,
                 tf.where(delta <= -close_thresh, -1.0, state))
        return state

    # tf.scan produces length T-1 → prepend init_state to match length T
    subsequent_states = tf.scan(scan_fn, diffs, initializer=init_state)
    return tf.concat([[init_state], subsequent_states], axis=0)


# === Bridge-V2 =>> Dataset-Specific Transform ===
def relabel_bridge_actions(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Relabels actions to use reached proprioceptive state; discards last timestep (no-action)."""
    movement_actions = traj["observation"]["state"][1:, :6] - traj["observation"]["state"][:-1, :6]
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], traj)
    traj_truncated["action"] = tf.concat([movement_actions, traj["action"][:-1, -1:]], axis=1)

    return traj_truncated


# === RLDS Dataset Initialization Utilities ===
def pprint_data_mixture(dataset_kwargs_list: List[Dict[str, Any]], dataset_weights: List[int]) -> None:
    print("\n######################################################################################")
    print(f"# Loading the following {len(dataset_kwargs_list)} datasets (incl. sampling weight):{'': >24} #")
    for dataset_kwargs, weight in zip(dataset_kwargs_list, dataset_weights):
        pad = 80 - len(dataset_kwargs["name"])
        print(f"# {dataset_kwargs['name']}: {weight:=>{pad}f} #")
    print("######################################################################################\n")


def get_dataset_statistics(
    dataset: dl.DLataset,
    hash_dependencies: Tuple[str, ...],
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Either computes the statistics of a dataset or loads them from a cache file if this function has been called before
    with the same `hash_dependencies`.

    Currently, the statistics include the min/max/mean/std of the actions and proprio as well as the number of
    transitions and trajectories in the dataset.
    """
    unique_hash = hashlib.sha256("".join(hash_dependencies).encode("utf-8"), usedforsecurity=False).hexdigest()

    # Fallback local path for when data_dir is not writable or not provided
    local_path = os.path.expanduser(os.path.join("~", ".cache", "orca", f"dataset_statistics_{unique_hash}.json"))
    if save_dir is not None:
        path = tf.io.gfile.join(save_dir, f"dataset_statistics_{unique_hash}.json")
    else:
        path = local_path

    # check if cache file exists and load
    try:
        if tf.io.gfile.exists(path):
            # print(f"Loading existing dataset statistics from {path}.")
            with tf.io.gfile.GFile(path, "r") as f:
                metadata = json.load(f)
            return metadata

        if os.path.exists(local_path):
            # print(f"Loading existing dataset statistics from {local_path}.")
            with open(local_path, "r") as f:
                metadata = json.load(f)
            return metadata
    except:
        pass

    dataset = dataset.traj_map(
        lambda traj: {
            "action": traj["action"],
            "proprio": (
                traj["observation"]["proprio"] if "proprio" in traj["observation"] else tf.zeros_like(traj["action"])
            ),
        }
    )

    cardinality = dataset.cardinality().numpy()
    if cardinality == tf.data.INFINITE_CARDINALITY:
        raise ValueError("Cannot compute dataset statistics for infinite datasets.")

    print("Computing dataset statistics. This may take a bit, but should only need to happen once.")
    actions, proprios, num_transitions, num_trajectories = [], [], 0, 0
    for traj in tqdm(dataset.iterator(), total=cardinality if cardinality != tf.data.UNKNOWN_CARDINALITY else None):
        actions.append(traj["action"])
        proprios.append(traj["proprio"])
        num_transitions += traj["action"].shape[0]
        num_trajectories += 1

    actions, proprios = np.concatenate(actions), np.concatenate(proprios)
    metadata = {
        "action": {
            "mean": actions.mean(0).tolist(),
            "std": actions.std(0).tolist(),
            "max": actions.max(0).tolist(),
            "min": actions.min(0).tolist(),
            "q01": np.quantile(actions, 0.01, axis=0).tolist(),
            "q99": np.quantile(actions, 0.99, axis=0).tolist(),
        },
        "proprio": {
            "mean": proprios.mean(0).tolist(),
            "std": proprios.std(0).tolist(),
            "max": proprios.max(0).tolist(),
            "min": proprios.min(0).tolist(),
            "q01": np.quantile(proprios, 0.01, axis=0).tolist(),
            "q99": np.quantile(proprios, 0.99, axis=0).tolist(),
        },
        "num_transitions": num_transitions,
        "num_trajectories": num_trajectories,
    }

    try:
        with tf.io.gfile.GFile(path, "w") as f:
            json.dump(metadata, f)
    except tf.errors.PermissionDeniedError:
        print(f"Could not write dataset statistics to {path}. Writing to {local_path} instead.")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w") as f:
            json.dump(metadata, f)

    return metadata


def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    dataset_statistics = deepcopy(dataset_statistics)
    out_path = f'{run_dir}/dataset_statistics.json'
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    print(f"Saved dataset statistics file at path {out_path}")


def json_to_numpy_compatible(data):
    """Converts lists back to numpy arrays in the loaded JSON data."""
    if isinstance(data, dict):
        return {k: json_to_numpy_compatible(v) for k, v in data.items()}
    elif isinstance(data, list):
        return np.array(data)  # Convert lists back to numpy arrays
    else:
        return data

def load_statistics_from_json(foldername):
    """Loads a dictionary from a JSON file, converting lists to numpy arrays."""
    if os.path.isdir(foldername):
        print(f"Loading dataset statistics from local folder {foldername}")
        with open(f'{foldername}/dataset_statistics.json', "r") as f:
            data = json.load(f)
    else:
        print(f"Loading dataset statistics from HuggingFace Hub {foldername}")
        stats_path = hf_hub_download(
            repo_id=foldername,
            filename='dataset_statistics.json'
        )
        with open(f'{stats_path}', "r") as f:
            data = json.load(f)
    return json_to_numpy_compatible(data)

def allocate_threads(n: Optional[int], weights: np.ndarray):
    """
    Allocates an integer number of threads across datasets based on weights.

    The final array sums to `n`, but each element is no less than 1. If `n` is None, then every dataset is assigned a
    value of AUTOTUNE.
    """
    if n is None:
        return np.array([tf.data.AUTOTUNE] * len(weights))

    assert np.all(weights >= 0), "Weights must be non-negative"
    assert len(weights) <= n, "Number of threads must be at least as large as length of weights"
    weights = np.array(weights) / np.sum(weights)

    allocation = np.zeros_like(weights, dtype=int)
    while True:
        # Give the remaining elements that would get less than 1 a 1
        mask = (weights * n < 1) & (weights > 0)
        if not mask.any():
            break
        n -= mask.sum()
        allocation += mask.astype(int)

        # Recompute the distribution over the remaining elements
        weights[mask] = 0
        weights = weights / weights.sum()

    # Allocate the remaining elements
    fractional, integral = np.modf(weights * n)
    allocation += integral.astype(int)
    n -= integral.sum()
    for i in np.argsort(fractional)[::-1][: int(n)]:
        allocation[i] += 1

    return allocation


from transformers.feature_extraction_utils import BatchFeature
IGNORE_INDEX = -100
@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int = 1024
    pad_token_id: int = 1
    padding_side: str = "right"

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # BatchTransform-agnostic function
        output_dict = BatchFeature()
        for key in instances[0].keys():
            output_dict[key] = []
        for instance in instances:
            for key in instance.keys():
                output_dict[key].append(instance[key])

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        for key in output_dict.keys():
            if key == 'dataset_name':
                pass
            elif key in ['action']:
                output_dict[key] = torch.stack(output_dict[key])
            elif key in ['pixel_values_primary', 'pixel_values_wrist', 'pixel_values']:
                output_dict[key] = torch.stack(output_dict[key])
            elif key == 'labels':
                output_dict[key] = pad_sequence(output_dict[key], batch_first=True, padding_value=IGNORE_INDEX)
            elif key == 'input_ids':
                output_dict[key] = pad_sequence(output_dict[key], batch_first=True, padding_value=self.pad_token_id)
            else:
                output_dict[key] = pad_sequence(output_dict[key], batch_first=True, padding_value=0)

        return output_dict


def pose_to_rotation_matrix(
    pose: tf.Tensor,
    rotation_type: Literal["QUAT", "EULER", "AXIS_ANGLE"],
    start_idx: int = 3
) -> tf.Tensor:
    """
    Converts a pose tensor with various rotation representations to a rotation matrix.
    
    Args:
        pose: Tensor of shape (T, N) containing position and rotation data
        rotation_type: Type of rotation representation in the pose tensor
            - "QUAT": [qx, qy, qz, qw] starting at index start_idx
            - "EULER": [roll, pitch, yaw] starting at index start_idx
            - "AXIS_ANGLE": [ax, ay, az] starting at index start_idx
        start_idx: Index where rotation data begins in the pose tensor
            
    Returns:
        Rotation matrix of shape (T, 3, 3)
    """
    if rotation_type == "QUAT":
        # Format: [qx, qy, qz, qw]
        quat_xyzw = pose[:, start_idx:start_idx+4]
        return tfg_transformation.rotation_matrix_3d.from_quaternion(quat_xyzw)
        
    elif rotation_type == "EULER":
        # Format: [roll, pitch, yaw]
        euler_angles = pose[:, start_idx:start_idx+3]
        
        # The TF Graphics function expects roll, pitch, yaw (xyz order)
        return tfg_transformation.rotation_matrix_3d.from_euler(angles=euler_angles)
    
    elif rotation_type == "AXIS_ANGLE":
        # Format: [ax, ay, az]
        axis_angle = pose[:, start_idx:start_idx+3]
        return axis_angle_to_rotation_matrix(axis_angle)
    
    else:
        raise ValueError(f"Unsupported rotation type: {rotation_type}. "
                        "Supported types are 'QUAT', 'EULER', and 'AXIS_ANGLE'")



def axis_angle_to_rotation_matrix(axis_angle: tf.Tensor) -> tf.Tensor:
    """
    Converts axis-angle representation to rotation matrix safely handling zero angles.
    
    Args:
        axis_angle: Tensor of shape (T, 3) containing axis * angle vectors
            
    Returns:
        Rotation matrix of shape (T, 3, 3)
    """
    # Convert axis-angle to axis and angle
    angle = tf.linalg.norm(axis_angle, axis=-1, keepdims=True)  # (T, 1)
    axis = tf.math.divide_no_nan(axis_angle, angle)             # (T, 3)

    # Handle zero-angle edge case
    epsilon = 1e-6
    is_nonzero = tf.squeeze(angle, axis=-1) > epsilon           # (T,)
    safe_axis = tf.where(
        is_nonzero[:, None],
        axis,
        tf.constant([1.0, 0.0, 0.0], dtype=axis.dtype)          # use arbitrary axis if angle ≈ 0
    )

    return tfg_transformation.rotation_matrix_3d.from_axis_angle(
        axis=safe_axis, angle=angle
    )

def proprio_to_abs_action(traj: Dict[str, Any], proprio_key: str, proprio_rotation_type: Literal["QUAT", "EULER", "AXIS_ANGLE"]) -> Dict[str, Any]:
    """Use next step proprioceptive state as action."""
    rotation_length = 4 if proprio_rotation_type == "QUAT" else 3
    new_action = tf.concat([traj["observation"][proprio_key][1:, :3+rotation_length], traj["action"][:-1, -1:]], axis=1)
    
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], traj)
    traj_truncated["action"] = new_action
    return traj_truncated


def convert_to_6d_rotation(rotation, rotation_type: Literal["QUAT", "EULER", "AXIS_ANGLE"]) -> tf.Tensor:
    """
    Converts various rotation representations to 6D rotation representation.
    
    Args:
        rotation: A tensor containing rotation data in the specified format.
            If rotation_type is:
            - "QUAT": tensor of shape (..., 4) with format [qx, qy, qz, qw]
            - "EULER": tensor of shape (..., 3) with format [roll, pitch, yaw]
            - "AXIS_ANGLE": tensor of shape (..., 3) with format [ax, ay, az]
        rotation_type: The type of rotation representation in the input tensor.
            
    Returns:
        Tensor of shape (..., 6) containing the 6D rotation representation
        (first two columns of the rotation matrix flattened).
    """
    # Convert to rotation matrix based on input type
    if rotation_type == "QUAT":
        # For quaternions [qx, qy, qz, qw]
        rotation_matrix = tfg_transformation.rotation_matrix_3d.from_quaternion(rotation)
    
    elif rotation_type == "EULER":
        # For Euler angles [roll, pitch, yaw]
        rotation_matrix = tfg_transformation.rotation_matrix_3d.from_euler(angles=rotation)
    
    elif rotation_type == "AXIS_ANGLE":
        # For axis-angle representation
        # Handle zero-angle case for numerical stability
        angle = tf.linalg.norm(rotation, axis=-1, keepdims=True)  # (..., 1)
        axis = tf.math.divide_no_nan(rotation, angle)             # (..., 3)
        
        # Handle zero-angle edge case
        epsilon = 1e-6
        is_nonzero = tf.squeeze(angle, axis=-1) > epsilon           # (...,)
        safe_axis = tf.where(
            tf.expand_dims(is_nonzero, axis=-1),
            axis,
            tf.constant([1.0, 0.0, 0.0], dtype=axis.dtype)          # use arbitrary axis if angle ≈ 0
        )
        
        rotation_matrix = tfg_transformation.rotation_matrix_3d.from_axis_angle(
            axis=safe_axis, angle=angle
        )
    
    else:
        raise ValueError(f"Unsupported rotation type: {rotation_type}. "
                         "Supported types are 'QUAT', 'EULER', and 'AXIS_ANGLE'")
    
    # Extract first two columns of the rotation matrix to create 6D representation
    col1 = rotation_matrix[..., 0]  # (..., 3)
    col2 = rotation_matrix[..., 1]  # (..., 3)
    rotation_6d = tf.concat([col1, col2], axis=-1)  # (..., 6)
    
    return rotation_6d


def heterogeneous_matrix_to_pos_euler(mat_4x4):
    """ row-vector convension """
    mat_4x4 = tf.convert_to_tensor(mat_4x4, dtype=tf.float32)
    position = mat_4x4[..., 3, :3]          # (..., 3)
    rotation = mat_4x4[..., :3, :3]         # (..., 3, 3)
    euler_angles = tfg_transformation.euler.from_rotation_matrix(rotation)

    return position, euler_angles