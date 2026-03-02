"""
transforms.py

Defines a registry of per-dataset standardization transforms for each dataset in Open-X Embodiment.

Transforms adopt the following structure:
    Input: Dictionary of *batched* features (i.e., has leading time dimension)
    Output: Dictionary `step` =>> {
        "observation": {
            <image_keys, depth_image_keys>
            State (in chosen state representation)
        },
        "action": Action (in chosen action representation),
        "language_instruction": str
    }
"""

from typing import Any, Dict

import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg_transformation

from twinvla.datasets.rlds.oxe.utils.droid_utils import droid_baseact_transform_pos, droid_finetuning_transform
from twinvla.datasets.rlds.utils.data_utils import (
    binarize_gripper_actions,
    aloha_binarize_gripper_actions,
    invert_gripper_actions,
    rel2abs_gripper_actions,
    zero_to_minus_one_gripper_actions,
    relabel_bridge_actions,
    heterogeneous_matrix_to_pos_euler,
    convert_to_6d_rotation,
    proprio_to_abs_action
)

# Robotwin task names
robotwin_task_names = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle_horizontally",
    "shake_bottle",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
]


def anubis_ee_6d_pos_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"]["eef_6d_pos"]
    trajectory["observation"]["state"] = trajectory["observation"]["eef_6d_pos"]
    return trajectory

def tabletop_ee_6d_pos_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"]["eef_6d_pos"]
    trajectory["observation"]["state"] = trajectory["observation"]["eef_6d_pos"]
    return trajectory

def robotwin_ee_6d_pos_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["eef_action"]
    return trajectory

def bridge_oxe_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies to version of Bridge V2 in Open X-Embodiment mixture.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
    for key in trajectory.keys():
        if key == "traj_metadata":
            continue
        elif key in ["observation", "action"]:
            for key2 in trajectory[key]:
                trajectory[key][key2] = trajectory[key][key2][1:]
        else:
            trajectory[key] = trajectory[key][1:]

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.cast(trajectory["action"]["open_gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    trajectory = relabel_bridge_actions(trajectory)
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def bridge_orig_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies to original version of Bridge V2 from the official project website.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
    for key in trajectory.keys():
        if key == "traj_metadata":
            continue
        elif key == "observation":
            for key2 in trajectory[key]:
                trajectory[key][key2] = trajectory[key][key2][1:]
        else:
            trajectory[key] = trajectory[key][1:]

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory = relabel_bridge_actions(trajectory)
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = 2 * ((trajectory["observation"]["state"][:, -1:] - 0.0463) / (1.11 - 0.0463)) - 1
    
    # convert to absolute 6D representation
    trajectory = proprio_to_abs_action(trajectory, "EEF_state", "EULER")
    trajectory["observation"]["EEF_state"] = tf.concat(
        [
            trajectory["observation"]["EEF_state"][:, :3],
            convert_to_6d_rotation(trajectory["observation"]["EEF_state"][:, 3:6], "EULER"),
        ],
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :3],
            convert_to_6d_rotation(trajectory["action"][:, 3:6], "EULER"),
            zero_to_minus_one_gripper_actions(trajectory["action"][:, -1:]),
        ],
        axis=-1,
    )
    return trajectory


def ppgm_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["cartesian_position"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["gripper_position"][:, -1:]
    return trajectory

def rt1_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, -1 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = zero_to_minus_one_gripper_actions(rel2abs_gripper_actions(gripper_action))

    trajectory["observation"]["base_pose_tool_reached"] = tf.concat(
        [
            trajectory["observation"]["base_pose_tool_reached"],
            trajectory["observation"]["gripper_closed"][:, 0:1],
        ],
        axis=-1
    )

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]

    # convert to absolute 6D representation
    trajectory = proprio_to_abs_action(trajectory, "base_pose_tool_reached", "QUAT")
    trajectory["observation"]["base_pose_tool_reached"] = tf.concat(
        [
            trajectory["observation"]["base_pose_tool_reached"][:, :3],
            convert_to_6d_rotation(trajectory["observation"]["base_pose_tool_reached"][:, 3:7], "QUAT"),
            2 * trajectory["observation"]["base_pose_tool_reached"][:, -1:] - 1
        ],
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :3],
            convert_to_6d_rotation(trajectory["action"][:, 3:7], "QUAT"),
            trajectory["action"][:, -1:],
        ],
        axis=-1,
    )

    return trajectory


def kuka_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, -1 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    # decode compressed state
    eef_value = tf.io.decode_compressed(
        trajectory["observation"]["clip_function_input/base_pose_tool_reached"],
        compression_type="ZLIB",
    )
    eef_value = tf.io.decode_raw(eef_value, tf.float32)
    trajectory["observation"]["clip_function_input/base_pose_tool_reached"] = tf.reshape(eef_value, (-1, 7))
    gripper_value = tf.io.decode_compressed(trajectory["observation"]["gripper_closed"], compression_type="ZLIB")
    gripper_value = tf.io.decode_raw(gripper_value, tf.float32)
    trajectory["observation"]["gripper_closed"] = tf.reshape(gripper_value, (-1, 1))
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]

    # convert to absolute 6D representation
    trajectory = proprio_to_abs_action(trajectory, "clip_function_input/base_pose_tool_reached", "QUAT")
    trajectory["observation"]["clip_function_input/base_pose_tool_reached"] = tf.concat(
        [
            trajectory["observation"]["clip_function_input/base_pose_tool_reached"][:, :3],
            convert_to_6d_rotation(trajectory["observation"]["clip_function_input/base_pose_tool_reached"][:, 3:7], "QUAT"),
            2 * trajectory["observation"]["gripper_closed"] - 1,
        ],
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :3],
            convert_to_6d_rotation(trajectory["action"][:, 3:7], "QUAT"),
            trajectory["action"][:, -1:],
        ],
        axis=-1,
    )
    return trajectory


def taco_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state_eef"] = trajectory["observation"]["robot_obs"][:, :6]
    trajectory["observation"]["state_gripper"] = 2 * (trajectory["observation"]["robot_obs"][:, 6:7] / 0.08) - 1
    trajectory["action"] = trajectory["action"]["rel_actions_world"]
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )

    # convert to absolute 6D representation
    trajectory = proprio_to_abs_action(trajectory, 'state_eef', 'EULER')
    trajectory["observation"]["state_eef"] = tf.concat(
        [
            trajectory["observation"]["state_eef"][:, :3],
            convert_to_6d_rotation(trajectory["observation"]["state_eef"][:, 3:6], "EULER"),
        ],
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :3],
            convert_to_6d_rotation(trajectory["action"][:, 3:6], "EULER"),
            trajectory["action"][:, -1:],
        ],
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory



def jaco_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state_eef"] = trajectory["observation"]["end_effector_cartesian_pos"][:, :6]

    # make gripper action absolute action, +1 = open, 0 = close
    relative_gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(relative_gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            tf.zeros_like(trajectory["action"]["world_vector"]),
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]

    # convert to absolute 6D representation
    trajectory = proprio_to_abs_action(trajectory, "state_eef", "QUAT")
    
    # Use specific rotation values instead of zeros
    batch_size = tf.shape(trajectory["observation"]["state_eef"])[0]
    default_rotation = tf.tile([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]], [batch_size, 1])
    
    trajectory["observation"]["state_eef"] = tf.concat(
        [
            trajectory["observation"]["state_eef"][:, :3],
            default_rotation,
            relative_gripper_action[:-1, None],
        ],
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :3],
            default_rotation,
            trajectory["action"][:, -1:],
        ],
        axis=-1,
    )
    return trajectory



def berkeley_cable_routing_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # NOTE: still relative action since prioception is joint.
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            # rotation_matrix_to_6d(tfg_transformation.rotation_matrix_3d.from_euler(trajectory["action"]["rotation_delta"])),
            tf.zeros_like(trajectory["action"]["world_vector"][:, :1]),
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def roboturk_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert absolute gripper action, +1 = open, 0 = close
    gripper_action = invert_gripper_actions(tf.clip_by_value(trajectory["action"]["gripper_closedness_action"], 0, 1))

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action,
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def nyu_door_opening_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def viola_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action, +1 = open, -1 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, None]
    # gripper_action = tf.clip_by_value(gripper_action, 0, 1)
    gripper_action = invert_gripper_actions(gripper_action)

    # code for absolute 6D representation
    pos_xyz, euler_rpy = heterogeneous_matrix_to_pos_euler(tf.reshape(trajectory["observation"]["ee_states"], [-1, 4, 4]))
    trajectory["observation"]["state"] = tf.concat((pos_xyz, euler_rpy), axis=-1)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action,
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]

    # convert to absolute 6D representation
    trajectory = proprio_to_abs_action(trajectory, "state", "EULER")
    trajectory["observation"]["state"] = tf.concat(
        [
            trajectory["observation"]["state"][:, :3],
            convert_to_6d_rotation(trajectory["observation"]["state"][:, 3:6], "EULER"),
        ],
        axis=-1,
    )
    trajectory["observation"]["gripper_states"] = 2 * ((trajectory['observation']['gripper_states'])/(0.077)) - 1
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :3],
            convert_to_6d_rotation(trajectory["action"][:, 3:6], "EULER"),
            trajectory["action"][:, -1:],
        ],
        axis=-1,
    )

    return trajectory


def berkeley_autolab_ur5_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = trajectory["observation"]["robot_state"][:, 6:14]
    trajectory["observation"]["depth"] = trajectory["observation"].pop("image_with_depth")

    # make gripper action absolute action, +1 = open, -1 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]

    # convert to absolute 6D representation
    trajectory = proprio_to_abs_action(trajectory, "state", "QUAT")
    trajectory["observation"]["state"] = tf.concat(
        [
            trajectory["observation"]["state"][:, :3],
            convert_to_6d_rotation(trajectory["observation"]["state"][:, 3:7], "QUAT"),
            2 * trajectory["observation"]["state"][:, -1:] - 1,
        ],
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :3],
            convert_to_6d_rotation(trajectory["action"][:, 3:7], "QUAT"),
            trajectory["action"][:, -1:],
        ],
        axis=-1,
    )
    return trajectory


def toto_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.cast(trajectory["action"]["open_gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def language_table_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # default to "open" gripper
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"]),
            tf.ones_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )

    # decode language instruction
    instruction_bytes = trajectory["observation"]["instruction"]
    instruction_encoded = tf.strings.unicode_encode(instruction_bytes, output_encoding="UTF-8")
    # Remove trailing padding --> convert RaggedTensor to regular Tensor.
    trajectory["language_instruction"] = tf.strings.split(instruction_encoded, "\x00")[:, :1].to_tensor()[:, 0]
    return trajectory


def pusht_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            trajectory["action"]["gripper_closedness_action"][:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def stanford_kuka_multimodal_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["depth_image"] = trajectory["observation"]["depth_image"][..., 0]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tf.zeros_like(trajectory["action"][:, :3]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def nyu_rot_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][..., :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][..., -1:]
    trajectory["action"] = trajectory["action"][..., :7]
    return trajectory

def stanford_hydra_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action, +1 = open, -1 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(zero_to_minus_one_gripper_actions(trajectory["action"][:, -1:])),
        ),
        axis=-1,
    )

    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            trajectory["observation"]["state"][:, 7:10],
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = 2 * ((trajectory["observation"]["state"][:, -3:-2]) / 0.081 ) - 1
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction

    # convert to absolute 6D representation
    trajectory = proprio_to_abs_action(trajectory, "eef_state", "EULER")
    trajectory["observation"]["eef_state"] = tf.concat(
        [
            trajectory["observation"]["eef_state"][:, :3],
            convert_to_6d_rotation(trajectory["observation"]["eef_state"][:, 3:6], "EULER"),
        ],
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :3],
            convert_to_6d_rotation(trajectory["action"][:, 3:6], "EULER"),
            trajectory["action"][:, -1:],
        ],
        axis=-1,
    )
    return trajectory



def austin_buds_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    pos_xyz, euler_rpy = heterogeneous_matrix_to_pos_euler(tf.reshape(trajectory["observation"]["state"][:, -16:], [-1, 4, 4]))
    trajectory["observation"]["state"] = tf.concat((pos_xyz, euler_rpy, trajectory["observation"]["state"][:, 7:8]), axis=-1)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction

    # convert to absolute 6D representation
    trajectory = proprio_to_abs_action(trajectory, "state", "EULER")
    trajectory["observation"]["state"] = tf.concat(
        [
            trajectory["observation"]["state"][:, :3],
            convert_to_6d_rotation(trajectory["observation"]["state"][:, 3:6], "EULER"),
            2 * (trajectory["observation"]["state"][:, -1:] / 0.0799) - 1,
        ],
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :3],
            convert_to_6d_rotation(trajectory["action"][:, 3:6], "EULER"),
            trajectory["action"][:, -1:],
        ],
        axis=-1,
    )
    return trajectory


def nyu_franka_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["depth"] = tf.cast(trajectory["observation"]["depth"][..., 0], tf.float32)
    trajectory["observation"]["depth_additional_view"] = tf.cast(
        trajectory["observation"]["depth_additional_view"][..., 0], tf.float32
    )
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, -6:]

    # clip gripper action, +1 = open, -1 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, -8:-2],
            trajectory["action"][:, -2:-1],
        ),
        axis=-1,
    )
    original_action_gripper = trajectory["action"][:, -1:]

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction

    # convert to absolute 6D representation
    trajectory = proprio_to_abs_action(trajectory, "eef_state", "EULER")
    trajectory["observation"]["eef_state"] = tf.concat(
        [
            trajectory["observation"]["eef_state"][:, :3],
            convert_to_6d_rotation(trajectory["observation"]["eef_state"][:, 3:6], "EULER"),
            original_action_gripper[:-1, :],
        ],
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :3],
            convert_to_6d_rotation(trajectory["action"][:, 3:6], "EULER"),
            trajectory["action"][:, -1:],
        ],
        axis=-1,
    )
    return trajectory


def maniskill_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][..., 7:8]
    return trajectory


def furniture_bench_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :7],
            trajectory["observation"]["state"][:, -1:],
        ),
        axis=-1,
    )

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )
    
    # convert to absolute 6D representation
    trajectory = proprio_to_abs_action(trajectory, "state", "QUAT")
    trajectory["observation"]["state"] = tf.concat(
        [
            trajectory["observation"]["state"][:, :3],
            convert_to_6d_rotation(trajectory["observation"]["state"][:, 3:7], "QUAT"),
            2 * (trajectory["observation"]["state"][:, -1:] / 0.0799) - 1,
        ],
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :3],
            convert_to_6d_rotation(trajectory["action"][:, 3:7], "QUAT"),
            trajectory["action"][:, -1:],
        ],
        axis=-1,
    )
    return trajectory


def cmu_franka_exploration_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def ucsd_kitchen_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["joint_state"] = trajectory["observation"]["state"][:, :7]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def ucsd_pick_place_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tf.zeros(shape=(tf.shape(trajectory["action"][:, :3])[0], 6)),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def austin_sailor_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, -1 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(zero_to_minus_one_gripper_actions(trajectory["action"][:, -1:])),
        ),
        axis=-1,
    )

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction

    # convert to absolute 6D representation
    trajectory = proprio_to_abs_action(trajectory, "state", "QUAT")
    trajectory["observation"]["state"] = tf.concat(
        [
            trajectory["observation"]["state"][:, :3],
            convert_to_6d_rotation(trajectory["observation"]["state"][:, 3:7], "QUAT"),
            2 * (trajectory["observation"]["state_gripper"][:, -1:] / 0.0778) - 1,
        ],
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :3],
            convert_to_6d_rotation(trajectory["action"][:, 3:7], "QUAT"),
            trajectory["action"][:, -1:],
        ],
        axis=-1,
    )
    return trajectory


def austin_sirius_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction

    # convert to absolute 6D representation
    pos_xyz, pos_euler = heterogeneous_matrix_to_pos_euler(tf.reshape(trajectory["observation"]["state_ee"][:, :], [-1, 4, 4]))
    trajectory["observation"]["state"] = tf.concat(
            [
            pos_xyz,
            pos_euler,
            2 * (trajectory["observation"]["state_gripper"] / 0.077) - 1,
            ],
            axis=-1
        )
    trajectory = proprio_to_abs_action(trajectory, "state", "EULER")
    trajectory["observation"]["state"] = tf.concat(
        [
            trajectory["observation"]["state"][:, :3],
            convert_to_6d_rotation(trajectory["observation"]["state"][:, 3:6], "EULER"),
            trajectory["observation"]["state"][:, -1:],
        ],
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :3],
            convert_to_6d_rotation(trajectory["action"][:, 3:6], "EULER"),
            trajectory["action"][:, -1:],
        ],
        axis=-1,
    )
    return trajectory


def bc_z_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["present/xyz"],
            trajectory["observation"]["present/axis_angle"],
            2 * ((trajectory["observation"]["present/sensed_close"] - 0.2) / (1.0 - 0.2)) - 1,
        ),
        axis=-1
    )
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["future/xyz_residual"][:, :3],
            trajectory["action"]["future/axis_angle_residual"][:, :3],
            invert_gripper_actions(zero_to_minus_one_gripper_actions(tf.cast(trajectory["action"]["future/target_close"][:, :1], tf.float32))),
        ),
        axis=-1,
    )
    # convert to absolute 6D representation
    trajectory = proprio_to_abs_action(trajectory, "state", "AXIS_ANGLE")
    trajectory["observation"]["state"] = tf.concat(
        [
            trajectory["observation"]["state"][:, :3],
            convert_to_6d_rotation(trajectory["observation"]["state"][:, 3:6], "AXIS_ANGLE"),
            trajectory["observation"]["state"][:, -1:],
        ],
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :3],
            convert_to_6d_rotation(trajectory["action"][:, 3:6], "AXIS_ANGLE"),
            trajectory["action"][:, -1:],
        ],
        axis=-1,
    )

    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def tokyo_pr2_opening_fridge_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def tokyo_pr2_tabletop_manipulation_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def utokyo_xarm_pick_place_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory


def utokyo_xarm_bimanual_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., -7:]
    return trajectory


def robo_net_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :4],
            tf.zeros_like(trajectory["observation"]["state"][:, :2]),
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :4],
            tf.zeros_like(trajectory["action"][:, :2]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def berkeley_mvp_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory


def berkeley_rpt_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory


def kaist_nonprehensible_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, -7:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


def stanford_mask_vit_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["end_effector_pose"][:, :4],
            tf.zeros_like(trajectory["observation"]["end_effector_pose"][:, :2]),
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["end_effector_pose"][:, -1:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :4],
            tf.zeros_like(trajectory["action"][:, :2]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def tokyo_lsmo_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def dlr_sara_pour_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory


def dlr_sara_grid_clamp_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, :6]
    return trajectory


def dlr_edan_shared_control_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(zero_to_minus_one_gripper_actions(trajectory["action"][:, -1:])),
        ),
        axis=-1,
    )
    original_action_gripper = invert_gripper_actions(trajectory["action"][:-1, -1:])

    # convert to absolute 6D representation
    trajectory = proprio_to_abs_action(trajectory, "state", "EULER")
    trajectory["observation"]["state"] = tf.concat(
        [
            trajectory["observation"]["state"][:, :3],
            convert_to_6d_rotation(trajectory["observation"]["state"][:, 3:6], "EULER"),
            original_action_gripper
        ],
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :3],
            convert_to_6d_rotation(trajectory["action"][:, 3:6], "EULER"),
            trajectory["action"][:, -1:],
        ],
        axis=-1,
    )
    return trajectory


def asu_table_top_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["ground_truth_states"]["EE"]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def robocook_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def imperial_wristcam_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def iamlab_pick_insert_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft
    # NOTE: still relative action since prioception is None.
    trajectory["observation"]["joint_state"] = trajectory["observation"]["state"][:, :7]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, 7:8]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            # rotation_matrix_to_6d(tft.rotation_matrix_3d.from_quaternion(trajectory["action"][:, 3:7])),
            trajectory["action"][:, 7:8],
        ),
        axis=-1,
    )
    return trajectory


def uiuc_d3field_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    return trajectory

def utaustin_mutex_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    pos_xyz, pos_euler = heterogeneous_matrix_to_pos_euler(tf.reshape(trajectory["observation"]["state"][:, 8:], [-1, 4, 4]))
    gripper_state = trajectory["observation"]["state"][:, 7:8]
    trajectory["observation"]["state"] = tf.concat((pos_xyz, pos_euler, gripper_state), axis=-1)

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    # convert to absolute 6D representation
    trajectory = proprio_to_abs_action(trajectory, "state", "EULER")
    trajectory["observation"]["state"] = tf.concat(
        [
            trajectory["observation"]["state"][:, :3],
            convert_to_6d_rotation(trajectory["observation"]["state"][:, 3:6], "EULER"),
            2 * (trajectory["observation"]["state"][:, -1:] / 0.07569) - 1,
        ],
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :3],
            convert_to_6d_rotation(trajectory["action"][:, 3:6], "EULER"),
            trajectory["action"][:, -1:],
        ],
        axis=-1,
    )
    return trajectory


def berkeley_fanuc_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # trajectory["observation"]["joint_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = invert_gripper_actions(zero_to_minus_one_gripper_actions(trajectory["observation"]["state"][:, 6:7]))

    # dataset does not store gripper actions, so use gripper state info, invert so +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            trajectory["observation"]["gripper_state"],
        ),
        axis=-1,
    )

    # convert to absolute 6D representation
    trajectory = proprio_to_abs_action(trajectory, "end_effector_state", "QUAT")
    trajectory["observation"]["end_effector_state"] = tf.concat(
        [
            trajectory["observation"]["end_effector_state"][:, :3],
            convert_to_6d_rotation(trajectory["observation"]["end_effector_state"][:, 3:7], "QUAT"),
        ],
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :3],
            convert_to_6d_rotation(trajectory["action"][:, 3:7], "QUAT"),
            trajectory["action"][:, -1:],
        ],
        axis=-1,
    )
    return trajectory


def cmu_playing_with_food_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def playfusion_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            trajectory["action"][:, -4:],
        ),
        axis=-1,
    )
    return trajectory


def cmu_stretch_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            tf.zeros_like(trajectory["observation"]["state"][:, :3]),
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = zero_to_minus_one_gripper_actions(tf.clip_by_value(trajectory["observation"]["state"][:, -1:], 0, 1))
    trajectory["action"] = zero_to_minus_one_gripper_actions(trajectory["action"][..., :-1])

    # convert to absolute 6D representation
    trajectory = proprio_to_abs_action(trajectory, "eef_state", "EULER")
    
    # Use specific rotation values instead of zeros
    batch_size = tf.shape(trajectory["observation"]["eef_state"])[0]
    default_rotation = tf.tile([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], [batch_size, 1])
    
    trajectory["observation"]["eef_state"] = tf.concat(
        [
            trajectory["observation"]["eef_state"][:, :3],
            default_rotation,
        ],
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :3],
            default_rotation,
            trajectory["action"][:, -1:],
        ],
        axis=-1,
    )
    return trajectory


def gnm_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["position"],
            tf.zeros_like(trajectory["observation"]["state"][:, :3]),
            trajectory["observation"]["yaw"],
        ),
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


def fmb_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["eef_pose"],
            trajectory["observation"]["state_gripper_pose"][:, None],
        ),
        axis=-1,
    )
    
    # convert to absolute 6D representation
    trajectory = proprio_to_abs_action(trajectory, "proprio", "EULER")
    trajectory["observation"]["proprio"] = tf.concat(
        [
            trajectory["observation"]["proprio"][:, :3],
            convert_to_6d_rotation(trajectory["observation"]["proprio"][:, 3:6], "EULER"),
            2 * trajectory["observation"]["proprio"][:, -1:] - 1,
        ],
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :3],
            convert_to_6d_rotation(trajectory["action"][:, 3:6], "EULER"),
            zero_to_minus_one_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=-1,
    )
    return trajectory


def dobbe_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    
    # convert to absolute 6D representation
    trajectory = proprio_to_abs_action(trajectory, "proprio", "EULER")
    trajectory["observation"]["proprio"] = tf.concat(
        [
            trajectory["observation"]["proprio"][:, :3],
            convert_to_6d_rotation(trajectory["observation"]["proprio"][:, 3:6], "EULER"),
            2 * trajectory["observation"]["proprio"][:, -1:] - 1,
        ],
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :3],
            convert_to_6d_rotation(trajectory["action"][:, 3:6], "EULER"),
            zero_to_minus_one_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=-1,
    )
    return trajectory


def roboset_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]

    # gripper action is in -1...1 --> clip to 0...1, flip
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :7],
            gripper_action,
        ),
        axis=-1,
    )
    return trajectory


def rh20t_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["tcp_base"],
            tf.cast(trajectory["action"]["gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["tcp_base"],
            trajectory["observation"]["gripper_width"][..., None],
        ),
        axis=-1,
    )
    return trajectory


def tdroid_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["cartesian_position"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["gripper_position"][:, -1:]
    return trajectory


def libero_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # gripper action is in -1 (open)...1 (close) --> clip to 0...1, flip --> +1 = open, 0 = close
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            gripper_action,
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -2:]  # 2D gripper state
    # trajectory["observation"][""]
    return trajectory

def libero_dataset_abs_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # gripper action is in -1 (open)...1 (close) --> clip to 0...1, flip --> +1 = open, 0 = close
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :-1],
            gripper_action,
        ],
        axis=1,
    )
    return trajectory

def aloha_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["observation"]["action_ee_abs"]
    trajectory["language_instruction"] = trajectory["global_instruction"]
    trajectory["observation"]["proprio_ee_abs"] = tf.concat(
        [
            trajectory["observation"]["proprio_ee_abs"][:, :9],
            2 * ((trajectory["observation"]["proprio_ee_abs"][:, 9:10] + 0.121) / (1.118 + 0.121)) - 1,
            trajectory["observation"]["proprio_ee_abs"][:, 10:19:],
            2 * ((trajectory["observation"]["proprio_ee_abs"][:, 19:] + 0.121) /  (1.118 + 0.121)) - 1,
        ],
        axis=-1,
    )

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :9],
            2 * ((trajectory["action"][:, 9:10] + 0.121) /  (1.118 + 0.121)) - 1,
            # aloha_binarize_gripper_actions(trajectory["action"][:, 9:10]),
            trajectory["action"][:, 10:19:],
            2 * ((trajectory["action"][:, 19:] + 0.121) /  (1.118 + 0.121)) - 1,
            # aloha_binarize_gripper_actions(trajectory["action"][:, 19:]),
        ],
        axis=-1,
    )
    return trajectory
    

def rdt_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_pose"] = tf.concat(
        [
            trajectory["observation"]["eef_pose"][:, :9],
            # 2 * ((trajectory["observation"]["eef_pose"][:, 9:10] + 0.18) / (4.55+0.18)) - 1,
            aloha_binarize_gripper_actions(2 * ((trajectory["observation"]["eef_pose"][:, 9:10] + 0.18) / (4.55+0.18)) - 1, open_thresh=0.15, close_thresh=0.15),

            trajectory["observation"]["eef_pose"][:, 10:19],
            # 2 * ((trajectory["observation"]["eef_pose"][:, 19:] + 0.18) / (4.55+0.18)) - 1,
            aloha_binarize_gripper_actions(2 * ((trajectory["observation"]["eef_pose"][:, 19:] + 0.18) / (4.55+0.18)) - 1, open_thresh=0.15, close_thresh=0.15),
        ],
        axis=-1,
    )
    # trajectory["action"] = trajectory["action"]["eef_action_pose"]
    trajectory["action"] = tf.concat(
        [
            trajectory["action"]["eef_action_pose"][:, :9],
            2 * (trajectory["action"]["eef_action_pose"][:, 9:10] / 8.06) - 1,
            trajectory["action"]["eef_action_pose"][:, 10:19],
            2 * (trajectory["action"]["eef_action_pose"][:, 19:] / 8.06) - 1,
        ],
        axis=-1,
    )
    return trajectory

# === Registry ===
OXE_STANDARDIZATION_TRANSFORMS = {
    "bridge_oxe": bridge_oxe_dataset_transform,
    "bridge_orig": bridge_orig_dataset_transform,
    "bridge_dataset_gresearch": bridge_orig_dataset_transform,
    "bridge_dataset": bridge_orig_dataset_transform,
    "ppgm": ppgm_dataset_transform,
    "ppgm_static": ppgm_dataset_transform,
    "ppgm_wrist": ppgm_dataset_transform,
    "fractal20220817_data_gresearch": rt1_dataset_transform,
    "fractal20220817_data": rt1_dataset_transform,
    "rt1": rt1_dataset_transform,
    "kuka_filtered_gresearch": kuka_dataset_transform,
    "kuka_filtered": kuka_dataset_transform,
    "kuka": kuka_dataset_transform,
    "taco_play_gresearch": taco_play_dataset_transform,
    "taco_play": taco_play_dataset_transform,
    "jaco_play_gresearch": jaco_play_dataset_transform,
    "jaco_play": jaco_play_dataset_transform,
    "berkeley_cable_routing_gresearch": berkeley_cable_routing_dataset_transform,
    "berkeley_cable_routing": berkeley_cable_routing_dataset_transform,
    "roboturk": roboturk_dataset_transform,
    "nyu_door_opening_surprising_effectiveness_gresearch": nyu_door_opening_dataset_transform,
    "nyu_door_opening_surprising_effectiveness": nyu_door_opening_dataset_transform,
    "viola_gresearch": viola_dataset_transform,
    "viola": viola_dataset_transform,
    "berkeley_autolab_ur5_gresearch": berkeley_autolab_ur5_dataset_transform,
    "berkeley_autolab_ur5": berkeley_autolab_ur5_dataset_transform,
    "toto": toto_dataset_transform,
    "language_table": language_table_dataset_transform,
    "columbia_cairlab_pusht_real": pusht_dataset_transform,
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": stanford_kuka_multimodal_dataset_transform,
    "nyu_rot_dataset_converted_externally_to_rlds_gresearch": nyu_rot_dataset_transform,
    "nyu_rot_dataset_converted_externally_to_rlds": nyu_rot_dataset_transform,
    "stanford_hydra_dataset_converted_externally_to_rlds_gresearch": stanford_hydra_dataset_transform,
    "stanford_hydra_dataset_converted_externally_to_rlds": stanford_hydra_dataset_transform,
    "austin_buds_dataset_converted_externally_to_rlds_gresearch": austin_buds_dataset_transform,
    "austin_buds_dataset_converted_externally_to_rlds": austin_buds_dataset_transform,
    "nyu_franka_play_dataset_converted_externally_to_rlds_gresearch": nyu_franka_play_dataset_transform,
    "nyu_franka_play_dataset_converted_externally_to_rlds": nyu_franka_play_dataset_transform,
    "maniskill_dataset_converted_externally_to_rlds": maniskill_dataset_transform,
    "furniture_bench_dataset_converted_externally_to_rlds_gresearch": furniture_bench_dataset_transform,
    "furniture_bench_dataset_converted_externally_to_rlds": furniture_bench_dataset_transform,
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": cmu_franka_exploration_dataset_transform,
    "ucsd_kitchen_dataset_converted_externally_to_rlds_gresearch": ucsd_kitchen_dataset_transform,
    "ucsd_kitchen_dataset_converted_externally_to_rlds": ucsd_kitchen_dataset_transform,
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds_gresearch": ucsd_pick_place_dataset_transform,
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": ucsd_pick_place_dataset_transform,
    "austin_sailor_dataset_converted_externally_to_rlds_gresearch": austin_sailor_dataset_transform,
    "austin_sailor_dataset_converted_externally_to_rlds": austin_sailor_dataset_transform,
    "austin_sirius_dataset_converted_externally_to_rlds_gresearch": austin_sirius_dataset_transform,
    "austin_sirius_dataset_converted_externally_to_rlds": austin_sirius_dataset_transform,
    "bc_z_filtered_gresearch": bc_z_dataset_transform,
    "bc_z_filtered": bc_z_dataset_transform,
    "bc_z": bc_z_dataset_transform,
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": tokyo_pr2_opening_fridge_dataset_transform,
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": tokyo_pr2_tabletop_manipulation_dataset_transform,
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": utokyo_xarm_pick_place_dataset_transform,
    "utokyo_xarm_bimanual_converted_externally_to_rlds": utokyo_xarm_bimanual_dataset_transform,
    "robo_net": robo_net_dataset_transform,
    "berkeley_mvp_converted_externally_to_rlds": berkeley_mvp_dataset_transform,
    "berkeley_rpt_converted_externally_to_rlds": berkeley_rpt_dataset_transform,
    "kaist_nonprehensile_converted_externally_to_rlds": kaist_nonprehensible_dataset_transform,
    "stanford_mask_vit_converted_externally_to_rlds": stanford_mask_vit_dataset_transform,
    "tokyo_u_lsmo_converted_externally_to_rlds": tokyo_lsmo_dataset_transform,
    "dlr_sara_pour_converted_externally_to_rlds": dlr_sara_pour_dataset_transform,
    "dlr_sara_grid_clamp_converted_externally_to_rlds": dlr_sara_grid_clamp_dataset_transform,
    "dlr_edan_shared_control_converted_externally_to_rlds_gresearch": dlr_edan_shared_control_dataset_transform,
    "dlr_edan_shared_control_converted_externally_to_rlds": dlr_edan_shared_control_dataset_transform,
    "asu_table_top_converted_externally_to_rlds": asu_table_top_dataset_transform,
    "stanford_robocook_converted_externally_to_rlds": robocook_dataset_transform,
    "imperialcollege_sawyer_wrist_cam": imperial_wristcam_dataset_transform,
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds_gresearch": iamlab_pick_insert_dataset_transform,
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": iamlab_pick_insert_dataset_transform,
    "uiuc_d3field": uiuc_d3field_dataset_transform,
    "utaustin_mutex_gresearch": utaustin_mutex_dataset_transform,
    "utaustin_mutex": utaustin_mutex_dataset_transform,
    "berkeley_fanuc_manipulation_gresearch": berkeley_fanuc_dataset_transform,
    "berkeley_fanuc_manipulation": berkeley_fanuc_dataset_transform,
    "cmu_playing_with_food": cmu_playing_with_food_dataset_transform,
    "cmu_play_fusion": playfusion_dataset_transform,
    "cmu_stretch_gresearch": cmu_stretch_dataset_transform,
    "cmu_stretch": cmu_stretch_dataset_transform,
    "berkeley_gnm_recon": gnm_dataset_transform,
    "berkeley_gnm_cory_hall": gnm_dataset_transform,
    "berkeley_gnm_sac_son": gnm_dataset_transform,
    "droid_gresearch": droid_baseact_transform_pos,
    "droid": droid_baseact_transform_pos,
    "fmb_gresearch": fmb_dataset_transform,
    "fmb_dataset": fmb_dataset_transform,
    "dobbe_gresearch": dobbe_dataset_transform,
    "dobbe": dobbe_dataset_transform,
    "roboset": roboset_dataset_transform,
    "rh20t": rh20t_dataset_transform,
    ### T-DROID datasets
    "tdroid_carrot_in_bowl": tdroid_dataset_transform,
    "tdroid_pour_corn_in_pot": tdroid_dataset_transform,
    "tdroid_flip_pot_upright": tdroid_dataset_transform,
    "tdroid_move_object_onto_plate": tdroid_dataset_transform,
    "tdroid_knock_object_over": tdroid_dataset_transform,
    "tdroid_cover_object_with_towel": tdroid_dataset_transform,
    ### DROID Finetuning datasets
    "droid_wipe": droid_finetuning_transform,
    ### LIBERO
    "libero_spatial_no_noops": libero_dataset_transform,
    "libero_object_no_noops": libero_dataset_transform,
    "libero_goal_no_noops": libero_dataset_transform,
    "libero_10_no_noops": libero_dataset_transform,
    "libero_suite": libero_dataset_transform,
    "libero_spatial_no_noops_abs": libero_dataset_abs_transform,
    "libero_object_no_noops_abs": libero_dataset_abs_transform,
    "libero_goal_no_noops_abs": libero_dataset_abs_transform,
    "libero_10_no_noops_abs": libero_dataset_abs_transform,
    "libero_90_no_noops_abs": libero_dataset_abs_transform,
    "libero_suite_abs": libero_dataset_abs_transform,
    ## Tabletop
    "aloha_dish_drainer": tabletop_ee_6d_pos_transform,
    "aloha_handover_box": tabletop_ee_6d_pos_transform,
    "aloha_shoes_table": tabletop_ee_6d_pos_transform,
    "aloha_lift_box": tabletop_ee_6d_pos_transform,
    "aloha_box_into_pot": tabletop_ee_6d_pos_transform,
    "aloha_box_into_pot_easy": tabletop_ee_6d_pos_transform,
    ## Aloha dataset
    "aloha_dough_cut_dataset": aloha_dataset_transform,
    "aloha_drawer_dataset": aloha_dataset_transform,
    "aloha_pen_uncap_diverse_dataset": aloha_dataset_transform,
    "aloha_pick_place_dataset": aloha_dataset_transform,
    "aloha_play_dataset": aloha_dataset_transform,
    "aloha_static_dataset": aloha_dataset_transform,
    "aloha_sushi_cut_full_dataset": aloha_dataset_transform,
    ## RDT 
    "rdt_ft_data": rdt_dataset_transform,
    "rdt_ft_data_skip_noop": rdt_dataset_transform,
    # Anubis
    "anubis_brush_to_pan": anubis_ee_6d_pos_transform,
    "anubis_carrot_to_bag": anubis_ee_6d_pos_transform,
    "anubis_towel_kirby": anubis_ee_6d_pos_transform,
    # Anubis new
    "anubis_pullout_wrench": anubis_ee_6d_pos_transform,
    "anubis_fold_towel": anubis_ee_6d_pos_transform,
    "anubis_put_into_pot": anubis_ee_6d_pos_transform,
}

for tn in robotwin_task_names:
    OXE_STANDARDIZATION_TRANSFORMS[f'robotwin_{tn}'] = robotwin_ee_6d_pos_transform