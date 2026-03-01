"""
configs.py

Defines per-dataset configuration (kwargs) for each dataset in Open-X Embodiment.

Configuration adopts the following structure:
    image_obs_keys:
        primary: primary external RGB
        secondary: secondary external RGB
        wrist: wrist RGB

    depth_obs_keys:
        primary: primary external depth
        secondary: secondary external depth
        wrist: wrist depth

    # Always 8-dim =>> changes based on `StateEncoding`
    state_obs_keys:
        StateEncoding.POS_EULER:    EEF XYZ (3) + Roll-Pitch-Yaw (3) + <PAD> (1) + Gripper Open/Close (1)
        StateEncoding.POS_QUAT:     EEF XYZ (3) + Quaternion (4) + Gripper Open/Close (1)
        StateEncoding.JOINT:        Joint Angles (7, <PAD> if fewer) + Gripper Open/Close (1)

    state_encoding: Type of `StateEncoding`
    action_encoding: Type of action encoding (e.g., EEF Position vs. Joint Position)
"""

from enum import IntEnum

from twinvla.datasets.rlds.oxe.utils.droid_utils import zero_action_filter


# Defines Proprioceptive State Encoding Schemes
class StateEncoding(IntEnum):
    # fmt: off
    NONE = -1               # No Proprioceptive State
    POS_EULER = 1           # EEF XYZ (3) + Roll-Pitch-Yaw (3) + <PAD> (1) + Gripper Open/Close (1)
    POS_QUAT = 2            # EEF XYZ (3) + Quaternion (4) + Gripper Open/Close (1)
    POS_R6 = 5              # EEF XYZ (3) + R6 (6) + Gripper Open/Close (1)
    POS_QUAT_BIMANUAL = 7      # EEF XYZ (2 x [ EEF XYZ (3) + Quaternion (4) + Gripper Open/Close (1) ])
    POS_R6_BIMANUAL = 6       # EEF XYZ (2 x [ EEF XYZ (3) + R6 (6) + Gripper Open/Close (1) ])
    JOINT = 3               # Joint Angles (7, <PAD> if fewer) + Gripper Open/Close (1)
    JOINT_BIMANUAL = 4      # Joint Angles (2 x [ Joint Angles (6) + Gripper Open/Close (1) ])
    # fmt: on


# Defines Action Encoding Schemes
class ActionEncoding(IntEnum):
    # fmt: off
    EEF_POS = 1             # EEF Delta XYZ (3) + Roll-Pitch-Yaw (3) + Gripper Open/Close (1)
    JOINT_POS = 2           # Joint Delta Position (6) + Gripper Open/Close (1)
    JOINT_POS_BIMANUAL = 3  # Joint Delta Position (2 x [ Joint Delta Position (6) + Gripper Open/Close (1) ])
    EEF_R6 = 4              # EEF Delta XYZ (3) + R6 (6) + Gripper Open/Close (1)
    EEF_R6_BIMANUAL = 6     # EEF XYZ (2 x [ EEF Delta XYZ (3) + R6 (6) + Gripper Open/Close (1) ])
    ABS_EEF_R6 = 5          # EEF Absolute XYZ (3) + R6 (6) + Gripper Open/Close (1)
    # fmt: on

tabletop_sim_benchmark_config = {
    "image_obs_keys": {"primary": "image", "secondary": "left_wrist_image", "wrist": "right_wrist_image"},
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    "state_obs_keys": ["state"],
    "state_encoding": StateEncoding.POS_R6_BIMANUAL,
    "action_encoding": ActionEncoding.EEF_R6_BIMANUAL,
}

anubis_benchmark_config = {
    "image_obs_keys": {"primary": "agentview_image", "secondary": "left_wrist_image", "wrist": "right_wrist_image"},
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    "state_obs_keys": ["state"],
    "state_encoding": StateEncoding.POS_R6_BIMANUAL,
    "action_encoding": ActionEncoding.EEF_R6_BIMANUAL,
}

anubis_extra_benchmark_config = {
    "image_obs_keys": {"primary": "image", "secondary": "left_wrist_image", "wrist": "right_wrist_image"},
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    "state_obs_keys": ["eef_state"],
    "state_encoding": StateEncoding.POS_R6_BIMANUAL,
    "action_encoding": ActionEncoding.EEF_R6_BIMANUAL,
}

robotwin_config = {
    "image_obs_keys": {"primary": "image", "secondary": "left_wrist_image", "wrist": "right_wrist_image"},
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    "state_obs_keys": ["eef_state"],
    "state_encoding": StateEncoding.POS_R6_BIMANUAL,
    "action_encoding": ActionEncoding.EEF_R6_BIMANUAL,
}

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

# === Individual Dataset Configs ===
OXE_DATASET_CONFIGS = {
    "anubis_pullout_wrench": anubis_extra_benchmark_config,
    "anubis_fold_towel": anubis_extra_benchmark_config,
    "anubis_put_into_pot": anubis_extra_benchmark_config,
    "anubis_brush_to_pan": anubis_benchmark_config,
    "anubis_carrot_to_bag": anubis_benchmark_config,
    "anubis_towel_kirby": anubis_benchmark_config,
    'aloha_dish_drainer': tabletop_sim_benchmark_config,
    'aloha_handover_box': tabletop_sim_benchmark_config,
    'aloha_shoes_table': tabletop_sim_benchmark_config,
    'aloha_lift_box': tabletop_sim_benchmark_config,
    'aloha_box_into_pot': tabletop_sim_benchmark_config,
    'aloha_box_into_pot_easy': tabletop_sim_benchmark_config,
    "fractal20220817_data": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["base_pose_tool_reached"],
        # "state_encoding": StateEncoding.POS_QUAT,
        "state_encoding": StateEncoding.POS_R6,
        # "action_encoding": ActionEncoding.EEF_POS,
        "action_encoding": ActionEncoding.ABS_EEF_R6, # abs_6d convertsion
    },
    "rt1": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["base_pose_tool_reached", "gripper_closed"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "kuka_filtered": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["clip_function_input/base_pose_tool_reached"],
        # "state_encoding": StateEncoding.POS_QUAT,
        # "action_encoding": ActionEncoding.EEF_POS,
        "state_encoding": StateEncoding.POS_R6,
        "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "kuka": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["clip_function_input/base_pose_tool_reached"],
        # "state_encoding": StateEncoding.POS_QUAT,
        # "action_encoding": ActionEncoding.EEF_POS,
        "state_encoding": StateEncoding.POS_R6,
        "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "bridge_oxe": {  # Version of Bridge V2 in Open X-Embodiment mixture
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "bridge_orig": {  # Original version of Bridge V2 from project website
        "image_obs_keys": {"primary": "image_0", "secondary": "image_1", "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        # "state_encoding"L StateEncoding.POS_EULER,
        # "action_encoding": ActionEncoding.EEF_POS,
        "state_encoding": StateEncoding.POS_R6,
        "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "bridge_dataset": {  # Original version of Bridge V2 from project website
        "image_obs_keys": {"primary": "image_0", "secondary": "image_1", "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", "gripper_state"],
        # "state_encoding"L StateEncoding.POS_EULER,
        # "action_encoding": ActionEncoding.EEF_POS,
        "state_encoding": StateEncoding.POS_R6,
        "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "taco_play": {
        "image_obs_keys": {
            "primary": "rgb_static",
            "secondary": "rgb_gripper",
            "wrist": None,
        },
        "depth_obs_keys": {
            "primary": "depth_static",
            "secondary": "depth_gripper",
            "wrist": None,
        },
        # "state_obs_keys": ["state_eef", None, "state_gripper"],
        # "state_encoding": StateEncoding.POS_EULER,
        # "action_encoding": ActionEncoding.EEF_POS,
        "state_obs_keys": ["state_eef", "state_gripper"],
        "state_encoding": StateEncoding.POS_R6,
        "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "jaco_play": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": "image_wrist",
            "wrist": None,
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        # "state_obs_keys": ["state_eef", None, "state_gripper"],
        # "state_encoding": StateEncoding.POS_EULER,
        # "action_encoding": ActionEncoding.EEF_POS,
        "state_obs_keys": ["state_eef"],
        "state_encoding": StateEncoding.POS_R6,
        "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "berkeley_cable_routing": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": "top_image",
            "wrist": "wrist45_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["robot_state", None],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
        # "state_obs_keys": ["robot_state", None, None, None],
        # "state_encoding": StateEncoding.JOINT,
        # "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "roboturk": {
        "image_obs_keys": {"primary": "front_rgb", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [None, None, None, None, None, None, None, None],
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "nyu_door_opening_surprising_effectiveness": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [None, None, None, None, None, None, None, None],
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "viola": {
        "image_obs_keys": {
            "primary": "agentview_rgb",
            "secondary": "eye_in_hand_rgb",
            "wrist": None,
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        # "state_obs_keys": ["joint_states", "gripper_states"],
        # "state_encoding": StateEncoding.JOINT,
        # "action_encoding": ActionEncoding.EEF_POS,
        "state_obs_keys": ["state", "gripper_states"],
        "state_encoding": StateEncoding.POS_R6,
        "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "berkeley_autolab_ur5": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": "hand_image",
            "wrist": None,
        },
        "depth_obs_keys": {"primary": "depth", "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        # "state_encoding": StateEncoding.POS_QUAT,
        # "action_encoding": ActionEncoding.EEF_POS,
        "state_encoding": StateEncoding.POS_R6,
        "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "toto": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},

        # "state_obs_keys": ["state", None],
        "state_obs_keys": ["state", None, None, None],
        "state_encoding": StateEncoding.JOINT,
        # "action_encoding": ActionEncoding.EEF_POS,
        "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "language_table": {
        "image_obs_keys": {"primary": "rgb", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["effector_translation", None, None, None, None, None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "columbia_cairlab_pusht_real": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["robot_state", None, None, None, None, None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["ee_position", "ee_orientation", None],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "nyu_rot_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "stanford_hydra_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": "wrist_image",
            "wrist": None,
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", "gripper_state"],
        # "state_encoding": StateEncoding.POS_EULER,
        # "action_encoding": ActionEncoding.EEF_POS,
        "state_encoding": StateEncoding.POS_R6,
        "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "austin_buds_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": "wrist_image",
            "wrist": None,
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        # "state_obs_keys": ["state"],
        # "state_encoding": StateEncoding.JOINT,
        # "action_encoding": ActionEncoding.EEF_POS,
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_R6,
        "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "nyu_franka_play_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": "image_additional_view",
            "wrist": None,
        },
        "depth_obs_keys": {
            "primary": "depth",
            "secondary": "depth_additional_view",
            "wrist": None,
        },
        # "state_obs_keys": ["eef_state", None, None],
        # "state_encoding": StateEncoding.POS_EULER,
        # "action_encoding": ActionEncoding.EEF_POS,
        "state_obs_keys": ["eef_state"],
        "state_encoding": StateEncoding.POS_R6,
        "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "maniskill_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {
            "primary": "depth",
            "secondary": None,
            "wrist": "wrist_depth",
        },
        "state_obs_keys": ["tcp_pose", "gripper_state"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "furniture_bench_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary":  "wrist_image",
            "wrist": None,
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        # "state_encoding": StateEncoding.POS_QUAT,
        # "action_encoding": ActionEncoding.EEF_POS,
        "state_encoding": StateEncoding.POS_R6,
        "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "highres_image",
            "secondary": None,
            "wrist": None,
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [None, None, None, None, None, None, None, None],
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "ucsd_kitchen_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["joint_state", None],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "austin_sailor_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary":  "wrist_image",
            "wrist": None,
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        # "state_encoding": StateEncoding.POS_QUAT,
        # "action_encoding": ActionEncoding.EEF_POS,
        "state_encoding": StateEncoding.POS_R6,
        "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "austin_sirius_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        # "state_encoding": StateEncoding.POS_QUAT,
        # "action_encoding": ActionEncoding.EEF_POS,
        "state_encoding": StateEncoding.POS_R6,
        "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "bc_z_filtered": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        # "state_obs_keys": [
        #     "present/xyz",
        #     "present/axis_angle",
        #     None,
        #     "present/sensed_close",
        # ],
        # "state_encoding": StateEncoding.POS_EULER,
        # "action_encoding": ActionEncoding.EEF_POS,
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_R6,
        "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "bc_z": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        # "state_obs_keys": [
        #     "present/xyz",
        #     "present/axis_angle",
        #     None,
        #     "present/sensed_close",
        # ],
        # "state_encoding": StateEncoding.POS_EULER,
        # "action_encoding": ActionEncoding.EEF_POS,
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_R6,
        "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": "image2",
            "wrist": "hand_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["end_effector_pose", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utokyo_xarm_bimanual_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["pose_r", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "robo_net": {
        "image_obs_keys": {"primary": "image", "secondary": "image1", "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_mvp_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "hand_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["pose", "gripper"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.JOINT_POS,
    },
    "berkeley_rpt_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "hand_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["joint_pos", "gripper"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.JOINT_POS,
    },
    "kaist_nonprehensile_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "stanford_mask_vit_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "tokyo_u_lsmo_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "dlr_sara_pour_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "dlr_sara_grid_clamp_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "dlr_edan_shared_control_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        # "state_encoding": StateEncoding.POS_EULER,
        # "action_encoding": ActionEncoding.EEF_POS,
        "state_encoding": StateEncoding.POS_R6,
        "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "asu_table_top_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "stanford_robocook_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image_1", "secondary": "image_2", "wrist": None},
        "depth_obs_keys": {"primary": "depth_1", "secondary": "depth_2", "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "imperialcollege_sawyer_wrist_cam": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [None, None, None, None, None, None, None, "state"],
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["joint_state", "gripper_state"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
        # "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "uiuc_d3field": {
        "image_obs_keys": {"primary": "image_1", "secondary": "image_2", "wrist": None},
        "depth_obs_keys": {"primary": "depth_1", "secondary": "depth_2", "wrist": None},
        "state_obs_keys": [None, None, None, None, None, None, None, None],
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utaustin_mutex": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": "wrist_image",
            "wrist": None,
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        # "state_encoding": StateEncoding.JOINT,
        # "action_encoding": ActionEncoding.EEF_POS,
        "state_encoding": StateEncoding.POS_R6,
        "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "berkeley_fanuc_manipulation": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": "wrist_image",
            "wrist": None,
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        # "state_obs_keys": ["joint_state", None, "gripper_state"],
        "state_obs_keys": ["end_effector_state", "gripper_state"],
        # "state_encoding": StateEncoding.JOINT,
        # "action_encoding": ActionEncoding.EEF_POS,
        "state_encoding": StateEncoding.POS_R6,
        "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "cmu_playing_with_food": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "finger_vision_1",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "cmu_play_fusion": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "cmu_stretch": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        # "state_obs_keys": ["eef_state", None, "gripper_state"],
        # "state_encoding": StateEncoding.POS_EULER,
        # "action_encoding": ActionEncoding.EEF_POS,
        "state_obs_keys": ["eef_state", "gripper_state"],
        "state_encoding": StateEncoding.POS_R6,
        "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "berkeley_gnm_recon": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_gnm_cory_hall": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_gnm_sac_son": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "droid": {
        "image_obs_keys": {
            "primary": "exterior_image_1_left",
            "secondary": "wrist_image_left",
            "wrist": None,
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio"],
        # "state_encoding": StateEncoding.POS_QUAT,
        # "action_encoding": ActionEncoding.EEF_POS,
        "state_encoding": StateEncoding.POS_R6,
        "action_encoding": ActionEncoding.ABS_EEF_R6,
        "aux_kwargs": {
            "dataset_frame_transform_kwargs": {
                "chunk_filter_fn": zero_action_filter,
            },
        },
    },
    "fmb_dataset": {
        "image_obs_keys": {
            "primary": "image_side_1",
            "secondary": "image_wrist_1",
            "wrist": None,
        },
        "depth_obs_keys": {
            "primary": "image_side_1_depth",
            "secondary": "image_wrist_1_depth",
            "wrist": None,
        },
        "state_obs_keys": ["proprio"],
        # "state_encoding": StateEncoding.POS_EULER,
        # "action_encoding": ActionEncoding.EEF_POS,
        "state_encoding": StateEncoding.POS_R6,
        "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "dobbe": {
        "image_obs_keys": {"primary": "wrist_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio"],
        # "state_encoding": StateEncoding.POS_EULER,
        # "action_encoding": ActionEncoding.EEF_POS,
        "state_encoding": StateEncoding.POS_R6,
        "action_encoding": ActionEncoding.ABS_EEF_R6,
    },
    "roboset": {
        "image_obs_keys": {
            "primary": "image_left",
            "secondary": "image_right",
            "wrist": "image_wrist",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.JOINT_POS,
    },
    "rh20t": {
        "image_obs_keys": {
            "primary": "image_front",
            "secondary": "image_side_right",
            "wrist": "image_wrist",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    ### T-DROID datasets
    "tdroid_carrot_in_bowl": {  # "put carrot in bowl" task, 50 demos @ 5 Hz control
        "image_obs_keys": {"primary": "static_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "static_depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "tdroid_pour_corn_in_pot": {  # "pour corn from red bowl into steel pot" task, 50 demos @ 5 Hz control
        "image_obs_keys": {"primary": "static_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "static_depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "tdroid_flip_pot_upright": {  # "flip pot upright" task, 10 demos @ 5 Hz control
        "image_obs_keys": {"primary": "static_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "static_depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "tdroid_move_object_onto_plate": {  # "move <object> onto plate" task, 150 demos @ 5 Hz control
        "image_obs_keys": {"primary": "static_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "static_depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "tdroid_knock_object_over": {  # "knock <object> over" task, 70 demos @ 5 Hz control
        "image_obs_keys": {"primary": "static_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "static_depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "tdroid_cover_object_with_towel": {  # "cover <object> with towel" task, 45 demos @ 5 Hz control
        "image_obs_keys": {"primary": "static_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "static_depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    ### DROID Finetuning datasets
    "droid_wipe": {
        "image_obs_keys": {"primary": "exterior_image_2_left", "secondary": None, "wrist": "wrist_image_left"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    ### ALOHA datasets
    "aloha_dough_cut_dataset": {
        "image_obs_keys": {"primary": "cam_low", "secondary": "cam_left_wrist", "wrist": "cam_right_wrist"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio_ee_abs"],
        "state_encoding": StateEncoding.POS_R6_BIMANUAL,
        "action_encoding": ActionEncoding.EEF_R6_BIMANUAL,
    },
    "aloha_drawer_dataset": {
        "image_obs_keys": {"primary": "cam_low", "secondary": "cam_left_wrist", "wrist": "cam_right_wrist"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio_ee_abs"],
        "state_encoding": StateEncoding.POS_R6_BIMANUAL,
        "action_encoding": ActionEncoding.EEF_R6_BIMANUAL,
    },
    "aloha_pen_uncap_diverse_dataset": {
        "image_obs_keys": {"primary": "cam_low", "secondary": "cam_left_wrist", "wrist": "cam_right_wrist"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio_ee_abs"],
        "state_encoding": StateEncoding.POS_R6_BIMANUAL,
        "action_encoding": ActionEncoding.EEF_R6_BIMANUAL,
    },
    "aloha_pick_place_dataset": {
        "image_obs_keys": {"primary": "cam_low", "secondary": "cam_left_wrist", "wrist": "cam_right_wrist"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio_ee_abs"],
        "state_encoding": StateEncoding.POS_R6_BIMANUAL,
        "action_encoding": ActionEncoding.EEF_R6_BIMANUAL,
    },
    "aloha_play_dataset": {
        "image_obs_keys": {"primary": "cam_low", "secondary": "cam_left_wrist", "wrist": "cam_right_wrist"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio_ee_abs"],
        "state_encoding": StateEncoding.POS_R6_BIMANUAL,
        "action_encoding": ActionEncoding.EEF_R6_BIMANUAL,
    },
    "aloha_static_dataset": {
        "image_obs_keys": {"primary": "cam_low", "secondary": "cam_left_wrist", "wrist": "cam_right_wrist"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio_ee_abs"],
        "state_encoding": StateEncoding.POS_R6_BIMANUAL,
        "action_encoding": ActionEncoding.EEF_R6_BIMANUAL,
    },
    "aloha_sushi_cut_full_dataset": {
        "image_obs_keys": {"primary": "cam_low", "secondary": "cam_left_wrist", "wrist": "cam_right_wrist"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio_ee_abs"],
        "state_encoding": StateEncoding.POS_R6_BIMANUAL,
        "action_encoding": ActionEncoding.EEF_R6_BIMANUAL,
    },
}

for tn in robotwin_task_names:
    OXE_DATASET_CONFIGS[f'robotwin_{tn}'] = robotwin_config