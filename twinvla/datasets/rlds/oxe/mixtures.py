"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""

from typing import Dict, List, Tuple

# fmt: off
OXE_NAMED_MIXTURES: Dict[str, List[Tuple[str, float]]] = {
    # === Bridge V2 Dataset ===
    "bridge": [
        # ("bridge_oxe", 1.0),                                    # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
    ],

    # === [Moderate-Scale] Bridge++ Mixtures ===
    "bridge_rt_1": [
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_dataset_gresearch", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("fractal20220817_data_gresearch", 0.54087122203),                          # Google RT-1 Robot Data (Large-Scale)
    ],

    # === RT-X Mixtures ===
    "rtx": [
        ("fractal20220817_data", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        ("bridge_oxe", 1.0),                                   # Version of Bridge V2 in Open-X GCP Bucket
        # ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 2.0),
        ("berkeley_cable_routing", 3.0),
        ("roboturk", 1.0),
        # ("nyu_door_opening_surprising_effectiveness", 5.0),   # Note --> only contains wrist camera images (skip?)
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 1.0),
        ("toto", 1.0),
    ],

    "rtx-remix": [
        ("berkeley_autolab_ur5", 0.06005037264283546),
        ("berkeley_cable_routing", 0.005061455385417833),
        ("bridge_oxe", 0.5047313843090732),
        ("jaco_play", 0.009778228590044982),
        ("kuka", 0.3058671122157752),
        ("roboturk", 0.02878607494967854),
        ("fractal20220817_data", 1.0),
        ("taco_play", 0.13318072993368815),
        ("toto", 0.48938038368586745),
        ("viola", 0.03840321662965635),
    ],

    "rtx_franka": [
        ("fractal20220817_data", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 2.0),
        ("berkeley_cable_routing", 3.0),
        ("roboturk", 1.0),
        # ("nyu_door_opening_surprising_effectiveness", 5.0),   # Note --> only contains wrist camera images (skip?)
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 1.0),
        ("toto", 1.0),
        ("taco_play", 1.0),
        ("berkeley_cable_routing", 1.0),
        ("viola", 1.0),
        ("toto", 1.0),
        ("stanford_hydra_dataset_converted_externally_to_rlds", 1.0),
        ("austin_buds_dataset_converted_externally_to_rlds", 3.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        ("maniskill_dataset_converted_externally_to_rlds", 0.1),
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
        ("cmu_franka_exploration_dataset_converted_externally_to_rlds", 5.0),
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
        ("berkeley_rpt_converted_externally_to_rlds", 1.0),
        ("kaist_nonprehensile_converted_externally_to_rlds", 3.0),
        ("stanford_robocook_converted_externally_to_rlds", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
        ("utaustin_mutex", 1.0),
        ("cmu_play_fusion", 1.0),
    ],

    # === Open-X Magic Soup ===
    "oxe_magic_soup": [
        ("fractal20220817_data", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 1.0),
        ("berkeley_cable_routing", 1.0),
        ("roboturk", 2.0),
        # ("nyu_door_opening_surprising_effectiveness", 1.0),   # Note --> only contains wrist camera images (skip?)
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 2.0),
        ("toto", 1.0),
        ("language_table", 0.1),
        ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
        ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
        # ("bc_z", 0.2),                                        # Note --> raw data is broken!
        ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
        # ("uiuc_d3field", 1.0),                                # Note --> raw data is broken!
        ("utaustin_mutex", 1.0),
        ("berkeley_fanuc_manipulation", 2.0),
        ("cmu_stretch", 1.0),
    ],

    # === Open-X Magic Soup++ ===
    "oxe_magic_soup_plus": [
        ("fractal20220817_data", 0.54087122203),  #              # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294), #
        ("bridge_orig", 1.0), #                                  # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0), #
        ("jaco_play", 1.0), #
        ("berkeley_cable_routing", 1.0), #
        ("roboturk", 2.0), #
        ("viola", 2.0), #
        ("berkeley_autolab_ur5", 2.0), #
        ("toto", 1.0), #
        ("language_table", 0.1), #
        ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0), #
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0), #
        ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0), #
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.1), #
        ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0), #
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0), #
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0), #
        ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0), #
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0), #
        ("utaustin_mutex", 1.0), #
        ("berkeley_fanuc_manipulation", 2.0), #
        ("cmu_stretch", 1.0), #
        ## New Datasets in MagicSoup++
        ("bc_z", 0.2),         #                                 # Note: use v0.1.0 --> later versions broken
        ("fmb", 1.0), #
        ("dobbe", 0.2),
        ("droid", 0.06), #
    ],

    # We renamed as _gresearch to avoid bugs with RLDS dataset loading
    "oxe_magic_soup_plus_minus_gresearch": [
        ("fractal20220817_data_gresearch", 1.0),                          # Google RT-1 Robot Data (Large-Scale)
        ("kuka_filtered_gresearch", 0.8341046294),
        ("bridge_dataset_gresearch", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play_gresearch", 2.0),
        ("jaco_play_gresearch", 1.0),
        # ("berkeley_cable_routing", 1.0),  # NOTE: not going to use since it doesn't contains eef info, cannot calculate absoulte eef position - Euijin Jeong, 250415
        # ("roboturk", 2.0),  # NOTE: not going to use since it doesn't contains eef info, cannot calculate absoulte eef position - Euijin Jeong, 250415
        ("viola_gresearch", 2.0),
        ("berkeley_autolab_ur5_gresearch", 2.0),
        # ("toto", 1.0),   # NOTE: not going to use since it doesn't contains eef info, cannot calculate absoulte eef position - Euijin Jeong, 250415
        # ("language_table", 0.1),
        ("stanford_hydra_dataset_converted_externally_to_rlds_gresearch", 2.0),
        ("austin_buds_dataset_converted_externally_to_rlds_gresearch", 1.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds_gresearch", 3.0),
        ("furniture_bench_dataset_converted_externally_to_rlds_gresearch", 0.1),
        # ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),   # # NOTE: not going to use since it doesn't contains eef info, cannot calculate absoulte eef position - Euijin Jeong, 250415
        ("austin_sailor_dataset_converted_externally_to_rlds_gresearch", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds_gresearch", 1.0),
        ("dlr_edan_shared_control_converted_externally_to_rlds_gresearch", 1.0),
        # ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),   # NOTE: not going to use since it doesn't contains eef info, cannot calculate absoulte eef position - Euijin Jeong, 250415
        ("utaustin_mutex_gresearch", 1.0),
        ("berkeley_fanuc_manipulation_gresearch", 2.0),
        ("cmu_stretch_gresearch", 1.0),
        ## New Datasets in MagicSoup++
        ("bc_z_filtered_gresearch", 0.2),                                          # Note: use v0.1.0 --> later versions broken
        ("fmb_gresearch", 1.0),
        ("dobbe_gresearch", 0.2),
        ("droid_gresearch", 0.06),
    ],

    "oxe_magic_soup_plus_minus": [
        ("fractal20220817_data", 1.0),                          # Google RT-1 Robot Data (Large-Scale)
        # ("kuka", 0.8341046294),#
        ("bridge_dataset", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 1.0),#
        # ("berkeley_cable_routing", 1.0),  # NOTE: not going to use since it doesn't contains eef info, cannot calculate absoulte eef position - Euijin Jeong, 250415
        # ("roboturk", 2.0),  # NOTE: not going to use since it doesn't contains eef info, cannot calculate absoulte eef position - Euijin Jeong, 250415
        ("viola", 2.0),#
        ("berkeley_autolab_ur5", 2.0),# 
        # ("toto", 1.0),   # NOTE: not going to use since it doesn't contains eef info, cannot calculate absoulte eef position - Euijin Jeong, 250415
        # ("language_table", 0.1),
        ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0), #
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.1), #
        # ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),   # # NOTE: not going to use since it doesn't contains eef info, cannot calculate absoulte eef position - Euijin Jeong, 250415
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
        ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
        # ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),   # NOTE: not going to use since it doesn't contains eef info, cannot calculate absoulte eef position - Euijin Jeong, 250415
        ("utaustin_mutex", 1.0), #
        ("berkeley_fanuc_manipulation", 2.0),
        ("cmu_stretch", 1.0),
        ## New Datasets in MagicSoup++
        ("bc_z", 0.2),                                          # Note: use v0.1.0 --> later versions broken
        # ("fmb_dataset", 1.0), #
        # ("dobbe", 0.2), #
        # ("droid", 0.06), #
    ],

    "libero_suite":[
        ("libero_spatial_no_noops", 1.0),
        ("libero_object_no_noops", 1.0),
        ("libero_goal_no_noops", 1.0),
        ("libero_10_no_noops", 1.0),
    ],

    "libero_suite_abs":[
        ("libero_spatial_no_noops_abs", 1.0),
        ("libero_object_no_noops_abs", 1.0),
        ("libero_goal_no_noops_abs", 1.0),
        ("libero_10_no_noops_abs", 1.0),
    ],

    # === T-DROID Dataset ===
    "tdroid_carrot_in_bowl": [
        ("tdroid_carrot_in_bowl", 1.0),
    ],
    "tdroid_pour_corn_in_pot": [
        ("tdroid_pour_corn_in_pot", 1.0),
    ],
    "tdroid_flip_pot_upright": [
        ("tdroid_flip_pot_upright", 1.0),
    ],
    "tdroid_move_object_onto_plate": [
        ("tdroid_move_object_onto_plate", 1.0),
    ],
    "tdroid_knock_object_over": [
        ("tdroid_knock_object_over", 1.0),
    ],
    "tdroid_cover_object_with_towel": [
        ("tdroid_cover_object_with_towel", 1.0),
    ],

    # === DROID Finetuning Datasets ===
    "droid_wipe": [
        ("droid_wipe", 1.0),
    ],

    "robotwin2": [
        ("robotwin_adjust_bottle", 1.0),
        ("robotwin_beat_block_hammer", 1.0),
        ("robotwin_blocks_ranking_rgb", 1.0),
        ("robotwin_blocks_ranking_size", 1.0),
        ("robotwin_click_alarmclock", 1.0),
        ("robotwin_click_bell", 1.0),
        ("robotwin_dump_bin_bigbin", 1.0),
        ("robotwin_grab_roller", 1.0),
        ("robotwin_handover_block", 1.0),
        ("robotwin_handover_mic", 1.0),
        ("robotwin_hanging_mug", 1.0),
        ("robotwin_lift_pot", 1.0),
        ("robotwin_move_can_pot", 1.0),
        ("robotwin_move_pillbottle_pad", 1.0),
        ("robotwin_move_playingcard_away", 1.0),
        ("robotwin_move_stapler_pad", 1.0),
        ("robotwin_open_laptop", 1.0),
        ("robotwin_open_microwave", 1.0),
        ("robotwin_pick_diverse_bottles", 1.0),
        ("robotwin_pick_dual_bottles", 1.0),
        ("robotwin_place_a2b_left", 1.0),
        ("robotwin_place_a2b_right", 1.0),
        ("robotwin_place_bread_basket", 1.0),
        ("robotwin_place_bread_skillet", 1.0),
        ("robotwin_place_burger_fries", 1.0),
        ("robotwin_place_can_basket", 1.0),
        ("robotwin_place_cans_plasticbox", 1.0),
        ("robotwin_place_container_plate", 1.0),
        ("robotwin_place_dual_shoes", 1.0),
        ("robotwin_place_empty_cup", 1.0),
        ("robotwin_place_fan", 1.0),
        ("robotwin_place_mouse_pad", 1.0),
        ("robotwin_place_object_basket", 1.0),
        ("robotwin_place_object_scale", 1.0),
        ("robotwin_place_object_stand", 1.0),
        ("robotwin_place_phone_stand", 1.0),
        ("robotwin_place_shoe", 1.0),
        ("robotwin_press_stapler", 1.0),
        ("robotwin_put_bottles_dustbin", 1.0),
        ("robotwin_put_object_cabinet", 1.0),
        ("robotwin_rotate_qrcode", 1.0),
        ("robotwin_scan_object", 1.0),
        ("robotwin_shake_bottle_horizontally", 1.0),
        ("robotwin_shake_bottle", 1.0),
        ("robotwin_stack_blocks_three", 1.0),
        ("robotwin_stack_blocks_two", 1.0),
        ("robotwin_stack_bowls_three", 1.0),
        ("robotwin_stack_bowls_two", 1.0),
        ("robotwin_stamp_seal", 1.0),
        ("robotwin_turn_switch", 1.0),
    ],
}
# fmt: on
