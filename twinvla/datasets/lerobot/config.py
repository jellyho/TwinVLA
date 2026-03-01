TabletopSimConfig = {
    'proprio' : 'observation.state.ee_6d_pos',
    'action' : 'action.ee_6d_pos',
    'language_instruction': 'task',
    'image_primary' : 'observation.images.agentview',
    'image_wrist_l' : 'observation.images.wrist_left',
    'image_wrist_r' : 'observation.images.wrist_right',
    'mask': [True] * 3 + [False] * 7 + [True] * 3 + [False] * 7 
}

LeRobotConfig = {
    'jellyho/aloha_dish_drainer' : TabletopSimConfig,
    'jellyho/aloha_handover_box' : TabletopSimConfig,
    'jellyho/aloha_lift_box' : TabletopSimConfig,
    'jellyho/aloha_shoes_table' : TabletopSimConfig,
    'euijinrnd/aloha_handover_box__lerobot' : TabletopSimConfig,
}