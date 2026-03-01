TabletopSimConfig = {
    'proprio' : 'observation.state',
    'action' : 'action',
    'language_instruction': 'task',
    'image_primary' : 'observation.images.agentview_image',
    'image_wrist_l' : 'observation.images.left_wrist_image',
    'image_wrist_r' : 'observation.images.right_wrist_image',
    'mask': [True] * 3 + [False] * 7 + [True] * 3 + [False] * 7 
}

LeRobotConfig = {
    'jellyho/aloha_dish_drainer' : TabletopSimConfig,
    'jellyho/aloha_handover_box' : TabletopSimConfig,
    'jellyho/aloha_lift_box' : TabletopSimConfig,
    'jellyho/aloha_shoes_table' : TabletopSimConfig,
    'euijinrnd/aloha_handover_box__lerobot' : TabletopSimConfig,
}