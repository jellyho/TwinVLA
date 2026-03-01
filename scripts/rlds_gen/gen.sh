#!/bin/bash

python rlds_builder_tabletop.py \
    --task_name aloha_dish_drainer \
    --data_root /data5/twinvla/tabletop-sim/hdf5 \
    --output_dir /data5/twinvla/tabletop-sim/rlds

python rlds_builder_tabletop.py \
    --task_name aloha_handover_box \
    --data_root /data5/twinvla/tabletop-sim/hdf5 \
    --output_dir /data5/twinvla/tabletop-sim/rlds

python rlds_builder_tabletop.py \
    --task_name aloha_lift_box \
    --data_root /data5/twinvla/tabletop-sim/hdf5 \
    --output_dir /data5/twinvla/tabletop-sim/rlds

python rlds_builder_tabletop.py \
    --task_name aloha_shoes_table \
    --data_root /data5/twinvla/tabletop-sim/hdf5 \
    --output_dir /data5/twinvla/tabletop-sim/rlds

python rlds_builder_tabletop.py \
    --task_name aloha_box_into_pot_easy \
    --data_root /data5/twinvla/tabletop-sim/hdf5 \
    --output_dir /data5/twinvla/tabletop-sim/rlds
