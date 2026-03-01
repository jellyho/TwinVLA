#!/usr/bin/env bash
checkpoint=$1
task_name=$2
MUJOCO_GL=egl

python3 scripts/simulations/tabletop_experiments.py \
    --checkpoint "$checkpoint" \
    --task_name "$task_name" \
    --unnorm_key "$task_name" \
    --action_len 20 \
    --num_trials_per_task 500 \
    --seed 0 \
    --benchmark true \
    --action_space "ee_6d_pos" \
    --cfg 1.1 \
    --save_video true
    # --wandb_entity "jellyho_" \