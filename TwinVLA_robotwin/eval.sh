#!/bin/bash
# srun --gres=gpu:1 -J robotwin-eval --pty bash
# conda activate RoboTwin
# watch -n 1 nvidia-smi
# conda activate RoboTwin && cd ~/workspace/RoboTwin/policy/TwinVLA
# sh eval_hokyun.sh put_bottles_dustbin demo_clean demo_clean 0 0; sh eval_hokyun.sh place_object_scale demo_randomized demo_clean 0 2

policy_name=TwinVLA # [TODO] 
saved_model_path=${1}
task_name=${2}
task_config=${3}
ckpt_setting=${4}
seed=${5}
gpu_id=${6}
# [TODO] add parameters here

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --saved_model_path ${saved_model_path}
    # [TODO] add parameters here
