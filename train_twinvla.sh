#!/bin/bash

NUM_GPUS=$3
BASE_LR=0.0001
BASE_MAX_STEPS=100000
BASE_SAVE_STEPS=20000

SCALED_LR=$(python3 -c "print(${BASE_LR} * ${NUM_GPUS})")
SCALED_MAX_STEPS=$(python3 -c "print(${BASE_MAX_STEPS} // ${NUM_GPUS})")
SCALED_SAVE_STEPS=$(python3 -c "print(${BASE_SAVE_STEPS} // ${NUM_GPUS})")

# Dynamically find a free port to avoid conflicts when running multiple jobs on the same node
MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
TASK_NAME="${1##*/}"
RDZV_ID="twinvla_${TASK_NAME}_$$"
OUTPUT_DIR="checkpoints/${TASK_NAME}"    

torchrun --rdzv_id="${RDZV_ID}" --rdzv_backend=c10d --rdzv_endpoint="localhost:${MASTER_PORT}" --nnodes 1 --nproc-per-node $NUM_GPUS scripts/train_twinvla.py \
    --model_type "Eagle2_1BTwinVLA" \
    --singlevla_pretrained_path "jellyho/TwinVLA" \
    --learning_rate $SCALED_LR \
    --gradient_accumulation_steps 1 \
    --max_steps $SCALED_MAX_STEPS \
    --save_steps $SCALED_SAVE_STEPS \
    --shuffle_buffer_size 50000 \
    --batch_size $2 \
    --data_root_dir "/path/to/rlds/data" \
    --data_mix "$1" \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "TwinVLA" \
    --image_aug false \
    --freeze_vision_backbone true \
    --resume false \
    --num_workers 16