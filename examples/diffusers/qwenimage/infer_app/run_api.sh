#!/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=4,5

# Distributed configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${WORLD_SIZE:-2}

entry_file="pipeline.py"
model_dir="/mnt/disk4/dxw/Qwen-Image"
port=5000

msrun --worker_num=${NPROC_PER_NODE} \
    --local_worker_num=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --log_dir="logs/api" \
    --join=True \
    ${entry_file} \
    --model_dir "${model_dir}" \
    --port "${port}"
