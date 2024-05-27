#!/bin/bash

# Set default values if not provided
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"12355"}

GPUS=$1
shift

# Export these variables so they are available to the Python script
export MASTER_ADDR
export MASTER_PORT

# Print the values to confirm they are set correctly
echo "Using MASTER_ADDR=$MASTER_ADDR and MASTER_PORT=$MASTER_PORT"

# Run the Python script for distributed training

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_bigger_bigger_dist.py "$@"

#python -m torch.distributed.launch \
#    --nnodes=1 \
#    --node_rank=0 \
#    --master_addr=$MASTER_ADDR \
#    --nproc_per_node=$GPUS \
#    --master_port=$MASTER_PORT \
#    train_bigger_bigger_dist.py \
#    --world_size=$GPUS \
#    --epoch=$2 \
#    --batch_size=$3 \
#    --image_size=$4

