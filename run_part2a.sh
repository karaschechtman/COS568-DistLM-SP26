#!/bin/bash

pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy scikit-learn tqdm pytorch_transformers apex

mkdir -p $HOME/glue_data
python3 download_glue_data.py --data_dir $HOME/glue_data

export GLUE_DIR=$HOME/glue_data
export TASK_NAME=RTE

# Please first use ifconfig to check the correct interface name and then use export GLOO_SOCKET_IFNAME=<exp if name>  to use that interface.
# https://edstem.org/us/courses/94959/discussion/7800541
export GLOO_SOCKET_IFNAME=enp5s0f0   # Replace with your interface

# Distributed training arguments
export MASTER_IP=10.10.1.1      # Add master node IP
export MASTER_PORT=12345        # Choose any unused port >1023
export WORLD_SIZE=4             # Number of nodes

# Set the local rank from the command-line argument (default to 0 if not provided)
LOCAL_RANK=0
# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --local_rank) LOCAL_RANK="$2"; shift ;;
    esac
    shift
done

# Run distributed training (Task 2a)
python3 run_glue_2a.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 1\
  --output_dir ./tmp/$TASK_NAME \
  --overwrite_output_dir \
  --world_size $WORLD_SIZE \
  --local_rank "$LOCAL_RANK" \
  --master_ip $MASTER_IP \
  --master_port $MASTER_PORT

