#!/bin/bash

export GLUE_DIR=$HOME/glue_data
export TASK_NAME=RTE

LOCAL_RANK=0
SOCKET_NAME=""
MASTER_IP=""
# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --local_rank) LOCAL_RANK="$2"; shift ;;
        --socket_name) SOCKET_NAME="$2"; shift ;;
        --master_ip) MASTER_IP="$2"; shift ;;
    esac
    shift
done


# Use ifconfig to check the correct interface name and then use --socket_name  to use that interface.
# https://edstem.org/us/courses/94959/discussion/7800541
export GLOO_SOCKET_IFNAME="$SOCKET_NAME"   # Set from command line

# Distributed training arguments - also use ifconfig to get name
export MASTER_IP="$MASTER_IP"      # Set from command line
export MASTER_PORT=12345        # Choose any unused port >1023
export WORLD_SIZE=4             # Number of nodes



# Run distributed training (Task 2a)
python3 run_glue_2a.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 1\
  --output_dir ./tmp/$TASK_NAME \
  --overwrite_output_dir \
  --world_size $WORLD_SIZE \
  --local_rank "$LOCAL_RANK" \
  --master_ip $MASTER_IP \
  --master_port $MASTER_PORT

