#!/bin/bash

sudo apt-get update
sudo apt-get install htop dstat python3-pip
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy scikit-learn tqdm pytorch_transformers apex

cd ~
git clone https://github.com/karaschechtman/COS568-DistLM-SP26.git
cd COS568-DistLM-SP26

mkdir -p $HOME/glue_data
python3 download_glue_data.py --data_dir $HOME/glue_data

export GLUE_DIR=$HOME/glue_data
export TASK_NAME=RTE

python run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir
