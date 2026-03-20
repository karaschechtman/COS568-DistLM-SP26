pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy scikit-learn tqdm pytorch_transformers apex

mkdir -p $HOME/glue_data
python3 download_glue_data.py --data_dir $HOME/glue_data