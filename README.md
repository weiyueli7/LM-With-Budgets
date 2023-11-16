# Revisiting-Vicuna

## Table of Contents
* [Setup](#setup)
* [Estimation](#model-estimation)
* [Train](#model-training)

## Setup
To initialize the environment and packages:
```bash
conda create -n b17
conda activate b17
pip install -r requirements.txt
```

## Model Estimation
For accessing the estimation scripts, please go to the folder [`estimations`](estimations/).

## Model training
For training the model with given config stored in [`configs`](configs)
```bash
# For default arguments
python3 train.py
# For more arguments
python3 train.py --config_path <config file> --model_name <save model name> --model_dir <path to model> --batch_size <batch size> --epochs <numebr of epoches> --gpu_id <GPU ids to use>
```