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
torchrun --nproc_per_node <number_of_gpu> --master_port=<port_number> train_vicuna.py --config <config_file_name>
```