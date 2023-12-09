# Training Language Models on a Computational Budget

Large Language Models (LLMs) have demonstrated impressive performance across various tasks, including question-answering, machine translation, and code implementation. Training these models, however, demands significant computational resources and time. In light of the limited availability of computational resources, developing a detailed budget plan has become increasingly vital in the training process, particularly in higher education settings. To address this, we have prepared a technical report that outlines our methodology for calculating model parameters, training FLOPs (floating-point operations), and memory costs. Based on these calculations and adhering to the Chinchilla scaling law, we designed three model configurations—large, medium, and small—to align with our computed computational budget. All models were trained using the SlimPajama-6B benchmark on eight NVIDIA A6000 48GB GPUs. We achieved cross-entropy losses for each model size: 2.339 for large, 2.165 for medium, and 2.091 for small. Lastly, we conducted an inference task using our most effective model.

## Table of Contents
* [Setup](#setup)
* [Estimation](#model-estimation)
    * [Data](#data)
    * [Running the code](#running-the-code)
* [Train](#model-training)
    * [Dataset](#training-data)
    * [Code and Hyperparameter](#code-and-hyperparameters)
* [Inference](#trained-language-model-inference)

## Setup
To initialize the environment and packages:
```bash
conda create -n model_training
conda activate model_training
pip install -r requirements.txt
```

## Model Estimation
We provided the functions for calculating parameters and FLOPs for GPT-3 decoder-only models.

### Data
The [`estimation/data`](/estimations/data) directory contains the structures (parameters and flops) for different GPT-3 models.

### Running the code
We provide the code to calculate total trainable parameters for training and total FLOPs for forward and backward passes of the GPT-3 decoder-only model. To run the code for estimating parameters and FLOPs for all GPT-3 models included in `/estimations/data/parameters.csv` and `/estimations/data/flops.csv`:
```bash
cd estimations
python3 main.py -o [parameter/forward_flop/backward_flop/all]

# Sample command for calculating total trainable parameters
python3 main.py -o parameter
```

## Model training
### Training Data
The dataset can be downloaded from the [HuggingFace](https://huggingface.co/datasets/DKYoon/SlimPajama-6B), make sure your git have Git Large File Storage [(LFS)](https://git-lfs.com/) package:
```bash
cd train
git clone https://huggingface.co/datasets/DKYoon/SlimPajama-6B
```

### Code and Hyperparameters
For training the model with given config stored in [`train/configs`](./train/configs). You also need to create a wandb account following the official [website](https://wandb.ai/site). After you created your account, use your API key at `line 44` in `train/train.py` for training. 

An example to train the model based on `train/configs/config.json` with 8 GPU and torchrun for parallel training:
```bash
torchrun --master_port=5001 train_vicuna.py \
    --nproc_per_node 8 \
    --wandb_run_name "config" \
    --data_dir_path "./SlimPajama-6B/" \
    --data_pt 566406 \
    --config "config.json" \
    --epochs 3 \
    --learning_rate 8e-4 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --save_steps 1200 \
    --version 2 \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --logging_steps 1 \
    --save_total_limit 10 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --lr_scheduler_type "cosine" \
    --cuda_visible_devices "0,1,2,3,4,5,6,7" \
    --max_length 1024
```
The trained model will be saved at `checkpoints_{version}` folder.

## Trained Language Model Inference
For using the trained language model to inference:
```bash
cd inference
python3 inference.py --model_dir <saved_trained_model> --prompt <prompt_content> --max_length <max_generated_text_length>
```