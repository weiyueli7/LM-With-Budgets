# DSC180A Project 1
We provided the functions for calculating parameters and FLOPs for GPT-3 decoder-only models for preparing Project 1.
## Table of Contents
* [Setup](#setup)
* [Dataset](#dataset)
* [Running the code](#running-the-code)

## Setup
To initialize the environment and packages:
```bash
conda create -n <your_env_name>
conda activate <your_env_name>
pip install -r requirements.txt
```

## Dataset
The `data/parameters.csv` contains the structures for different GPT-3 models.

## Running the code
We provide the code to calculate total trainable parameters for training and total FLOPs for forward and backward passes of the GPT-3 decoder-only model. To run the code for estimating parameters and FLOPs for all GPT-3 models included in `data/parameters.csv`:
```bash
python3 main.py -o [parameter/flop/all]

# Sample command for calculating total trainable parameters
python3 main.py -o parameter
```
