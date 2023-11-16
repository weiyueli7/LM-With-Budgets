# DSC180A Project 1
We provided the functions for calculating parameters and FLOPs for GPT-3 decoder-only models for preparing Project 1.
## Table of Contents
* [Dataset](#dataset)
* [Running the code](#running-the-code)

## Dataset
The [`data`](data) directory contains the structures (parameters and flops) for different GPT-3 models.

## Running the code
We provide the code to calculate total trainable parameters for training and total FLOPs for forward and backward passes of the GPT-3 decoder-only model. To run the code for estimating parameters and FLOPs for all GPT-3 models included in `data/parameters.csv`:
```bash
python3 main.py -o [parameter/forward_flop/backward_flop/all]
# Sample command for calculating total trainable parameters
python3 main.py -o parameter
```
