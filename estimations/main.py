from utils import *

import pandas as pd
import numpy as np
import argparse


def main(option):

    # Parameters Estimations
    if option in ["parameter", "all"]:
        # Load data
        df = pd.read_csv('data/parameters.csv')
        answer = df['nparams']
        s_vocab = np.ones(df.shape[0])*50257
        n_ctx = np.ones(df.shape[0])*1024
        n_layers = df['nlayers']
        d_model = df['dmodel']
        n_heads = df['nheads']
        d_head = df['dhead']
        print("Estimating total trainable parameters")
        for i in range(int(df.shape[0])):
            print('-------------------------')
            print(f"Estimating parameters for model {df['Model Name'][i]}...")
            print(f'Answer: {answer[i]}; Estimate: {estimate_parameters(s_vocab[i], n_ctx[i], n_layers[i], d_model[i], n_heads[i], d_head[i])}M')
            print('-------------------------')
        print('='*60)

    # # Forward Pass Flops Estimations
    # if option in ["forward_flop", "all"]:
    #     # Load data
    #     df = pd.read_csv('data/parameters.csv')
    #     flops = pd.read_csv('data/flops.csv')
    #     answer = flops['Total train compute (flops)']
    #     s_vocab = np.ones(df.shape[0])*50257
    #     n_ctx = np.ones(df.shape[0])*1024
    #     n_layers = df['nlayers']
    #     d_model = df['dmodel']
    #     n_heads = df['nheads']
    #     d_head = df['dhead']
    #     print("Estimating total forward FLOPs")
    #     for i in range(int(df.shape[0])):
    #         print('-------------------------')
    #         print(f"Estimating Forward FLOPs for model {df['Model Name'][i]}...")
    #         print(f'Answer: {answer[i]}; Estimate: {estimate_forward_flops(s_vocab[i], n_ctx[i], n_layers[i], d_model[i], n_heads[i], d_head[i])}')
    #         print('-------------------------')
    #     print('='*60)

    # # Backward Pass Flops Estimations
    # if option in ["backward_flop", "all"]:
    #     print("Estimating total backward FLOPs")
    #     # Load data
    #     df = pd.read_csv('data/parameters.csv')
    #     flops = pd.read_csv('data/flops.csv')
    #     answer = flops['Total train compute (flops)']
    #     s_vocab = np.ones(df.shape[0])*50257
    #     n_ctx = np.ones(df.shape[0])*1024
    #     n_layers = df['nlayers']
    #     d_model = df['dmodel']
    #     n_heads = df['nheads']
    #     d_head = df['dhead']
    #     for i in range(int(df.shape[0])):
    #         print('-------------------------')
    #         print(f"Estimating Backward FLOPs for model {df['Model Name'][i]}...")
    #         print(f'Answer: {answer[i]/2}; Estimate: {estimate_backward_flops(s_vocab[i], n_ctx[i], n_layers[i], d_model[i], n_heads[i], d_head[i])}')
    #         print('-------------------------')
    #     print('='*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='DSC 180A Project 1',
        epilog='Created by Weiyue, Xiaoyue, Yi'
    )

    parser.add_argument('-o', '--option', required=True, help='specify whether calculating for parameters, forward-flops, backward-flops, or all')
    
    args = parser.parse_args()
    option = args.option
    main(option)
