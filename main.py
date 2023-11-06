from utils import *

import pandas as pd
import numpy as np



def main():

    # Parameters Estimations

    # Load data
    df = pd.read_csv('data/parameters.csv')
    answer = df['nparams']
    s_vocab = np.ones(df.shape[0])*50257
    n_ctx = np.ones(df.shape[0])*1024
    n_layers = df['nlayers']
    d_model = df['dmodel']
    n_heads = df['nheads']
    d_head = df['dhead']
    for i in range(int(df.shape[0])):
        print('-------------------------')
        print(f"Estimating parameters for model {df['Model Name'][i]}...")
        print(f'Answer: {answer[i]}; Estimate: {estimate_parameters(s_vocab[i], n_ctx[i], n_layers[i], d_model[i], n_heads[i], d_head[i])}M')
        print('-------------------------')


if __name__ == '__main__':
    main()
