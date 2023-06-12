#!/usr/bin/env python 
import load_data as ld
from models import vae_models
import training
import utils
import utils_train_predict as utp
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import subprocess
import argparse
from vae_train_loocv import full_main

import sys
import os
import shutil
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAEs and perform leave-one-out cross-validation.')
    parser.add_argument('-o','--outfolder', nargs='?', default='output_loocv/', type=str, help='default = %(default)s; output folder for saving results')
    parser.add_argument('-i','--input_data', nargs='?', default='example_input/training_data_masked.csv', type=str, help='default = %(default)s; csv input table containing the columns target_sequence and Sequence (recombinase in amino acid).')
    parser.add_argument('-m','--model_type', nargs='?', default='CVAE', type=str, help='default = %(default)s; select the type of VAE model to use; options: VAE, CVAE, SVAE, MMD_VAE, VQ_VAE')
    parser.add_argument('--specific_libs', nargs='*', default='all', type=str, help='default = %(default)s; leave one out testing only for specific libraries, seperate names space')
    parser.add_argument('-l','--layer_sizes', nargs='*', default=[64,32], type=int, help='default = %(default)s; the hidden layer dimensions in the model', dest='layer_sizes')
    args = parser.parse_args()
    # for es in [10, 40]:
    #     for bs in [64, 128, 512]:
    #         for lr in [1e-3, 1e-4, 1e-5]:
    #             for las in [2, 4, 6]:
    for lr in [1e-3, 1e-4, 1e-5]:
        lys = args.layer_sizes
        lys = ' '.join(str(x) for x in lys)
        libs = ' '.join(args.specific_libs)
        es = 40
        bs = 128
        las = 2
        model_folder = os.path.join(args.outfolder, f'{es}-{bs}-{lr}-{las}-{lys.replace(" ", "_")}-{libs.replace(" ", "_")}')
        if os.path.exists(model_folder):
            pred_path = os.path.join(model_folder, 'prediction_hamming.csv')
            if os.path.exists(pred_path):
                continue
            else:
                shutil.rmtree(model_folder)
        settings = f'vae --outfolder {model_folder} \
                --input_data {args.input_data} \
                --epochs {es}\
                --batch_size {bs}\
                --latent_size {las}\
                --layer_sizes {lys}\
                --model_type {args.model_type}\
                --learning_rate {lr}\
                --specific_libs {libs}'
        print(settings.split())
        sys.argv = settings.split()
        full_main()
