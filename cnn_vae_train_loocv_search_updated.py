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
import math
from vae_train_val import full_main

import sys
import os
import shutil
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAEs and perform leave-one-out cross-validation.')
    parser.add_argument('-o','--outfolder', nargs='?', default='output_loocv/', type=str, help='default = %(default)s; output folder for saving results')
    parser.add_argument('-i','--input_data', nargs='?', default='example_input/training_data_masked.csv', type=str, help='default = %(default)s; csv input table containing the columns target_sequence and Sequence (recombinase in amino acid).')
    parser.add_argument('-m','--model_type', nargs='?', default='CVAE', type=str, help='default = %(default)s; select the type of VAE model to use; options: VAE, CVAE, SVAE, MMD_VAE, VQ_VAE')
    parser.add_argument('--specific_libs', nargs='*', default='all', type=str, help='default = %(default)s; leave one out testing only for specific libraries, seperate names space')
    parser.add_argument('-z','--latent_size', nargs='?', default=2, type=int, help='default = %(default)s; the latent size dimensions', dest='latent_size')
    parser.add_argument('-l','--layer_sizes', nargs='*', default=[64,32], type=int, help='default = %(default)s; the hidden layer dimensions in the model', dest='layer_sizes')
    parser.add_argument('-b','--batch_size', nargs='?', default=128, type=int, help='default = %(default)s; the number of samples in each processing batch', dest='batch_size')
    parser.add_argument('-e','--epochs', nargs='?', default=40, type=int, help='default = %(default)s; the number of iterations the training is going through', dest='epochs')
    parser.add_argument('-nl','--num_layers', nargs='?', default=1, type=int, help='default = %(default)s; the number of LSTM layers', dest='num_layers')
    parser.add_argument('-a','--beta', nargs='?', default=1, type=float, help='default = %(default)s; the final weight on the KL-Divergence', dest='beta')
    parser.add_argument('-lr','--learning_rate', nargs='?', default=0.001, type=float, help='default = %(default)s; the rate of learning, higher means faster learning, but can lead to less accuracy', dest='learning_rate')
    # parser.add_argument('--sample_orig', default=False, action='store_true', help='use batch normalisation in the hidden layers', dest='sample_orig')
    parser.add_argument('-dec_prop','--decoder_proportion', nargs='?', default=1, type=float, help='default = %(default)s; the multiplyer applied to the reconstruction loss of the target site', dest='decoder_proportion')
    parser.add_argument('--beta_ramping', default=True, action='store_false', help='use batch normalisation in the hidden layers', dest='beta_ramping')

    # parser.add_argument('-dup','--maximum_duplicates', nargs='?', default=1, type=int, help='default = %(default)s; the multiplyer applied to the reconstruction loss of the target site', dest='ts_weight')
    # parser.add_argument('-prop','--maximum_proportion', nargs='?', default=1, type=float, help='default = %(default)s; the multiplyer applied to the reconstruction loss of the target site', dest='ts_weight')

    args = parser.parse_args()
    # for es in [10, 40]:
    #     for bs in [64, 128, 512]:
    #         for lr in [1e-3, 1e-4, 1e-5]:
    #             for las in [2, 4, 6]:

    count = 0
    for beta in [1, 0.1, 0.01]:
        for lr in [1e-3, 1e-4, 1e-5]:
            for dup_big in [1, 5, math.inf]:
                for dup_small in [1, 5, math.inf]:
                    for prop in [1, 5, math.inf]:
                        for dec_prop in [1, 2]:
                            count += 1
                            print(count)
                            lys = args.layer_sizes
                            lys = ' '.join(str(x) for x in lys)
                            libs = ' '.join(args.specific_libs)
                            es = args.epochs
                            # bs = args.batch_size
                            nl = args.num_layers
                            beta_ramping = args.beta_ramping
                            model_folder = args.outfolder
                            # model_folder = os.path.join(args.outfolder, f'{es}-{bs}-{lr}-{las}-{lys.replace(" ", "_")}-{libs.replace(" ", "_")}-{nl}-{beta}-{dup}-{prop}-{sample_orig}-{decoder_proportion}')
                            # if os.path.exists(model_folder):
                            #     pred_path = os.path.join(model_folder, 'prediction_hamming.csv')
                            #     if os.path.exists(pred_path):
                            #         continue
                            #     else:
                            #         shutil.rmtree(model_folder)
                            if beta_ramping:
                                settings = f'vae --outfolder {model_folder} \
                                        --input_data {args.input_data} \
                                        --epochs {es}\
                                        --batch_size {args.batch_size}\
                                        --latent_size {args.latent_size}\
                                        --layer_sizes {lys}\
                                        --model_type {args.model_type}\
                                        --learning_rate {lr}\
                                        --specific_libs {libs}\
                                        --num_layers {nl}\
                                        --beta {beta}\
                                        --maximum_duplicates_small {dup_small}\
                                        --maximum_duplicates_big {dup_big}\
                                        --maximum_proportion {prop}\
                                        --decoder_proportion {dec_prop}'
                            else:
                                settings = f'vae --outfolder {model_folder} \
                                        --input_data {args.input_data} \
                                        --epochs {es}\
                                        --batch_size {args.batch_size}\
                                        --latent_size {args.latent_size}\
                                        --layer_sizes {lys}\
                                        --model_type {args.model_type}\
                                        --learning_rate {lr}\
                                        --specific_libs {libs}\
                                        --num_layers {nl}\
                                        --beta {beta}\
                                        --beta_ramping\
                                        --maximum_duplicates_small {dup_small}\
                                        --maximum_duplicates_big {dup_big}\
                                        --maximum_proportion {prop}\
                                        --decoder_proportion {dec_prop}'
                            print(settings.split())
                            sys.argv = settings.split()
                            try:
                                full_main()
                            except Exception as e:
                                print(e)
