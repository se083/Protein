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
from vae_train_val import full_main
from vae_train_ssl import pre_train

import sys
import os
import shutil
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAEs and perform leave-one-out cross-validation.')
    parser.add_argument('-o','--outfolder', nargs='?', default='output_loocv/', type=str, help='default = %(default)s; output folder for saving results')
    parser.add_argument('-pi','--pre_input_data', nargs='?', default='example_input/training_data_masked.csv', type=str, help='default = %(default)s; fasta file containing alignment sequences')
    parser.add_argument('-i','--input_data', nargs='?', default='example_input/training_data_masked.csv', type=str, help='default = %(default)s; csv input table containing the columns Sequence (recombinase in amino acid).')
    parser.add_argument('-ptm','--pre_train_model_type', nargs='?', default='CVAE', type=str, help='default = %(default)s; select the type of VAE model to use; options: VAE, CVAE, SVAE, MMD_VAE, VQ_VAE', dest='pre_train_model_type')
    parser.add_argument('-ftm','--fine_tune_model_type', nargs='?', default='CVAE', type=str, help='default = %(default)s; select the type of VAE model to use; options: VAE, CVAE, SVAE, MMD_VAE, VQ_VAE', dest='fine_tune_model_type')
    parser.add_argument('--specific_libs', nargs='*', default='all', type=str, help='default = %(default)s; leave one out testing only for specific libraries, seperate names space')
    parser.add_argument('-b','--batch_size', nargs='?', default=128, type=int, help='default = %(default)s; the number of samples in each processing batch', dest='batch_size')
    parser.add_argument('-e','--epochs', nargs='?', default=40, type=int, help='default = %(default)s; the number of iterations the training is going through', dest='epochs')
    parser.add_argument('-l','--layer_sizes', nargs='*', default=[64,32], type=int, help='default = %(default)s; the hidden layer dimensions in the model', dest='layer_sizes')
    parser.add_argument('-lr','--learning_rate', nargs='?', default=0.001, type=float, help='default = %(default)s; the rate of learning, higher means faster learning, but can lead to less accuracy', dest='learning_rate')
    parser.add_argument('-nl','--num_layers', nargs='?', default=1, type=int, help='default = %(default)s; the number of LSTM layers', dest='num_layers')

    args = parser.parse_args()
    for es in [10, 20, 40]:
        for lr in [1e-3, 1e-4, 1e-5]:
            for bs in [32, 64, 128]:
                lys = args.layer_sizes
                lys = ' '.join(str(x) for x in lys)
                libs = ' '.join(args.specific_libs)
                las = 2
                nl = args.num_layers
                model_folder = os.path.join(args.outfolder, f'{es}-{bs}-{lr}-{las}-{lys.replace(" ", "_")}-{libs.replace(" ", "_")}-{nl}')
                if os.path.exists(model_folder):
                    pred_path = os.path.join(model_folder, 'prediction_hamming.csv')
                    if os.path.exists(pred_path):
                        continue
                    else:
                        shutil.rmtree(model_folder)
                settings = f'vae --outfolder {model_folder} \
                        --input_data {args.pre_input_data} \
                        --epochs {es}\
                        --batch_size {bs}\
                        --model_type {args.pre_train_model_type}\
                        --learning_rate {lr}\
                        -l {lys}\
                        -nl {nl}'
                print(settings.split())
                sys.argv = settings.split()
                pre_train()
                pre_model = model_folder + '/' + args.pre_train_model_type + '_weights.pt'
                model_folder = os.path.join(model_folder, f'40-128-0.001-{libs.replace(" ", "_")}')
                settings = f'vae --outfolder {model_folder} \
                        --input_data {args.input_data} \
                        --epochs 40\
                        --batch_size 128\
                        --model_type {args.fine_tune_model_type}\
                        --learning_rate 0.001\
                        --specific_libs {args.specific_libs}\
                        --pre_model {pre_model}\
                        -l {lys}\
                        -nl {nl}'
                print(settings.split())
                sys.argv = settings.split()
                full_main()

