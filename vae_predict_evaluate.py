#!/usr/bin/env python
# Predict a specific sequence
from models import vae_models
import training
import utils
import numpy as np
import pandas as pd
import load_data as ld
import utils_train_predict as utp
import torch
from datetime import datetime
from itertools import chain
import re
import os
import argparse


# argument parser
parser = argparse.ArgumentParser(description='Predict recombinases for defined target sites with saved models.')
parser.add_argument('-o','--outfolder', nargs='?', default='output_prediction', type=str, help='default = %(default)s; output folder for saving results', dest='outprefix')
parser.add_argument('-m','--model_folder', nargs='?', default='saved_models', type=str, help='Select the folder where the model files are saved', dest='model_folder')
parser.add_argument('-t','--target_sequences', nargs='?', default='example_input/predict_ts.csv', type=str, help='The target sites you want to predict recombinases for. In csv format, must contain column "target_sequence"', dest='target_sequences')
parser.add_argument('-d','--training_data', nargs='?', default='example_input/training_data_masked.csv', type=str, help='default = %(default)s; define csv input file with training data. Necessary for estimation of latent space spread.', dest='training_data')
parser.add_argument('-n','--n_out', nargs='?', default=100, type=int, help='default = %(default)s; number of predictions to make for each model', dest='n_out')

args = parser.parse_args()


# define variables
target_sequence = list(pd.read_csv(args.target_sequences).target_sequence)
out_samples = args.n_out
modeldir = args.model_folder

###### load and prepare data ######

# read paramters from models
with open(modeldir + '/parameters.txt') as f:
    params = {k:eval(v) for (k,v) in [x.split(':\t') for x in f.read().splitlines()]}

# get model files from modeldir
regex = re.compile(r'.*.pt$')
model_files = list(filter(regex.search, os.listdir(modeldir)))

# subset target_sequence
target_sequence = [''.join(np.array(list(x))[params['ts_subset_index']]) for x in target_sequence]
ts_len = len(params['ts_subset_index'])

# load training data
combdf = ld.load_Rec_TS_orig(file = args.training_data, nreads = params['nreads'], ts_subset_index=params['ts_subset_index'])

# make indices
vocab_list = utils.vocab_list
yx_ind = np.array(utils.seqaln_to_indices(combdf.combined_sequence,vocab_list))
target_lookup = {}
target_ids = []
for target in combdf['target_sequence_subset']:
    if target not in target_lookup:
        target_lookup[target] = len(target_lookup)
    target_ids.append(target_lookup[target])
target_ids = np.array(target_ids)
num_targets = len(target_lookup)
y_oh = utils.get_one_hot(target_lookup.keys(), len(vocab_list))
# convert to one hot arrays
yx_oh = utils.get_one_hot(yx_ind, len(vocab_list))

# convert target sequences to one hot
y_pred_ind = np.array(utils.seqaln_to_indices(target_sequence,vocab_list))
y_pred_oh = utils.get_one_hot(y_pred_ind, len(vocab_list))
ts_hamming = utils.np_hamming_dist(y_pred_ind[:,None], y_oh[None,:], axis=-1) 
if 'CNN' in params['model_type'] or 'RNN' in params['model_type']:
    ts_oh = np.expand_dims(y_pred_oh, axis = 0)
    ts_oh = np.repeat(ts_oh, repeats=out_samples, axis=0)
else:
    ts_oh = np.repeat(np.reshape(a=y_pred_oh, newshape=(len(target_sequence),ts_len*yx_oh.shape[2])), repeats=out_samples, axis=0)

###### Predict ########

pred_str_list = []
for i in model_files:
    print('Predicting with: ' + i)
    model = torch.load(modeldir + '/' + i)

    z_train = training.model_predict(model.encoder, yx_oh, 10000)
    

    z_found = utp.z_unif_sampling(z_values=z_train, n_samples=len(target_sequence)*out_samples)
    if 'CNN' in params['model_type'] or 'RNN' in params['model_type']:
        z_found = np.expand_dims(z_found, axis = -2)
        z_found = np.repeat(z_found, repeats=ts_oh.shape[-2], axis=-2)
    z_found = np.concatenate((ts_oh,z_found),1)

    yx_pred_zsearch_ind = utp.predict_and_index(model.decoder, z_found, 32)

    x_hamming = utils.np_hamming_dist(yx_pred_zsearch_ind[:,None], yx_ind[:,ts_len:][None,:], axis=-1) 
    for target_idx in range(num_targets):
        target_mask = target_idx == target_ids
        target_x_hamming = x_hamming[target_mask]
        target_min_hamming = np.min(target_x_hamming, axis=-1)
        pred_hamming = utp.hamming_distance_uneven_df(
            loop_array=yx_pred_zsearch_ind, 
            array1=yx_ind[target_mask], 
            ts_len=ts_len, 
            vocab_list=vocab_list, 
            ts_labels=np.array([yx_ind[test_index[0],:ts_len]]), 
            summarise_function=np.min)


    pred_str_list.append(pd.DataFrame(
        {'Sequence' : utils.indices_to_seqaln(yx_pred_zsearch_ind, vocab_list, join = True), 
        'TargetSequence' : list(chain(*[[x] * out_samples for x in target_sequence])), 
        'Model' : i }))

# create output folder
folderstr = args.outprefix
folderstr = utils.check_mkdir(folderstr)

# write parameters
params['modeldir'] = modeldir
params['target_sequence'] = target_sequence
with open(folderstr + "/parameters.txt","w") as f: f.writelines([str(key) + f':\t' + str(val) + '\n' for key, val in params.items()])

# write predictions
df = pd.concat(pred_str_list)

out_dict = {}
for ts in df['TargetSequence'].uniques():
    df_ts = df['TargetSequence'==ts]
    pred_str = df_ts.to_numpy()
    ts_hamming = utils.np_hamming_dist(ts, yx_ind[:,:ts_len])
    ts_hamming = sorted(ts_hamming)
    for i in len(ts_hamming):
        closest_index = np.array(train_index)[ts_hamming == ts_hamming[i]]
        closest_hamming = utp.hamming_distance_uneven_df(loop_array=yx_ind[closest_index], array1=yx_ind[test_index], ts_len=ts_len, vocab_list=vocab_list, ts_labels=np.array([yx_ind[test_index[0],:ts_len]]), summarise_function=summary_function)
        closest_hamming['DataType'] = 'Closest<>Truth'
        out_dict['prediction_hamming'].append(closest_hamming)

df.to_csv(folderstr + '/prediction_str.csv', index = False)

