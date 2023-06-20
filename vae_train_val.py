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
import torch
import random
from vae_train_loocv import analyse_model, count_parameters

def main(
    data = 'example_input/training_data_masked.csv',
    model_type = 'CVAE',
    latent_size = 2,
    layer_sizes = [64, 32],
    batch_size = 128,
    epochs = 40,
    beta = 2,
    weight_decay = 0,
    ts_weight = 1,
    nreads = 1000,
    ts_subset_index = list(range(13)),
    learning_rate = 1e-04,
    dropout_p = 0.1,
    num_embeddings=10,
    embedding_dim=1,
    summary_function=np.min,
    hamming_cutoff=1,
    specific_libs = 'all',
    n_out = 1000,
    pre_model = None):

    ###### load and prepare data ######
    # some variables needed later
    vocab_list = utils.vocab_list
    ts_len = len(ts_subset_index)
    summary_function = np.min

    # load the data
    combdf = ld.load_Rec_TS(file = data, nreads = nreads, ts_subset_index=ts_subset_index)

    # make indices and encode to one-hot
    yx_ind = np.array(utils.seqaln_to_indices(combdf.combined_sequence,vocab_list))
    yx_oh = utils.get_one_hot(yx_ind, len(vocab_list))

    ###### training leave one out ######
    # initial setup for loop
    out_dict = defaultdict(list)

    # loop through all possible leave one out testings
    if specific_libs == 'all' or (len(specific_libs)==1 and specific_libs[0] == 'all'):
        uts = pd.unique(combdf.target_sequence_subset)
    else:
        print(specific_libs)
        uts = pd.unique(combdf.target_sequence_subset[combdf.trained_target_site.isin(specific_libs)])
        if len(uts) == 0:
            print('no valid libraries named!')
            sys.exit()
        elif len(uts) != len(specific_libs):
            print('some of the named libraries are not valid!')

    print('Training')
    # prepare data and train
    train_index, test_index = utp.validation_indices(combdf.target_sequence_subset, uts)

    model = vae_models[model_type](input_shape=yx_oh.shape[1:], layer_sizes=layer_sizes, latent_size=latent_size, ts_len=ts_len, num_embeddings=num_embeddings, embedding_dim=embedding_dim, layer_kwargs={'dropout_p':dropout_p})
    model.to('cuda')
    # summary(model, input_size = (357,22))
    print(f'{count_parameters(model):,}')
    #load the model 
    if pre_model is not None:
        weights = torch.load(pre_model)
        model.load_state_dict(weights)
    model, loss_df = training.model_training(model=model, x_train=yx_oh[train_index], x_test=yx_oh[test_index], epochs=epochs, batch_size=batch_size, loss_kwargs={'beta':beta, 'ts_weight':ts_weight, 'ts_len':ts_len}, optimizer_kwargs={'weight_decay':weight_decay, 'lr':learning_rate})
    
    val_index = test_index
    for i, leave_out_y in enumerate(uts):
        print('#### Evaluating on target site Nr.',i+1,'/',len(uts),':', leave_out_y, ' ####')
        train_index, test_index = utp.leave_out_indices(combdf.target_sequence_subset, leave_out_y, yx_ind[:,:ts_len], hamming_cutoff=hamming_cutoff)
        # train_index = train_index[~train_index.isin(val_index)]
        train_index = list(set(train_index).difference(set(val_index)))
        out_dict = analyse_model(out_dict, loss_df, summary_function, leave_out_y, yx_oh, yx_ind, model, train_index, test_index, vocab_list, ts_len, model_type, n_out)

    for key, value in out_dict.items():
        out_dict[key] = pd.concat(value)

    return out_dict, model


def full_main():
    # argument parser
    parser = argparse.ArgumentParser(description='Train VAEs and perform leave-one-out cross-validation.')
    parser.add_argument('-o','--outfolder', nargs='?', default='output_loocv/', type=str, help='default = %(default)s; output folder for saving results', dest='outprefix')
    parser.add_argument('-i','--input_data', nargs='?', default='example_input/training_data_masked.csv', type=str, help='default = %(default)s; csv input table containing the columns target_sequence and Sequence (recombinase in amino acid).', dest='input_data')
    parser.add_argument('-m','--model_type', nargs='?', default='CVAE', type=str, help='default = %(default)s; select the type of VAE model to use; options: VAE, CVAE, SVAE, MMD_VAE, VQ_VAE', dest='model_type')
    parser.add_argument('-z','--latent_size', nargs='?', default=2, type=int, help='default = %(default)s; the latent size dimensions', dest='latent_size')
    parser.add_argument('-l','--layer_sizes', nargs='*', default=[64,32], type=int, help='default = %(default)s; the hidden layer dimensions in the model', dest='layer_sizes')
    parser.add_argument('-b','--batch_size', nargs='?', default=128, type=int, help='default = %(default)s; the number of samples in each processing batch', dest='batch_size')
    parser.add_argument('-e','--epochs', nargs='?', default=40, type=int, help='default = %(default)s; the number of iterations the training is going through', dest='epochs')
    parser.add_argument('-a','--beta', nargs='?', default=1, type=float, help='default = %(default)s; the final weight on the KL-Divergence', dest='beta')
    parser.add_argument('-wd','--weight_decay', nargs='?', default=0, type=int, help='default = %(default)s; a regularisation factor, does not go well with VQ VAE', dest='weight_decay')
    parser.add_argument('-tw','--ts_weight', nargs='?', default=1, type=int, help='default = %(default)s; the multiplyer applied to the reconstruction loss of the target site', dest='ts_weight')
    parser.add_argument('-r','--nreads', nargs='?', default=10, type=int, help='default = %(default)s; number of protein sequences per library to use for training', dest='nreads')
    parser.add_argument('-ts','--ts_slice', nargs='?', default='half', type=str, help='default = %(default)s; what part of the target site to use; options: "half", "+pos14", "donut", "full"', dest='ts_slice')
    parser.add_argument('-lr','--learning_rate', nargs='?', default=0.001, type=float, help='default = %(default)s; the rate of learning, higher means faster learning, but can lead to less accuracy', dest='learning_rate')
    parser.add_argument('-d','--dropout_p', nargs='?', default=0.1, type=float, help='default = %(default)s; dropout_p probability for every layer, 0 means no dropout', dest='dropout_p')
    parser.add_argument('-D','--num_embeddings', nargs='?', default=10, type=int, help='default = %(default)s; VQ_VAE only, number of "categories" to embed', dest='num_embeddings')
    parser.add_argument('-K','--embedding_dim', nargs='?', default=1, type=int, help='default = %(default)s; VQ_VAE only, number of values to represent each embedded "category"', dest='embedding_dim')
    parser.add_argument('-s','--summary_function', nargs='?', default='min', type=str, help='default = %(default)s; summary function to use to reduce the distances measured from the prediction to the truth', dest='summary_function')
    parser.add_argument('-c','--hamming_cutoff', nargs='?', default=1, type=int, help='default = %(default)s; hamming distance cutoff to exclude other libraries from the training dataset', dest='hamming_cutoff')
    parser.add_argument('--specific_libs', nargs='*', default='all', type=str, help='default = %(default)s; leave one out testing only for specific libraries, seperate names space', dest='specific_libs')
    parser.add_argument('-n','--n_models', nargs='?', default=1, type=int, help='default = %(default)s; number of models to train for each leave one out', dest='n_models')
    parser.add_argument('--n_out', nargs='?', default=1000, type=int, help='default = %(default)s; number of predictions to make for each model and library', dest='n_out')
    parser.add_argument('--seed', nargs='?', default=0, type=int, help='default = %(default)s; default random seed', dest='seed')
    parser.add_argument('-p','--pre_model', nargs='?', default=None, type=str, help='default = %(default)s; path to the pre-trained model', dest='pre_model')

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    ts_slice_options = {'half':list(range(13)),'+pos14':list(range(14)) + [20], 'donut':list(range(14)) + list(range(20,34)), 'full':list(range(34)) }
    ts_subset_index = ts_slice_options[args.ts_slice]

    summary_function_options = {'min':np.min, '10per':lambda x : np.percentile(x,10,interpolation='nearest'), 'mean':np.mean}
    summary_function = summary_function_options[args.summary_function]

    # string for saving
    folderstr = args.outprefix
    print('output going into: ' + folderstr)

    out_collect = defaultdict(list)

    for model_nr in range(args.n_models):
        out, model = main(
            data = args.input_data,
            model_type = args.model_type,
            latent_size = args.latent_size,
            layer_sizes = args.layer_sizes,
            batch_size = args.batch_size,
            epochs = args.epochs,
            beta = args.beta,
            weight_decay = args.weight_decay,
            ts_weight = args.ts_weight,
            nreads = args.nreads,
            ts_subset_index = ts_subset_index,
            learning_rate = args.learning_rate,
            dropout_p = args.dropout_p,
            num_embeddings = args.num_embeddings,
            embedding_dim = args.embedding_dim,
            summary_function = summary_function,
            hamming_cutoff = args.hamming_cutoff,
            specific_libs = args.specific_libs,
            n_out = args.n_out,
            pre_model = args.pre_model)

        # collect output data frames in lists and add the model_nr
        for key, value in out.items():
            out_collect[key].append( value.assign(Model_Nr=model_nr) )
        
        # save model
        # torch.save(model, folderstr + '/' + args.model_type + '_' + str(model_nr) + '.pt')

    # concatenate the lists into dataframe
    for key, value in out_collect.items():
        out_collect[key] = pd.concat(value)


    #### save into csv ####
    print('saving data...')

    # check if folder exists - make new string if existing
    folderstr = utils.check_mkdir(folderstr)

    # save parameters
    with open(folderstr + "/parameters.txt","w") as f: 
        f.writelines([str(key) + f':\t' + str(val) + '\n' for key, val in vars(args).items()])

    # save all dataframes
    for key, value in out_collect.items():
        value.to_csv(folderstr + '/' + key + '.csv', index = False)

if __name__ == '__main__':
    full_main()