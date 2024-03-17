#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
import re


def load_Rec_TS(file = 'example_input/RecGen-training-data.csv', nreads = 1000, ts_subset_index = list(range(13)), max_dups=1, max_prop=1):
    # get input
    combdf = pd.read_csv(file)

    # limit # instances of each recombinase sequence for each ts to max_dups
    combdf['dups'] = 1
    cdf = combdf.groupby(['trained_target_site', 'Sequence', 'target_sequence']).count().reset_index()
    cdf.to_csv('/content/drive/MyDrive/Data/Protein/combdf_count.csv')
    cdf['dups'] = cdf['dups'].apply(lambda x: min(x,max_dups))
    combdf = cdf.loc[cdf.index.repeat(cdf['dups'])]


    # calculating minimum number of sequences per library
    mdf = combdf.groupby('trained_target_site').count()["target_sequence"].min()
    print(mdf)
    # combdf = combdf.groupby('trained_target_site').apply(
    #     lambda x : x.sample(mdf)
    # )

    # limit the size of each ts library to mdf*max_prop
    max_mdf = mdf*max_prop
    combdf = combdf.groupby('trained_target_site').apply(
        lambda x : x.sample(max_mdf) if len(x) >= max_mdf else x.sample(len(x))
    )

    # sample to get nreads from each targetsite
    # combdf = combdf.groupby('target_sequence').apply(lambda x : x.sample(nreads))

    # combine targetsite with Sequence for training input
    combdf['target_sequence_subset'] = [''.join(np.array(list(x))[ts_subset_index]) for x in combdf.target_sequence]
    combdf['combined_sequence'] = combdf.target_sequence_subset + combdf.Sequence
    combdf.reset_index(drop = True, inplace=True) # reset necessary to get integer indices later

    print(f'{len(combdf):,}')

    return combdf

def load_Rec_TS_orig(file = 'example_input/RecGen-training-data.csv', nreads = 1000, ts_subset_index = list(range(13)), max_dups=1, max_prop=1):
    # get input
    combdf = pd.read_csv(file)

    # sample to get nreads from each targetsite
    combdf = combdf.groupby('target_sequence').apply(lambda x : x.sample(1000))

    # combine targetsite with Sequence for training input
    combdf['target_sequence_subset'] = [''.join(np.array(list(x))[ts_subset_index]) for x in combdf.target_sequence]
    combdf['combined_sequence'] = combdf.target_sequence_subset + combdf.Sequence
    combdf.reset_index(drop = True, inplace=True) # reset necessary to get integer indices later

    print(f'{len(combdf):,}')

    return combdf

def split_train_test(combdf, by = 'target_sequence_subset', train_split = 0.9):
    # split into training and test
    combdf.reset_index(drop = True, inplace=True) # reset necessary to get integer indices later
    train_data = combdf.groupby(by).apply(lambda x: x.sample(frac=train_split, random_state = 0))
    train_index = train_data.index.get_level_values(1).sort_values()
    test_index = combdf.drop(train_index).index.sort_values()
    return train_index, test_index

def transform_protein(c):
    if c == 'X' or c == 'Z' or c == 'B':
        c = '-'
    return c

def strip_protein(line):
    return ''.join([transform_protein(c) for c in line if c.isupper() or c == '-'])

def load_txt(file_path):
    with open(file_path) as f:
        sequences = []
        identifiers = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                sequences.append('')
                identifiers.append(line[1:])
            else:
                sequences[-1] += strip_protein(line)  
    assert len(sequences) == len(identifiers)
    df = pd.DataFrame({
        'Identifier':identifiers, 
        'Sequence': sequences
    })
    return df  

def load_unlabeled(file):
    # get input
    df = load_txt(file)
    df['combined_sequence'] = df.Sequence
    return df

