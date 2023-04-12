#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
import re


def load_Rec_TS(file = 'example_input/RecGen-training-data.csv', nreads = 1000, ts_subset_index = list(range(13))):
    # get input
    combdf = pd.read_csv(file)

    # calculating minimum number of sequences per library
    mdf = combdf.groupby('trained_target_site').count()["target_sequence"].min()
    combdf = combdf.groupby('trained_target_site').apply(
        lambda x : x.sample(mdf)
    )

    # sample to get nreads from each targetsite
    # combdf = combdf.groupby('target_sequence').apply(lambda x : x.sample(nreads))

    # combine targetsite with Sequence for training input
    combdf['target_sequence_subset'] = [''.join(np.array(list(x))[ts_subset_index]) for x in combdf.target_sequence]
    combdf['combined_sequence'] = combdf.target_sequence_subset + combdf.Sequence
    combdf.reset_index(drop = True, inplace=True) # reset necessary to get integer indices later
    return combdf


def split_train_test(combdf, by = 'target_sequence_subset', train_split = 0.9):
    # split into training and test
    combdf.reset_index(drop = True, inplace=True) # reset necessary to get integer indices later
    train_data = combdf.groupby(by).apply(lambda x: x.sample(frac=train_split, random_state = 0))
    train_index = train_data.index.get_level_values(1).sort_values()
    test_index = combdf.drop(train_index).index.sort_values()
    return train_index, test_index

def strip_protein(line):
    return ''.join([c for c in line if c.isupper() or c == '-'])

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
#might change column names

