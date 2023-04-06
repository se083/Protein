import models.rnn as rnn
from models.rnn import VaeRNNDecoder

import torch
import torch.utils.data
from utils_pytorch import *
from torch import nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
import math, os
from math import prod

class VaeEncoder(nn.Module):
    def __init__(self, layer_sizes, **kwargs):
        super(VaeEncoder,self).__init__()
        #self.flatten = Flatten()
        #self.fc_blocks = nn.Sequential(*[fc_block(in_size, out_size, **kwargs) for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:-1])])
        self.hidden_size = layer_sizes[-2]
        self.num_layers = 1
        self.rnn = nn.LSTM(
            input_size = layer_sizes[0], 
            hidden_size = self.hidden_size,
            num_layers = self.num_layers, #can play around with later
            batch_first = True,
            dropout = 0,
            bidirectional = False
        )          
        self.fc_mu = nn.Linear(layer_sizes[-2], layer_sizes[-1])

    def forward(self, x):
        y = x[:,:self.ts_len] # target site part
        initial_state = (
            torch.zeros([x.shape[0], self.num_layers, self.hidden_size]), 
            torch.zeros([x.shape[0], self.num_layers, self.hidden_size])
        )
        output, (h, c) = self.rnn(x, initial_state)
        x = output[:, -1]
        z = self.fc_mu(x)
        return z, y

class MMD_VAE(nn.Module):
    def __init__(self, input_shape, layer_sizes, latent_size, ts_len, layer_kwargs={}, *args,**kwargs):
        super(MMD_VAE, self).__init__()
        self.layer_sizes = [input_shape[1], *layer_sizes, latent_size]
        self.encoder = VaeEncoder(self.layer_sizes, **layer_kwargs)
        self.dec_layer_sizes = [[ts_len, input_shape[1], latent_size], *layer_sizes[::-1], self.input_shape]
        self.decoder = VaeRNNDecoder(self.dec_layer_sizes, output_shape = input_shape, **layer_kwargs)

    def forward(self, x):
        self.z = self.encoder(x)
        return self.decoder(self.z)

    def gaussian_kernel(self, a, b):
        dim1_1, dim1_2 = a.shape[0], b.shape[0]
        depth = a.shape[1]
        a = a.view(dim1_1, 1, depth)
        b = b.view(1, dim1_2, depth)
        a_core = a.expand(dim1_1, dim1_2, depth)
        b_core = b.expand(dim1_1, dim1_2, depth)
        numerator = (a_core - b_core).pow(2).mean(2)/depth
        return torch.exp(-numerator)

    def compute_mmd(self, a, b):
        return self.gaussian_kernel(a, a).mean() + self.gaussian_kernel(b, b).mean() - 2*self.gaussian_kernel(a, b).mean()

    def loss_function(self, recon_x, x, **kwargs):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='none')
        recon_loss = 0.5*(torch.mean((recon_loss[:,:kwargs.get('ts_len',13)] * kwargs.get('ts_weight',1))) + torch.mean(recon_loss[:,kwargs.get('ts_len',13):]))

        reference_distribution = torch.randn(1000, self.z.shape[1], requires_grad=False)
        reference_distribution = reference_distribution.to(torch.device('cuda'))
        mmd_loss = self.compute_mmd(reference_distribution, self.z)
        adj_mmd = kwargs.get('beta',1) * mmd_loss

        return {'loss': recon_loss + adj_mmd, 'recon_loss': recon_loss, 'mmd_loss': mmd_loss, 'adj_mmd': adj_mmd}

