import models.rnn as rnn
from models.rnn import rnn_forward, VaeRNNDecoder

import torch
import torch.utils.data
from torch import nn
from utils_pytorch import *
from torch.nn import functional as F
from math import prod

class Splitter(nn.Module):
    def __init__(self, ts_len, **kwargs):
        super(Splitter, self).__init__()
        self.ts_len = ts_len
        self.flatten = Flatten()

    def forward(self, x):
        y = x[:,:self.ts_len] # target site part
        y_flat = self.flatten(y)
        return y, y_flat
    
class MLP(nn.Module):
    def __init__(self, input_shape, layer_sizes, latent_size, ts_len, layer_kwargs={}, *args, **kwargs):
        super(MLP, self).__init__()
        self.splitter = Splitter(ts_len)
        output_shape = (input_shape[0]-ts_len, input_shape[1])
        self.dec_layer_sizes = [[ts_len, input_shape[1] + latent_size], *layer_sizes[::-1], self.input_shape]
        self.decoder = VaeRNNDecoder(self.dec_layer_sizes, output_shape = output_shape, **layer_kwargs)

    def forward(self, x):
        y, y_flat = self.splitter(x)
        return torch.cat((y,self.decoder(y_flat)),1)

    def loss_function(self, recon_x, x, **kwargs):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        # change contribution weight of ts to loss - for some weird reason the model does not train with mean readuction

        return {'loss': recon_loss}