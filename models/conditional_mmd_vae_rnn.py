import models.rnn as rnn
from models.rnn import rnn_forward, VaeRNNDecoder

import torch
import torch.utils.data
from torch import nn
from utils_pytorch import *
from torch.nn import functional as F
from math import prod

class VaeEncoder(nn.Module):
    def __init__(self, layer_sizes, ts_len, **kwargs):
        super(VaeEncoder, self).__init__()
        self.ts_len = ts_len
        #self.flatten = Flatten()
        #self.fc_blocks = nn.Sequential(*[fc_block(in_size, out_size, **kwargs) for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:-1])])
        # self.conv_blocks = nn.Sequential(
        #     *[conv_block(in_size, out_size) 
        #       for in_size, out_size in zip(layer_sizes[:-2], layer_sizes[1:-1])]
        # )
        self.hidden_size = layer_sizes[-2]
        self.num_layers = 1
        self.rnn = nn.LSTM(
            input_size = layer_sizes[1], 
            hidden_size = self.hidden_size,
            num_layers = self.num_layers, #can play around with later
            batch_first = True,
            dropout = 0,
            bidirectional = False
        )          
        self.fc_mu = nn.Linear(layer_sizes[-2], layer_sizes[-1])

    def forward(self, x):
        #x = x.transpose(1, 2)
        y = x[:,:self.ts_len] # target site part
        initial_state = (
            torch.zeros([x.shape[0], self.num_layers, self.hidden_size]), 
            torch.zeros([x.shape[0], self.num_layers, self.hidden_size])
        )
        output, (h, c) = self.rnn(x, initial_state)
        x = output[:, -1]
        z = self.fc_mu(x)
        return z, y

# VaeEncoder & VaeDecoder both used in below code (MMD_CVAE) but never directly appeared in "Bio.ipynb" or all the other code
# MMD_CVAE used in demo code (vae_train_loocv.py, vae_train_save.py, vae_load_predict.py)

class MMD_CVAE(nn.Module):
    def __init__(self, input_shape, layer_sizes, latent_size, ts_len, layer_kwargs={}, *args, **kwargs):
        super(MMD_CVAE, self).__init__()
        #new
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.input_shape = (input_shape[0]-ts_len, input_shape[1])
        self.layer_sizes = [input_shape[1], *layer_sizes, latent_size]
        self.encoder = VaeEncoder(self.layer_sizes, ts_len, **layer_kwargs)
        self.dec_layer_sizes = [[ts_len, input_shape[1], latent_size], *layer_sizes[::-1], self.input_shape] #changed
        self.decoder = VaeRNNDecoder(self.dec_layer_sizes, output_shape = self.input_shape, **layer_kwargs)

    def forward(self, x):
        self.z, y = self.encoder(x)
        #z = torch.cat((y.view(-1, prod(y.shape[1:])), self.z), 1) # combine ts with z
        return torch.cat((y,self.decoder(y, self.z)),1)

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
        #recon_loss = F.binary_cross_entropy(recon_x, x, reduction='none')
        recon_loss = self.loss(recon_x, x)
        recon_loss = 0.5*(torch.mean((recon_loss[:,:kwargs.get('ts_len',13)] * kwargs.get('ts_weight',1))) + torch.mean(recon_loss[:,kwargs.get('ts_len',13):]))
        reference_distribution = torch.randn(1000, self.z.shape[1], requires_grad=False)
        reference_distribution = reference_distribution.to(torch.device('cuda'))
        mmd_loss = self.compute_mmd(reference_distribution, self.z)
        adj_mmd = kwargs.get('beta',1) * mmd_loss

        return {'loss': recon_loss + adj_mmd, 'recon_loss': recon_loss, 'mmd_loss': mmd_loss, 'adj_mmd': adj_mmd}

