import models.cnn as cnn
from models.cnn import conv_block, deconv_block, deconv_decoder, VaeCNNDecoder

import torch
import torch.utils.data
from torch import nn
from utils_pytorch import *
from torch.nn import functional as F
from math import prod

class VaeEncoder(nn.Module):
    def __init__(self, layer_sizes, **kwargs):
        super(VaeEncoder, self).__init__()
        #self.flatten = Flatten()
        #self.fc_blocks = nn.Sequential(*[fc_block(in_size, out_size, **kwargs) for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:-1])])
        self.conv_blocks = nn.Sequential(
            *[conv_block(in_size, out_size) 
              for in_size, out_size in zip(layer_sizes[:-2], layer_sizes[1:-1])]
        )        
        self.fc_mu = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        self.fc_logvar = nn.Linear(layer_sizes[-2], layer_sizes[-1])

    def forward(self, x):
        x = x.transpose(1, 2)
        #x = self.flatten(x)
        #x = self.fc_blocks(x)
        x = self.conv_blocks(x)
        x = x.mean(dim=-1)
        return self.fc_mu(x), self.fc_logvar(x)


class VAE(nn.Module):
    def __init__(self, input_shape, layer_sizes, latent_size, ts_len, layer_kwargs={}, *args, **kwargs):
        super(VAE, self).__init__()
        self.layer_sizes = [input_shape[1], *layer_sizes, latent_size]
        self.encoder = VaeEncoder(self.layer_sizes, **layer_kwargs)
        self.dec_layer_sizes = [[ts_len, input_shape[1] + latent_size], *layer_sizes[::-1], self.input_shape]
        self.decoder = VaeCNNDecoder(self.dec_layer_sizes, output_shape = input_shape, **layer_kwargs)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        self.mu, self.logvar = self.encoder(x)
        z = self.reparameterize(self.mu, self.logvar)
        return self.decoder(z)

    def loss_function(self, recon_x, x, **kwargs):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='none')
        # change contribution weight of ts to loss - for some weird reason the model does not train with mean readuction
        #recon_loss = torch.mean((recon_loss[:,:kwargs.get('ts_len',13)] * kwargs.get('ts_weight',1))) + torch.mean(recon_loss[:,kwargs.get('ts_len',13):])
        recon_loss = torch.sum((recon_loss[:,:kwargs.get('ts_len',13)] * kwargs.get('ts_weight',1))) + torch.sum(recon_loss[:,kwargs.get('ts_len',13):])

        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        #kld_loss = torch.mean(-0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp(), dim = 1), dim = 0)
        kld_loss = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        adj_kld = kwargs.get('beta',1) * kld_loss
        return {'loss': recon_loss + adj_kld, 'recon_loss': recon_loss, 'kld_loss': kld_loss, 'adj_kld': adj_kld}
