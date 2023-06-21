import models.cnn as cnn
from models.cnn import conv_block, deconv_block, deconv_decoder, VaeCNNDecoder

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
        # new code
        #self.flatten = Flatten()
        self.conv_blocks = nn.Sequential(
            *[conv_block(in_size, out_size) 
              for in_size, out_size in zip(layer_sizes[:-2], layer_sizes[1:-1])]
        )
        # changed: self.conv_blocks = nn.Sequential(*[conv_block(in_size, out_size) for in_size, out_size in zip(layer_sizes[:-2], layer_sizes[1:-1])])
        self.fc_mu = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        self.fc_logvar = nn.Linear(layer_sizes[-2], layer_sizes[-1])

    def forward(self, x):
        x = x.transpose(1, 2)
        y = x[:,:,:self.ts_len]
        y = y.transpose(1, 2)
        #x = x[:,self.ts_len:] # protein part - original is without it
        #x = self.flatten(x)
        x = self.conv_blocks(x)
        #x = self.blocks(x)
        x = x.mean(dim=-1) #avg of channels over the sequence lengths (spatial dim) --> we have bath size * channel size
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar, y

class CVAE(nn.Module):
    def __init__(self, input_shape, layer_sizes, latent_size, ts_len, layer_kwargs={}, *args, **kwargs):
        super(CVAE, self).__init__()
        self.padding = 0
        self.input_shape = (input_shape[0]-ts_len, input_shape[1])
        self.ts_len = ts_len
        if ts_len == 0:
            ts_len = 13
            self.padding = ts_len
        #self.layer_sizes = [prod(input_shape), *layer_sizes, latent_size]
        #new code
        self.layer_sizes = [input_shape[1], *layer_sizes, latent_size]
        self.encoder = VaeEncoder(self.layer_sizes, ts_len, **layer_kwargs)
        self.dec_layer_sizes = [[ts_len, input_shape[1] + latent_size], *layer_sizes[::-1], self.input_shape]
        self.decoder = VaeCNNDecoder(self.dec_layer_sizes, output_shape = self.input_shape, **layer_kwargs)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        if self.padding > 0:
            x = torch.nn.functional.pad(
                x,
                (0, 0, self.padding, 0)
            )
        self.mu, self.logvar, y = self.encoder(x)
        z = self.reparameterize(self.mu, self.logvar)
        z = z.unsqueeze(dim = -2)
        z = z.repeat(1, y.shape[-2], 1)
        z = torch.cat((y, z), -1) # combine ts with z
        x_reconstructed = self.decoder(z)
        x_y_reconstructed = torch.cat((y,x_reconstructed),1)
        x_y_reconstructed = x_y_reconstructed[self.padding:]
        return x_y_reconstructed

    def loss_function(self, recon_x, x, **kwargs):
        # recon_x = recon_x[:, max(self.padding, self.ts_len):]
        # x = x[:, self.ts_len:]
        recon_loss = F.cross_entropy(recon_x.transpose(1,2), x.transpose(1,2), reduction='none')
        # recon_loss = F.binary_cross_entropy(recon_x, x, reduction='none')
        # change contribution weight of ts to loss - for some weird reason the model does not train with mean readuction
        # recon_loss = torch.mean((recon_loss[:,:kwargs.get('ts_len',13)] * kwargs.get('ts_weight',1))) + torch.mean(recon_loss[:,kwargs.get('ts_len',13):])
        # if 'ts_len' in kwargs and kwargs['ts_len']>0:
        #     recon_loss = torch.sum((recon_loss[:,:kwargs.get('ts_len',13)] * kwargs.get('ts_weight',1))) + torch.sum(recon_loss[:,kwargs.get('ts_len',13):])
        # else: 
        recon_loss = torch.sum(recon_loss)


        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        #kld_loss = torch.mean(-0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp(), dim = 1), dim = 0)
        kld_loss = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        adj_kld = kwargs.get('beta',1) * kld_loss
        return {'loss': recon_loss + adj_kld, 'recon_loss': recon_loss, 'kld_loss': kld_loss, 'adj_kld': adj_kld}
