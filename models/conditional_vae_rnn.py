import models.rnn as rnn
from models.rnn import VaeRNNDecoder

import torch
import torch.utils.data
from torch import nn
from utils_pytorch import *
from torch.nn import functional as F
from math import prod

class VaeEncoder(nn.Module):
    def __init__(self, layer_sizes, ts_len, num_layers, **kwargs):
        super(VaeEncoder, self).__init__()
        self.ts_len = ts_len
        # new code
        #self.flatten = Flatten()
        self.hidden_size = layer_sizes[-2]
        # self.num_layers = len(layer_sizes)-2
        self.num_layers = num_layers
        self.rnn = nn.LSTM(
            input_size = layer_sizes[0], 
            hidden_size = self.hidden_size,
            num_layers = self.num_layers, #can play around with later
            batch_first = True,
            dropout = 0,
            bidirectional = False
        )          
        self.fc_mu = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        self.fc_logvar = nn.Linear(layer_sizes[-2], layer_sizes[-1])

    def forward(self, x):
        y = x[:,:self.ts_len] # target site part
        initial_state = (
            torch.zeros([self.num_layers, x.shape[0], self.hidden_size], device = x.device), 
            torch.zeros([self.num_layers, x.shape[0], self.hidden_size], device = x.device)
        )
        output, (h, c) = self.rnn(x, initial_state)
        x = output[:, -1]
        return self.fc_mu(x), self.fc_logvar(x), y

class CVAE(nn.Module):
    def __init__(self, input_shape, layer_sizes, latent_size, ts_len, num_layers, decoder_proportion, layer_kwargs={}, *args, **kwargs):
        super(CVAE, self).__init__()
        self.input_shape = (input_shape[0]-ts_len, input_shape[1])
        #self.layer_sizes = [prod(input_shape), *layer_sizes, latent_size]
        #new code
        self.layer_sizes = [input_shape[1], *layer_sizes, latent_size]
        self.encoder = VaeEncoder(self.layer_sizes, ts_len, num_layers, **layer_kwargs)
        dec_layer_sizes = [x//decoder_proportion for x in layer_sizes[::-1]]
        self.dec_layer_sizes = [[ts_len, input_shape[1], latent_size], *dec_layer_sizes, self.input_shape]
        self.decoder = VaeRNNDecoder(self.dec_layer_sizes, num_layers, output_shape = self.input_shape, **layer_kwargs)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        self.mu, self.logvar, y = self.encoder(x)
        z = self.reparameterize(self.mu, self.logvar)
        z = z.unsqueeze(dim = -2)
        z = z.repeat(1, y.shape[-2], 1)
        z = torch.cat((y, z), -1)
        #z = torch.cat((y.view(-1, prod(y.shape[1:])), z), 1) outdated
        return torch.cat((y,self.decoder(z)),1)

    def loss_function(self, recon_x, x, **kwargs):
        recon_loss = F.cross_entropy(recon_x.transpose(1,2), x.transpose(1,2), reduction='none')
        # change contribution weight of ts to loss - for some weird reason the model does not train with mean readuction
        #recon_loss = torch.mean((recon_loss[:,:kwargs.get('ts_len',13)] * kwargs.get('ts_weight',1))) + torch.mean(recon_loss[:,kwargs.get('ts_len',13):])
        # recon_loss = torch.sum((recon_loss[:,:kwargs.get('ts_len',13)] * kwargs.get('ts_weight',1))) + torch.sum(recon_loss[:,kwargs.get('ts_len',13):])
        recon_loss = torch.sum(recon_loss)

        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # kld_loss = torch.mean(-0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp(), dim = 1), dim = 0)
        kld_loss = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        adj_kld = kwargs.get('beta',1) * kld_loss
        return {'loss': recon_loss + adj_kld, 'recon_loss': recon_loss, 'kld_loss': kld_loss, 'adj_kld': adj_kld}