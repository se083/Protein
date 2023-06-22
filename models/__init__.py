#!/usr/bin/env python

import models.vae
import models.mmd_vae
import models.vq_vae
import models.conditional_vae
import models.conditional_mmd_vae
import models.mlp
import models.vae_cnn
import models.mmd_vae_cnn
import models.vq_vae_cnn
import models.conditional_vae_cnn
import models.conditional_mmd_vae_cnn
import models.mlp_cnn
import models.vae_rnn
import models.mmd_vae_rnn
import models.vq_vae_rnn
import models.conditional_vae_rnn
import models.conditional_mmd_vae_rnn
import models.mlp_rnn
import models.vae_ssl
import models.conditional_vae_ssl

vae_models = {
    'VAE':vae.VAE,
    'VAE_SSL':vae_ssl.VAE,
    'CVAE':conditional_vae.CVAE,
    'CVAE_SSL':conditional_vae_ssl.CVAE,
    'MMD_VAE':mmd_vae.MMD_VAE,
    'MMD_CVAE':conditional_mmd_vae.MMD_CVAE,
    'VQ_VAE':vq_vae.VQVAE,
    'MLP':mlp.MLP,
    'CNN_VAE':vae_cnn.VAE,
    'CNN_CVAE':conditional_vae_cnn.CVAE,
    'CNN_MMD_VAE':mmd_vae_cnn.MMD_VAE,
    'CNN_MMD_CVAE':conditional_mmd_vae_cnn.MMD_CVAE,
    'CNN_VQ_VAE':vq_vae_cnn.VQVAE,
    'CNN_MLP':mlp_cnn.MLP,
    'RNN_VAE':vae_rnn.VAE,
    'RNN_CVAE':conditional_vae_rnn.CVAE,
    'RNN_MMD_VAE':mmd_vae_rnn.MMD_VAE,
    'RNN_MMD_CVAE':conditional_mmd_vae_rnn.MMD_CVAE,
    'RNN_VQ_VAE':vq_vae_rnn.VQVAE,
    'RNN_MLP':mlp_rnn.MLP
    }


