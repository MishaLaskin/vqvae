
import torch
import torch.nn as nn
import numpy as np
from vqvae.models.encoder import Encoder
from vqvae.models.quantizer import VectorQuantizer
from vqvae.models.decoder import Decoder


class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False, gpu_id=0):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta, gpu_id=gpu_id)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):

        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity


class TemporalVQVAE(VQVAE):

    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False, gpu_id=0, temporal_penalty=100.0):
        super().__init__(h_dim, res_h_dim, n_res_layers,
                         n_embeddings, embedding_dim, beta, save_img_embedding_map=False, gpu_id=0)
        self.temporal_penalty = temporal_penalty

    def forward(self, x, x2, x3, verbose=False):

        z_e = self.encoder(x)
        z_e2 = self.encoder(x2)
        z_e3 = self.encoder(x3)

        z_e = self.pre_quantization_conv(z_e)
        z_e2 = self.pre_quantization_conv(z_e2)
        z_e3 = self.pre_quantization_conv(z_e3)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e)
        x_hat = self.decoder(z_q)

        margin = 1e-6
        temporal_loss = self.temporal_penalty * \
            (torch.mean(z_e-z_e2.detach())**2 -
             torch.mean(z_e-z_e3.detach())**2 + margin)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            return z_e, z_e2, z_e3

        return embedding_loss, temporal_loss, x_hat, perplexity
