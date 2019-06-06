
import torch
import torch.nn as nn
import numpy as np
from models.encoder import Encoder
from models.quantizer import VectorQuantizer
from models.decoder import Decoder


class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

    def forward(self, x):
        x_np = x.detach().numpy()
        # print('x mean / std / shape', np.mean(x_np),
        #      np.std(x_np), x_np.shape)
        z_e = self.encoder(x)
        z_e_np = z_e.detach().numpy()
        # print('z_e mean / std / shape', np.mean(z_e_np),
        #      np.std(z_e_np), z_e_np.shape)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)
        return embedding_loss, x_hat, perplexity
