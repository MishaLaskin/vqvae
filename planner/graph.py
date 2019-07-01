
import os
import torch
import argparse
from models.vqvae import VQVAE
import utils
from tqdm import tqdm
from torch.utils.data import DataLoader

import numpy as np

"""
load dataset and its corresponding VQ VAE representation model

run trained VQ VAE on dataset

specify total used representations and initialize adjacency dict for graph ({hash_z: set()})

iterate through data:

hash representations

add transitions to graph

save graph as well as pointers to data and vqvae model used


"""


class RepresentationGraph:

    def __init__(self,
                 model_path='vqvae_data_point_mass_v2ne16nd16.pth',
                 dataset_name='POINTMASS',
                 min_rep_count=100,
                 batch_size=256,
                 path_length=100):
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.min_rep_count = min_rep_count
        self.batch_size = batch_size
        self.path_length = path_length
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        print('loading VQ VAE model')
        self._load_model()
        print('loading collected path data')
        self._load_data()
        print('encoding data into latent representations')
        self._encode_data()
        print('building representation transition graph')
        self._build_graph()

    def _load_model(self):
        path = os.getcwd() + '/results/'

        if torch.cuda.is_available():
            data = torch.load(path + self.model_path)
        else:
            data = torch.load(path+self.model_path,
                              map_location=lambda storage, loc: storage)

        params = data["hyperparameters"]

        model = VQVAE(params['n_hiddens'], params['n_residual_hiddens'],
                      params['n_residual_layers'], params['n_embeddings'],
                      params['embedding_dim'], params['beta']).to(self.device)

        model.load_state_dict(data['model'])

        self.model = model
        self.model_params = data["hyperparameters"]

    def _load_data(self):
        tr_data, val_data, _, _, _ = utils.load_data_and_data_loaders(
            self.dataset_name, self.batch_size)
        self.train_loader = DataLoader(tr_data,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       pin_memory=True)

        self.train_loader_over_paths = DataLoader(tr_data,
                                                  batch_size=self.path_length,
                                                  shuffle=False,
                                                  pin_memory=True)

    def _encode_data(self):

        d = {}
        # iterate over all data

        for i, data in tqdm(enumerate(iter(self.train_loader))):
            _, _, _, e_indices = self._encode_one_batch(data)

            # 64 because the encoding should be z.shape = (8,8,n_embeddings)
            n = int(len(e_indices)/64)
            for i in range(n):
                k = e_indices[64*i:64*i+64].squeeze().cpu().detach().numpy()
                hash_ = hash(tuple(k))

                if hash_ not in d:
                    d[hash_] = 1
                else:
                    d[hash_] += 1

        self.rep_dict = dict((k, v)
                             for k, v in d.items() if v >= self.min_rep_count)

    def _encode_one_batch(self, data_batch):
        x, _ = data_batch
        x = torch.tensor(x).float().to(self.device)
        x = x.to(self.device)
        vq_encoder_output = self.model.pre_quantization_conv(
            self.model.encoder(x))
        _, z_q, _, _, e_indices = self.model.vector_quantization(
            vq_encoder_output)

        x_recon = self.model.decoder(z_q)
        return x, x_recon, z_q, e_indices

    def _build_graph(self):

        self.graph = {k: set([]) for k in self.rep_dict.keys()}
        # iterate over all data

        for i, data in tqdm(enumerate(iter(self.train_loader_over_paths))):
            _, _, _, e_indices = self._encode_one_batch(data)

            k = e_indices[0:64].squeeze().cpu().detach().numpy()
            last_hash_ = hash(tuple(k))

            # 64 because the encoding should be z.shape = (8,8,n_embeddings)
            n = int(len(e_indices)/64)
            for i in range(n):
                k = e_indices[64*i:64*i+64].squeeze().cpu().detach().numpy()
                hash_ = hash(tuple(k))
                if hash_ != last_hash_:
                    self.graph[last_hash_].add(hash_)


if __name__ == "__main__":
    graph = RepresentationGraph()
    print('N reps', len(graph.rep_dict.keys()))
    print('Total count', np.sum(graph.rep_dict.values()), '/ 20,000')
