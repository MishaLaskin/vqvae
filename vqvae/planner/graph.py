
import os
import torch
import argparse
from vqvae.models.vqvae import VQVAE
from vqvae import utils
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
                 model_filename='must_specify_this',
                 data_file_path='must_specify_this',
                 model_dir=None,
                 dataset_name='POINTMASS',
                 min_rep_count=100,
                 batch_size=256,
                 path_length=100):
        self.model_filename = model_filename
        self.data_file_path = data_file_path
        self.model_dir = model_dir
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
        self._load_state_data()
        print('encoding data into latent representations')
        self.rep_dict, self.rep_to_state, self.hash_to_rep = self._encode_data()
        print('building representation transition graph')
        self.graph = self._build_graph()

    def _load_model(self):
        path = os.getcwd() + '/results/' if self.model_dir is None else self.model_dir

        if torch.cuda.is_available():
            data = torch.load(path + self.model_filename)
        else:
            data = torch.load(path+self.model_filename,
                              map_location=lambda storage, loc: storage)

        params = data["hyperparameters"]

        model = VQVAE(params['n_hiddens'], params['n_residual_hiddens'],
                      params['n_residual_layers'], params['n_embeddings'],
                      params['embedding_dim'], params['beta']).to(self.device)

        model.load_state_dict(data['model'])

        self.model = model
        self.model_params = data["hyperparameters"]

    def _load_state_data(self):
        train, val = utils.load_point_mass(
            data_file_path=self.data_file_path, state_version=True)
        self.state_train_loader, self.state_val_loader = utils.data_loaders(
            train, val, self.batch_size, shuffle=False)

    def _load_data(self):
        tr_data, val_data, _, _, _ = utils.load_data_and_data_loaders(
            self.dataset_name, self.data_file_path, self.batch_size)
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
        rep_to_state = {}
        hash_to_rep = {}
        # iterate over all data - state and image data are ordered in the same way

        for i, (data, state_data) in tqdm(enumerate(iter(zip(self.train_loader, self.state_train_loader)))):
            _, _, _, e_indices = self._encode_one_batch(data)
            # 64 because the encoding should be z.shape = (8,8,n_embeddings)
            n = int(len(e_indices)/64)
            for j in range(n):
                k = e_indices[64*j:64*j+64].squeeze().cpu().detach().numpy()
                hash_ = hash(tuple(k))

                if hash_ not in d:
                    d[hash_] = 1
                else:
                    d[hash_] += 1

                if hash_ not in rep_to_state:
                    rep_to_state[hash_] = state_data[0][j]

                if hash_ not in hash_to_rep:
                    hash_to_rep[hash_] = k

        rep_dict = dict((k, v)
                        for k, v in d.items() if v >= self.min_rep_count)

        return rep_dict, rep_to_state, hash_to_rep

    def _encode_one_batch(self, data_batch):
        x, _ = data_batch
        x = torch.tensor(x).float().to(self.device)

        vq_encoder_output = self.model.pre_quantization_conv(
            self.model.encoder(x))
        _, z_q, _, _, e_indices = self.model.vector_quantization(
            vq_encoder_output)

        x_recon = self.model.decoder(z_q)
        return x, x_recon, z_q, e_indices

    def _build_graph(self, min_rep_count=-1):

        rep_dict = dict((k, v)
                        for k, v in self.rep_dict.items() if v >= min_rep_count)

        graph = {k: set([]) for k in rep_dict.keys()}

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
                if hash_ != last_hash_ and hash_ in rep_dict and last_hash_ in rep_dict:
                    graph[last_hash_].add(hash_)

        return graph

    def return_shortest_path(self, graph, start, end):

        queue = [(start, [start])]
        visited = set()

        while queue:
            vertex, path = queue.pop(0)
            visited.add(vertex)
            for node in graph[vertex]:
                if node == end:
                    return path + [end]
                else:
                    if node not in visited:
                        visited.add(node)
                        queue.append((node, path + [node]))

        #print('no path between',start,'and',end)
        return False


def return_shortest_path(graph, start, end):

    queue = [(start, [start])]
    visited = set()

    while queue:
        vertex, path = queue.pop(0)
        visited.add(vertex)
        for node in graph[vertex]:
            if node == end:
                return path + [end]
            else:
                if node not in visited:
                    visited.add(node)
                    queue.append((node, path + [node]))

    return False


if __name__ == "__main__":
    graph = RepresentationGraph()
    print('N reps', len(graph.rep_dict.keys()))
    print('Total count', np.sum(list(graph.rep_dict.values())), '/ 2,000,000')
