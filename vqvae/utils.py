import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
from datasets.block import BlockDataset, LatentBlockDataset
from datasets.base import ImageDataset, StateDataset
from vqvae.models.vqvae import VQVAE, TemporalVQVAE
import numpy as np


def load_model(model_filename, temporal=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        data = torch.load(model_filename)
    else:
        data = torch.load(
            model_filename, map_location=lambda storage, loc: storage)

    params = data["hyperparameters"]

    if temporal:
        model = TemporalVQVAE(params['n_hiddens'], params['n_residual_hiddens'],
                              params['n_residual_layers'], params['n_embeddings'],
                              params['embedding_dim'], params['beta']).to(device)
    else:
        model = VQVAE(params['n_hiddens'], params['n_residual_hiddens'],
                      params['n_residual_layers'], params['n_embeddings'],
                      params['embedding_dim'], params['beta']).to(device)

    model.load_state_dict(data['model'])

    return model, data


def load_cifar():
    train = datasets.CIFAR10(root="data", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

    val = datasets.CIFAR10(root="data", train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    return train, val


def load_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/randact_traj_length_100_n_trials_1000_n_contexts_1.npy'

    train = BlockDataset(data_file_path, train=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ]))

    val = BlockDataset(data_file_path, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(
                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    return train, val


def load_dm_data(data_file_path=None, state_version=False, make_temporal=False,include_goals=False):

    if data_file_path is None:
        raise ValueError('Please provide a data_file_path input string')

    if state_version:
        train = StateDataset(data_file_path, train=True)
        val = StateDataset(data_file_path, train=False)
    else:
        train = ImageDataset(data_file_path, train=True, make_temporal=make_temporal,
                            include_goals=include_goals,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))

        val = ImageDataset(data_file_path, train=False, make_temporal=False,
                            include_goals=include_goals,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))
    return train, val


def load_latent_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/latent_e_indices.npy'

    train = LatentBlockDataset(data_file_path, train=True,
                               transform=None)

    val = LatentBlockDataset(data_file_path, train=False,
                             transform=None)
    return train, val


def data_loaders(train_data, val_data, batch_size, shuffle=True):

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            pin_memory=True)
    return train_loader, val_loader


def load_data_and_data_loaders(dataset_name, data_file_path, batch_size, make_temporal=False,include_goals=False):
    if dataset_name == 'CIFAR10':
        training_data, validation_data = load_cifar()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.train_data / 255.0)

    elif dataset_name == 'BLOCK':
        training_data, validation_data = load_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data / 255.0)
    elif dataset_name == 'LATENT_BLOCK':
        training_data, validation_data = load_latent_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data)

    # DM stands from dm_control library of envs
    elif dataset_name == 'POINTMASS' or dataset_name == 'REACHER' or dataset_name == 'PUSHER':
        training_data, validation_data = load_dm_data(
            data_file_path, make_temporal=make_temporal,include_goals=include_goals)
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data)

    else:
        raise ValueError(
            'InvalFalseid dataset: only CIFAR10 and BLOCK datasets are supported.')
    return training_data, validation_data, training_loader, validation_loader, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_model_and_results(model, results, hyperparameters, saved_name):
    SAVE_MODEL_PATH = os.getcwd() + '/results/'

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,
               SAVE_MODEL_PATH + saved_name)


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

    #print('no path between',start,'and',end)
    return False
