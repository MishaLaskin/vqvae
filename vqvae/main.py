import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import vqvae.utils as utils
from tqdm import tqdm
from models.vqvae import VQVAE, TemporalVQVAE

parser = argparse.ArgumentParser()

"""
Hyperparameters
"""
timestamp = utils.readable_timestamp()

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_updates", type=int, default=int(1e5))
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=2)
parser.add_argument("--n_embeddings", type=int, default=128)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("--dataset",  type=str, default='PUSHER')

# whether or not to save model
parser.add_argument("-save", action="store_true")
parser.add_argument("-temporal", action="store_true")
parser.add_argument("--filename",  type=str, default='just_place_aug1')
parser.add_argument("--data_file_path", type=str,
                    default='/home/misha/research/vqvae/data/just_place_length100_paths_400.npy')
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'vqvae_temporal' if args.temporal else 'vqvae'
saved_name = model_name + '_data_' + args.filename + '_ne' + \
    str(args.n_embeddings) + 'nd' + str(args.embedding_dim) + '.pth'
if args.save:

    print('Results will be saved in ./results/'+saved_name)

"""
Load data and define batch data loaders
"""

training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
    args.dataset, args.data_file_path, args.batch_size, make_temporal=args.temporal)
"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

if args.temporal:
    model = TemporalVQVAE(args.n_hiddens, args.n_residual_hiddens,
                          args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).to(device)
else:
    model = VQVAE(args.n_hiddens, args.n_residual_hiddens,
                  args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).to(device)

"""
Set up optimizer and training loop
"""
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

model.train()

if args.temporal:
    results = {
        'n_updates': 0,
        'recon_errors': [],
        'loss_vals': [],
        'perplexities': [],
        'temporal_loss': []
    }
else:
    results = {
        'n_updates': 0,
        'recon_errors': [],
        'loss_vals': [],
        'perplexities': [],
    }


def train_vqvae():

    for i in tqdm(range(args.n_updates)):
        (x, _) = next(iter(training_loader))
        x = x.to(device)
        optimizer.zero_grad()

        embedding_loss, x_hat, perplexity = model(x)
        recon_loss = 100.0 * torch.mean((x_hat - x)**2) / x_train_var
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

        if i % args.log_interval == 0:
            """
            save model and print values
            """
            if args.save:
                hyperparameters = args.__dict__
                utils.save_model_and_results(
                    model, results, hyperparameters, saved_name)

            print('Update #', i, 'Recon Error:',
                  np.mean(results["recon_errors"][-args.log_interval:]),
                  'Loss', np.mean(results["loss_vals"][-args.log_interval:]),
                  'Perplexity:', np.mean(results["perplexities"][-args.log_interval:]))


def train_temporal_vqvae():

    for i in tqdm(range(args.n_updates)):

        (x, x2, _) = next(iter(training_loader))
        (x3, _, _) = next(iter(training_loader))

        x = x.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        optimizer.zero_grad()

        embedding_loss, temporal_loss, x_hat, perplexity = model(x, x2, x3)
        recon_loss = 100.0 * torch.mean((x_hat - x)**2) / x_train_var
        loss = recon_loss + embedding_loss + temporal_loss

        loss.backward()
        optimizer.step()

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i
        results["temporal_loss"].append(temporal_loss.cpu().detach().numpy())

        if i % args.log_interval == 0:
            """
            save model and print values
            """
            if args.save:
                hyperparameters = args.__dict__
                utils.save_model_and_results(
                    model, results, hyperparameters, saved_name)

            print('Update #', i, 'Recon Error:',
                  np.mean(results["recon_errors"][-args.log_interval:]),
                  'Loss', np.mean(results["loss_vals"][-args.log_interval:]),
                  'Temporal Loss', np.mean(
                      results["temporal_loss"][-args.log_interval:]),
                  'Perplexity:', np.mean(results["perplexities"][-args.log_interval:]))


if __name__ == "__main__":
    if args.temporal:
        train_temporal_vqvae()
    else:
        train_vqvae()
