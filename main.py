import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from utils import load_cifar, data_loaders
from models.vqvae import VQVAE

parser = argparse.ArgumentParser()

"""
Hyperparameters
"""

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_updates", type=int, default=5000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=50)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Load data and define batch data loaders
"""
training_data, validation_data = load_cifar()

training_loader, validation_loader = data_loaders(
    training_data, validation_data, args.batch_size)

x_train_var = np.var(training_data.train_data / 255.0)

"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

model = VQVAE(args.n_hiddens, args.n_residual_hiddens,
              args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).to(device)

"""
Set up optimizer and training loop
"""
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

model.train()
reconstruction_errors = []
perplexities = []
loss_vals = []


def train():
    for i in range(args.n_updates):
        (x, _) = next(iter(training_loader))
        x = x.to(device)
        optimizer.zero_grad()

        embedding_loss, x_hat, perplexity = model(x)
        recon_loss = torch.mean((x_hat - x)**2) / x_train_var
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()

        reconstruction_errors.append(recon_loss.detach().numpy())
        perplexities.append(perplexity.detach().numpy())
        loss_vals.append(loss.detach().numpy())

        if i % args.log_interval == 0:
            print('Update #', i, 'Recon Error:',
                  np.mean(reconstruction_errors[-args.log_interval:]),
                  'Loss', np.mean(loss_vals[-args.log_interval:]),
                  'Perplexity:', np.mean(perplexities[-args.log_interval:]))


if __name__ == "__main__":
    train()
