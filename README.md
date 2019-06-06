# Vector Quantized Variational Autoencoder

This is a PyTorch implementation of the vector quantized variational autoencoder (https://arxiv.org/abs/1711.00937). 

You can find the author's [original implementation in Tensorflow here](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py) with [an example you can run in a Jupyter notebook](https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb).

## Installing Dependencies

To install dependencies, create a conda or virtual environment with Python 3 and then run `pip install -r requirements.txt`.

## Running the VQ VAE

To run the VQ-VAE simply run `python3 main.py`. You can also add parameters in the command line. The default values are specified below:

```python
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
```

## Models

The VQ VAE has the following fundamental model components:

1. An `Encoder` class which defines the map `x -> z_e`
2. A `VectorQuantizer` class which transform the encoder output into a discrete one-hot vector that is the index of the closest embedding vector `z_e -> z_q`
3. A `Decoder` class which defines the map `z_q -> x_hat` and reconstructs the original image

The Encoder / Decoder classes are convolutional and inverse convolutional stacks, which include Residual blocks in their architecture [see ResNet paper](https://arxiv.org/abs/1512.03385). The residual models are defined by the `ResidualLayer` and `ResidualStack` classes.

These components are organized in the following folder structure:

```
models/
    - decoder.py -> Decoder
    - encoder.py -> Encoder
    - quantizer.py -> VectorQuantizer
    - residual.py -> ResidualLayer, ResidualStack
    - vqvae.py -> VQVAE
```