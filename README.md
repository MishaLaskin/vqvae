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

## PixelCNN - Sampling from the VQ VAE latent space 

To sample from the latent space, we fit a PixelCNN over the latent pixel values `z_ij`. The trick here is recognizing that the VQ VAE maps an image to a latent space that has the same structure as a 1 channel image. For example, if you run the default VQ VAE parameters you'll RGB map images of shape `(32,32,3)` to a latent space with shape `(8,8,1)`, which is equivalent to an 8x8 grayscale image. Therefore, you can use a PixelCNN to fit a distribution over the "pixel" values of the 8x8 1-channel latent space.

To train the PixelCNN on latent representations, you first need to follow these steps:

1. Train the VQ VAE on your dataset of choice
2. Use saved VQ VAE parameters to encode your dataset and save discrete latent space representations with `np.save` API. In the `quantizer.py` this is the `min_encoding_indices` variable. 
3. Specify path to your saved latent space dataset in `utils.load_latent_block` function.
4. Run the PixelCNN script

To run the PixelCNN, simply type 

`python pixelcnn/gated_pixelcnn.py`

as well as any parameters (see the argparse statements). The default dataset is `LATENT_BLOCK` which will only work if you have trained your VQ VAE and saved the latent representations.