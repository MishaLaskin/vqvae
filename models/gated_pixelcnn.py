"""
CODE NEEDS TO REFACTORED

SECTION 1: Helper functions and GatedMaskedConv blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence


def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)



class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return F.tanh(x) * F.sigmoid(y)


class GatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10):
        super().__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual

        self.class_cond_embedding = nn.Embedding(
            n_classes, 2 * dim
        )

        kernel_shp = (kernel // 2 + 1, kernel)  # (ceil(n/2), n)
        padding_shp = (kernel // 2, kernel // 2)
        self.vert_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.horiz_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.horiz_resid = nn.Conv2d(dim, dim, 1)

        self.gate = GatedActivation()

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

    def forward(self, x_v, x_h, h):
        if self.mask_type == 'A':
            self.make_causal()

        h = self.class_cond_embedding(h)
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-1), :]
        out_v = self.gate(h_vert + h[:, :, None, None])

        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)

        out = self.gate(v2h + h_horiz + h[:, :, None, None])
        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)

        return out_v, out_h


class GatedPixelCNN(nn.Module):
    def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10):
        super().__init__()
        self.dim = dim

        # Create embedding layer to embed input
        self.embedding = nn.Embedding(input_dim, dim)

        # Building the PixelCNN layer by layer
        self.layers = nn.ModuleList()

        # Initial block with Mask-A convolution
        # Rest with Mask-B convolutions
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.layers.append(
                GatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes)
            )

        # Add the output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, input_dim, 1)
        )

        self.apply(weights_init)

    def forward(self, x, label):
        shp = x.size() + (-1, )
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, W, H)

        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label)

        return self.output_conv(x_h)

    def generate(self, label, shape=(8, 8), batch_size=64):
        param = next(self.parameters())
        x = torch.zeros(
            (batch_size, *shape),
            dtype=torch.int64, device=param.device
        )

        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x, label)
                probs = F.softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(
                    probs.multinomial(1).squeeze().data
                )
        return x


"""
SECTION 2: hyperparameters and data loaders
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from torchvision.utils import save_image
import time
import utils

BATCH_SIZE = 32
N_EPOCHS = 100
PRINT_INTERVAL = 100
ALWAYS_SAVE = True
DATASET = 'LATENT_BLOCK'  # CIFAR10 | MNIST | FashionMNIST | LATENT_BLOCK
NUM_WORKERS = 4

IMAGE_SHAPE = (8, 8)  # (32, 32) | (28, 28) | (8,8)
INPUT_DIM = 1  # 3 (RGB) | 1 (Grayscale)
K = 256*2
DIM = 64
N_LAYERS = 15
LR = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DATASET == 'LATENT_BLOCK':
    _, _, train_loader, test_loader, _ = utils.load_data_and_data_loaders('LATENT_BLOCK', BATCH_SIZE)
else:
    train_loader = torch.utils.data.DataLoader(
        eval('datasets.'+DATASET)(
            '../data/{}/'.format(DATASET), train=True, download=True,
            transform=transforms.ToTensor(),
        ), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        eval('datasets.'+DATASET)(
            '../data/{}/'.format(DATASET), train=False,
            transform=transforms.ToTensor(),
        ), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

model = GatedPixelCNN(K, DIM, N_LAYERS).to(device)
criterion = nn.CrossEntropyLoss().cuda()
opt = torch.optim.Adam(model.parameters(), lr=LR)


"""
SECTION 3: train, test, and logging
"""

GENERATE_SAMPLES = True

def train():
    train_loss = []
    for batch_idx, (x, label) in enumerate(train_loader):
        start_time = time.time()
        
        if DATASET == 'LATENT_BLOCK':
            x = (x[:, 0]).cuda()
        else:
            x = (x[:, 0] * (K-1)).long().cuda()
        label = label.cuda()
       
    
        # Train PixelCNN with images
        logits = model(x, label)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        loss = criterion(
            logits.view(-1, K),
            x.view(-1)
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss.append(loss.item())

        if (batch_idx + 1) % PRINT_INTERVAL == 0:
            print('\tIter: [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                batch_idx * len(x), len(train_loader.dataset),
                PRINT_INTERVAL * batch_idx / len(train_loader),
                np.asarray(train_loss)[-PRINT_INTERVAL:].mean(0),
                time.time() - start_time
            ))


def test():
    start_time = time.time()
    val_loss = []
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(test_loader):
            if DATASET == 'LATENT_BLOCK':
                x = (x[:, 0]).cuda()
            else:
                x = (x[:, 0] * (K-1)).long().cuda()
            label = label.cuda()

            logits = model(x, label)
            
            
            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss = criterion(
                logits.view(-1, K),
                x.view(-1)
            )
            
            val_loss.append(loss.item())

    print('Validation Completed!\tLoss: {} Time: {}'.format(
        np.asarray(val_loss).mean(0),
        time.time() - start_time
    ))
    return np.asarray(val_loss).mean(0)


def generate_samples(epoch):
    label = torch.arange(10).expand(10, 10).contiguous().view(-1)
    label = label.long().cuda()

    x_tilde = model.generate(label, shape=IMAGE_SHAPE, batch_size=100)
    
    print(x_tilde[0])
    """
    images = x_tilde.cpu().data.float() / (K - 1)
    
    save_image(
        images[:, None],
        'samples/pixelcnn_baseline_samples_{}_{}.png'.format(DATASET,str(epoch)),
        nrow=10
    )"""



BEST_LOSS = 999
LAST_SAVED = -1
for epoch in range(1, N_EPOCHS):
    print("\nEpoch {}:".format(epoch))
    train()
    cur_loss = test()

    if ALWAYS_SAVE or cur_loss <= BEST_LOSS:
        BEST_LOSS = cur_loss
        LAST_SAVED = epoch

        print("Saving model!")
        torch.save(model.state_dict(), 'models/{}_pixelcnn.pt'.format(DATASET))
    else:
        print("Not saving model! Last saved: {}".format(LAST_SAVED))
    if GENERATE_SAMPLES:
        generate_samples(epoch)
