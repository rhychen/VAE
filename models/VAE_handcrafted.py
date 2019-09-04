# -*- coding: utf-8 -*-
"""
Vanilla VAE with handcrafted neural network model (i.e. not using torch.nn) as an exercise.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision
import torchvision.transforms as T
import torchvision.datasets as dset

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Output directory
output_dir = 'test/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
            
###############################
# Hyperparameters
###############################

# Dataset
NUM_TRAIN  = 50000
NUM_VAL    = 5000
batch_size = 128

# Dimension of hidden layers (excl. latent code)
h_dim = 120

# Dimension of latent code
z_dim = 100

# Training
num_epochs = 100
lr         = 3e-4

###############################
# Helper functions
###############################
        
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def show_images(images):
    images  = np.reshape(images, [images.shape[0], -1])
    sqrtn   = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))
    
    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs  = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    plt.show()
    
    return fig

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def count_params(model):
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(p.size()) for p in model.parameters()])
    return param_count

answers = dict(np.load('C:/Convolutional Neural Networks for Visual Recognition, CS231n Stanford/Assignments/Assignment 3/gan-checks-tf.npz'))

###############################
# Dataset
###############################
# MNIST dataset: 60,000 training, 10,000 test images. Each picture contains a centered image of white digit on black
# background (0 through 9).
## To simplify our code here, we will use the PyTorch MNIST wrapper, which downloads and loads the MNIST dataset. See
# the documentation for more information about the interface. The default parameters will take 5,000 of the training
# examples and place them into a validation dataset. The data will be saved into a folder called MNIST_data.

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


mnist_train = dset.MNIST('C:/datasets/MNIST', train=True, download=True,
                         transform=T.ToTensor())
loader_train = DataLoader(mnist_train, batch_size=batch_size,
                          sampler=ChunkSampler(NUM_TRAIN, 0))

mnist_val = dset.MNIST('C:/datasets/MNIST', train=True, download=True,
                       transform=T.ToTensor())
loader_val = DataLoader(mnist_val, batch_size=batch_size,
                        sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))

# Input dimensions
train_iter     = iter(loader_train)
images, labels = train_iter.next()
_, C, H, W     = images.size()
in_dim         = H * W          # Flatten

###############################
# Initialisation
###############################

# Note this section is largely unnecessary but just for fun.
# There is the torch.nn.init.xavier_normal_() so we don't actually
# need to write our own. Also, nn.linear by default uses Kaiming He's
# initialization.

# Kaiming He et al.'s modified Xavier initialization
def xavier_init(in_dim, h_dim):    
    stddev = np.sqrt(2 / in_dim)
    # Sample from standard normal distribution
    return (torch.randn(in_dim, h_dim) * stddev).requires_grad_()

# Input layer weights and bias
w_xh = xavier_init(in_dim, h_dim)
b_xh = torch.zeros(h_dim, requires_grad=True)

# Encoder weights and bias
w_z_mean   = xavier_init(h_dim, z_dim)
b_z_mean   = torch.zeros(z_dim, requires_grad=True)
w_z_logvar = xavier_init(h_dim, z_dim)
b_z_logvar = torch.zeros(z_dim, requires_grad=True)

# Decoder weights and bias
w_zh = xavier_init(z_dim, h_dim)
b_zh = torch.zeros(h_dim, requires_grad=True)

w_hx = xavier_init(h_dim, in_dim)
b_hx = torch.zeros(in_dim, requires_grad=True)

###############################
# Encoder
###############################

# Reparameterisation trick
def reparameterize(mean, logvar):
    sd  = torch.exp(0.5 * logvar)   # Standard deviation
    # We'll assume a Gaussian posterior
    eps = torch.randn_like(sd)      
    z   = eps.mul(sd).add(mean)
    return z

# Takes dataset as input x, and outputs the latent code z
def encoder(x):
    h      = F.relu(torch.matmul(x, w_xh).add(b_xh))
    mean   = torch.matmul(h, w_z_mean).add(b_z_mean)
    logvar = torch.matmul(h, w_z_logvar).add(b_z_logvar)
    z      = reparameterize(mean, logvar)
    return z, mean, logvar

###############################
# Decoder
###############################

def decoder(z):
    h = F.relu(torch.matmul(z, w_zh).add(b_zh))
    x = torch.sigmoid(torch.matmul(h, w_hx).add(b_hx))
    return x

###############################
# Loss function
###############################

# The Evidence Lower Bound (ELBO) gives the negative loss, so
# minimising the loss maximises the ELBO
def loss_fn(x_original, x_recon, mean, logvar):
    # Reconstruction loss
    # Each pixel is a Bernoulli variable (black and white image), so we use
    # binary cross entropy. For each batch we sum the losses from every image
    # in that batch.
    recon_loss = F.binary_cross_entropy(x_recon, x_original, reduction='sum')

    # KL divergence has the following closed-form solution if we assume a
    # Gaussian prior and posterior:
    # -0.5 * sum(1 + log(sd ** 2) - mean ** 2 - sd ** 2)
    # Derivation in Appendix B of VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp())

    return recon_loss + KLD

###############################
# Training
###############################

parameters = [w_xh, b_xh, w_z_mean, b_z_mean, w_z_logvar, b_z_logvar,
              w_zh, b_zh, w_hx, b_hx]
optimizer  = optim.Adam(parameters, lr=lr)

for epoch in range(num_epochs):
    for batch_idx, (x, _) in enumerate(loader_train):
        # Reshape (flatten) input images
        x_flat = x.view(x.size(0), -1).to(device)
    
        # Generate latent code and statistics
        z, mean, logvar = encoder(x_flat)
        
        # Generate sample
        x_recon = decoder(z)
        
        # Compute the loss
        loss = loss_fn(x_flat, x_recon, mean, logvar)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

                
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}, {}]\tLoss: {:.6f}'.format(
                  epoch, batch_idx, batch_idx * len(x), loss.item() / len(x)))
            
    if epoch % 10 == 0:
        with torch.no_grad():
            # Generate new samples by sampling from the prior distribution
            z_gaussian = torch.randn(128, 100)
            sample     = decoder(z_gaussian).detach()
            imgs_numpy = sample.cpu().numpy()
            fig = show_images(imgs_numpy[0:16])
            # Save image to disk
            fig.savefig('{}/{}.png'.format(output_dir, str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)


with torch.no_grad():
    sample = torch.randn(128, 100)
    sample = decoder(sample)
    imgs_numpy = sample.cpu().numpy()
    show_images(imgs_numpy[0:16])



