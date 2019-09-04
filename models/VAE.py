# -*- coding: utf-8 -*-
"""
Vanilla VAE.
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

# Number of units in each hidden layer (excl. latent code layer)
h_len = 120

# Length of latent code (number of units in latent code layer)
z_len = 100

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
    """Count the number of parameters in the current model graph """
    param_count = np.sum([np.prod(p.size()) for p in model.parameters()])
    return param_count

###############################
# Dataset
###############################

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    
    Arguments:
        num_samples: Number of desired datapoints
        start:       offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


# MNIST dataset: 60,000 training, 10,000 test images.
# We'll take NUM_VAL of the training examples and place them into a validation dataset.
NUM_TRAIN  = 55000
NUM_VAL    = 5000

# Training set
mnist_train = dset.MNIST('C:/datasets/MNIST',
                         train=True, download=True,
                         # Converts (H x W x C) image in the range [0, 255] to a
                         # torch.FloatTensor of shape (C x H x W) in the range [0, 1].
                         transform=T.ToTensor())
loader_train = DataLoader(mnist_train, batch_size=batch_size,
                          sampler=ChunkSampler(NUM_TRAIN, 0))
# Validation set
mnist_val = dset.MNIST('C:/datasets/MNIST',
                       train=True, download=True,
                       transform=T.ToTensor())
loader_val = DataLoader(mnist_val, batch_size=batch_size,
                        sampler=ChunkSampler(NUM_VAL, start=NUM_TRAIN))
# Test set
mnist_test = dset.MNIST('C:/datasets/MNIST',
                       train=False, download=True,
                       transform=T.ToTensor())
loader_test = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

# Input dimensions
train_iter     = iter(loader_train)
images, labels = train_iter.next()
_, C, H, W     = images.size()
in_dim         = H * W          # Flatten

###############################
# Model
###############################

class VAE(nn.Module):
    def __init__(self, in_dim, h_len, z_len):
        super().__init__()
                
        self.in_dim = in_dim
        self.h_len  = h_len
        self.z_len  = z_len

        # Fully connected layers
        self.fc1   = nn.Linear(self.in_dim, self.h_len)
        self.fc2_1 = nn.Linear(self.h_len, self.z_len)    # 'Mean' layer
        self.fc2_2 = nn.Linear(self.h_len, self.z_len)    # 'Log variance' layer
        self.fc3   = nn.Linear(self.z_len, self.h_len)
        self.fc4   = nn.Linear(self.h_len, self.in_dim)
                
    def encoder(self, x):
        h1     = F.relu(self.fc1(x))
        mean   = self.fc2_1(h1)
        logvar = self.fc2_2(h1)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        sd  = torch.exp(0.5 * logvar)   # Standard deviation
        # We'll assume the posterior is a multivariate Gaussian
        eps = torch.randn_like(sd)
        z   = eps.mul(sd).add(mean)
        return z
        
    def decoder(self, z):
        h2 = F.relu(self.fc3(z))
        # For binarised image use sigmoid function to get the probability
        x  = torch.sigmoid(self.fc4(h2))
        return x

    # Note this takes flattened images as input
    def forward(self, x_flat):
        mean, logvar = self.encoder(x_flat)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decoder(z)
        
        return x_recon, mean, logvar

###############################
# Loss function
###############################

# The Evidence Lower Bound (ELBO) gives the negative loss, so
# minimising the loss maximises the ELBO
def loss_fn(x_original, x_recon, mean, logvar):
    # Reconstruction loss
    # Each pixel is a Bernoulli variable (black and white image), so we use
    # binary cross entropy (BCE). For each batch we sum the losses from every image
    # in that batch. Note binary_cross_entropy() output has already been negated.
    # REVISIT: The original image's pixel value has been re-scaled to the range
    # REVISIT: [0, 1]. This rescaled value is treated here as if it's Bernoulli
    # REVISIT: i.e. {0, 1} rather than [0, 1] and fed directly to the BCE function.
    # REVISIT: However, this is not ideal, see https://arxiv.org/abs/1907.06845
    recon_loss = F.binary_cross_entropy(x_recon, x_original, reduction='sum')

    # KL divergence has the following closed-form solution if we assume a
    # Gaussian prior and posterior:
    # -( 0.5 * sum(1 + log(sd ** 2) - mean ** 2 - sd ** 2) )
    # Derivation in Appendix B of VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KL_loss = -0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp())
    
    return recon_loss + KL_loss

###############################
# Main
###############################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae       = VAE(in_dim, h_len, z_len).to(device)
optimiser = optim.Adam(vae.parameters(), lr=lr)

def train(epoch):
    vae.train()
    loss_train = 0
    for batch, (x, _) in enumerate(loader_train):
        # Reshape (flatten) input images
        x_flat = x.view(x.size(0), -1).to(device)
        
        x_recon, mean, logvar = vae(x_flat)
        loss        = loss_fn(x_flat, x_recon, mean, logvar)
        loss_train += loss.item()
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
    print('Epoch {} avg. training loss: {:.3f}'.format(epoch, loss_train / len(loader_train.dataset)))

def validation(epoch):
    vae.eval()
    loss_val = 0
    with torch.no_grad():
        for batch, (x, _) in enumerate(loader_val):
            # Reshape (flatten) input images
            x_flat = x.view(x.size(0), -1).to(device)

            x_recon, mean, logvar = vae(x_flat)
            loss      = loss_fn(x_flat, x_recon, mean, logvar)
            loss_val += loss.item()
            
    print('Epoch {} validation loss: {:.3f}'.format(epoch, loss_val / len(loader_val.dataset)))

for epoch in range(num_epochs):
    train(epoch)
    validation(epoch)
    if epoch % 10 == 0:
        with torch.no_grad():
            z      = torch.randn(128, z_len).to(device)
            # REVISIT: The decoder output produced by sigmoid function is the
            # REVISIT: mean of Bernoulli distribution for each pixel. Strictly
            # REVISIT: speaking, we should sample from this distribution to get
            # REVISIT: binarised images, but below it is treated as gray scale
            # REVISIT: pixel value in the range [0, 1] instead.
            sample = vae.decoder(z)
            imgs_numpy = sample.cpu().numpy()
            fig = show_images(imgs_numpy[0:16])
            # Save image to disk
            fig.savefig('{}/{}.png'.format(output_dir, str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)
            
        # Save checkpoints
        torch.save({
            'model'      : model.state_dict(),
            'optimiser'  : optimiser.state_dict(),
            'hyperparams': {'h_len'      : 256,
                            'z_len'      : 100,
                            'N'          : 10,
                            'num_epochs' : 100}
        }, '{}/epoch_{}.pth'.format(chkpt_dir, epoch))

