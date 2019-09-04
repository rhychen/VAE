# -*- coding: utf-8 -*-
'''
Convolutional VAE
'''

import sys
sys.path.append(r'C:\AI, Machine learning')

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
import time
from datetime import datetime
import zipfile
import pickle
from collections import namedtuple
from collections import defaultdict

import util

###############################
# Google Colab setup
###############################

in_colab = False
if in_colab:
    from google.colab import widgets
    
    grid = widgets.Grid(2, 4)

    # Google Drive is mounted as 'gdrive' in Google Colab
    gdrive_path = '/content/gdrive/My Drive/'
else:
    gdrive_path = 'C:/'

###############################
# Global variables
###############################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

timestamp = datetime.now()
timestr   = timestamp.strftime("%d") + timestamp.strftime("%m") +\
            timestamp.strftime("%H") + timestamp.strftime("%M")

# Results & logs directory
output_dir = 'ConvVAE_out_h256c64z64lr3e-3_' + timestr
if in_colab:
    output_dir = gdrive_path + output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Saved model directory
chkpt_dir = 'ConvVAE_chkpt_h256c64z64lr3e-3_' + timestr
if in_colab:
    chkpt_dir = gdrive_path + chkpt_dir
if not os.path.exists(chkpt_dir):
    os.makedirs(chkpt_dir)

LogMoments = namedtuple("LogMoments", ["mean", "logvar"])

Losses       = namedtuple("Losses", ["total", "recon", "KLD", "Wasserstein"])
train_losses = Losses([], [], [], [])
val_losses   = Losses([], [], [], [])

###############################
# Hyperparameters
###############################

# Dataset
batch_size = 128

# Number of units in each hidden layer (excl. latent code layer)
h_len = 256

# Length of latent code (number of units in latent code layer)
z_len = 64

# Initial scaling factor for KL loss
beta = 1

# Training
num_epochs = 400
lr         = 3e-4

###############################
# Helper functions
###############################

# Debug aids
mean_dict     = {}
logvar_dict   = {}
feature_map   = []
forward_dict  = {}
backward_dict = {}

def forward_hook(layer, feat_in, feat_out):
    if layer in forward_dict:
        forward_dict[layer]["feat_in"].append(feat_in)
        forward_dict[layer]["feat_out"].append(feat_out)
    else :
        forward_dict[layer] = {}
        forward_dict[layer]["feat_in"]  = []
        forward_dict[layer]["feat_out"] = []

def backward_hook(layer, grad_in, grad_out):
    if layer in backward_dict:
        backward_dict[layer]["grad_in"]  = grad_in
        backward_dict[layer]["grad_out"] = grad_out
    else :
        backward_dict[layer] = {}
        backward_dict[layer]["grad_in"]  = []
        backward_dict[layer]["grad_out"] = []

class PrintLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        print(x)
        return x

###############################
# Dataset
###############################
colour_img = False

archive_path = '/content/gdrive/My Drive/kkanji2_split.zip'
data_path    = '.'
train_data   = 'kkanji2_split/train'
val_data     = 'kkanji2_split/validation'
train_data   = 'C:/Datasets/test'
val_data     = 'C:/Datasets/test'

zip_ref = zipfile.ZipFile(archive_path, 'r')
zip_ref.extractall(data_path)
zip_ref.close()

# The curated CNS dataset has 3818 characters, each having 10 images (one original
# CNS image 9 augmented images), for a total of 38,180 images.
NUM_TRAIN = 79264
NUM_VAL   =  3000

start_time = time.time()

# Training set
train_dataset = dset.ImageFolder(root=train_data,
								 transform=T.ToTensor(),
								 loader=pil_l_loader
								)
train_loader = DataLoader(train_dataset,
						  batch_size=batch_size,
						  num_workers=0,
						  #sampler=ChunkSampler(NUM_TRAIN, 0, shuffle=True)
						  shuffle=True
						 )

# Validation set
val_dataset = dset.ImageFolder(root=val_data,
							   transform=T.ToTensor(),
							   loader=pil_l_loader
							  )
val_loader = DataLoader(val_dataset,
						batch_size=batch_size,
						num_workers=0,
						#sampler=ChunkSampler(NUM_VAL, start=NUM_TRAIN)
						shuffle=True
					   )

dataloader_time = time.time() - start_time
print("Time Taken dataloader: {:.2f}s".format(dataloader_time))

# Input dimensions
train_iter     = iter(train_loader)
images, labels = train_iter.next()
_, C, H, W     = images.size()
in_dim         = H * W          # Flatten

###############################
# Attention mechanism
###############################

# Self-attention mechanism
class SelfAttn(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.gamma =  nn.Parameter(torch.zeros(1))

        # 1x1 convolution layers. [2] uses out_channels = in_channels / 8
        # for f and g.
        scale = 8
        if (in_channels < scale):
            scale = 1
        self.conv_f = nn.Conv2d(in_channels, in_channels // scale, kernel_size=1)
        self.conv_g = nn.Conv2d(in_channels, in_channels // scale, kernel_size=1)
        self.conv_h = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward(self, x):
        mb_size, in_C, in_H, in_W = x.size()
        in_N = in_H * in_W
        
        f = self.conv_f(x)  # Query
        g = self.conv_g(x)  # Key
        h = self.conv_h(x)  # Value
        
        # Reshape to turn Query and Key into 2D matrices (see [1], Fig. 2):
        # Keep the channel dimension and flatten height & width into one dimension
        f = f.view(mb_size, -1, in_N)
        g = g.view(mb_size, -1, in_N)
        
        # REVISIT: Try the dot-product alternative, Eq. (4) in [1]
        s = torch.matmul(f.transpose(1,2), g)
        b = F.softmax(s, dim=1)                 # Each row sums to 1
        o = torch.matmul(h.view(mb_size, in_C, in_N), b).view(mb_size, in_C, in_H, in_W)
        
        # REVISIT: Try introducing a parameter to scale the residual connection
        # REVISIT: so the model can learn to reduce the residual contribution to
        # REVISIT: 0 if needed (initialise the scale parameter to 1). Perhaps try
        # REVISIT: y = self.gamma * o + (1 - self.gamma) * x
        # Non-local block: non-local op (attention) + residual connection [1]
        y = self.gamma * o + x
                
        return y

# Spatial Transformer Network
class STN(nn.Module):
    # For fully-connected (FC) localization-network, input_num is the number
    # of features or units. For CNN localization-network, input_num is the
    # number of channels.
    def __init__(self, loc_nn, input_num):
        super().__init__()
        
        self.loc_nn    = loc_nn
        self.input_num = input_num
        
        # Spatial transformer localization-network (FC or CNN)
        if (self.loc_nn == 'fc'):
            self.localization = nn.Sequential(
                nn.Linear(self.input_num, 32),
                nn.ReLU(True),
                nn.Linear(32, 32),
                nn.ReLU(True)
            )
        else:
            self.localization = nn.Sequential(
                nn.Conv2d(self.input_num, 8, kernel_size=8),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(8, 10, kernel_size=6),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(10, 20, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                Flatten(),
                nn.Linear(20 * 3 * 3, 32),
                nn.ReLU(True)
            )

        # "Regressor" for the 3 * 2 affine transformation matrix (the transformation parameters)
        self.fc_loc = nn.Sequential(
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[0].weight.data.zero_()
        self.fc_loc[0].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        # Transformation parameters
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # affine_grid() produces a 'flow field' in the output space. Given an affine
        # transformation matrix and a regular grid in the output space, the function
        # works out the warped sampling grid in the input space. Since the warped
        # grid doesn't necessarily have a 1-to-1 correspondence with the input pixels,
        # interpolation is required and the 'flow field' specifies the interpolation
        # for each pixel in the output.
        # grid_sample() performs the interpolation accordingly.
        grid = F.affine_grid(theta, x.size())
        xt   = F.grid_sample(x, grid)

        return xt

    def forward(self, x):
        # Spatial transform the input
        x = self.stn(x)
        return x

###############################
# Model
###############################

class ConvVAE(nn.Module):
    def __init__(self, h_len, z_len, use_attn=True):
        super().__init__()
        
        # Convolutional VAE
        self.h_len  = h_len
        self.z_len  = z_len
        
        # For 32x32 image
        self.encoder = nn.Sequential(
                           nn.Conv2d(1, 32, kernel_size=4, stride=1),
                           nn.LeakyReLU(inplace=True), # Default negative slope is 0.01
                           nn.BatchNorm2d(32),
                           nn.Conv2d(32, 64, kernel_size=3, stride=1),
                           nn.LeakyReLU(inplace=True),
                           nn.BatchNorm2d(64),
                           nn.Conv2d(64, 128, kernel_size=3, stride=1),
                           nn.LeakyReLU(inplace=True),
                           nn.BatchNorm2d(128),
                           nn.Conv2d(128, 256, kernel_size=3, stride=1),
                           nn.BatchNorm2d(256),
                           nn.LeakyReLU(inplace=True),
                           Flatten(),
                       )

        self.fc_mean = nn.Linear(self.h_len, self.z_len)
        self.fc_var  = nn.Linear(self.h_len, self.z_len)
        
        # For 32x32 image
        self.decoder = nn.Sequential(           
                   nn.Linear(self.z_len, 512),
                   nn.ReLU(inplace=True),
                   nn.BatchNorm1d(512),            
                   Unflatten(-1, 512, 1, 1),
                   nn.ConvTranspose2d(512, 128, kernel_size=5, stride=2),
                   nn.ReLU(inplace=True),
                   nn.BatchNorm2d(128),
                   nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
                   nn.ReLU(inplace=True),
                   nn.BatchNorm2d(64),
                   nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
                   nn.ReLU(inplace=True),
                   nn.BatchNorm2d(32),
                   nn.ConvTranspose2d(32, 1, kernel_size=5, stride=1),
                   nn.BatchNorm2d(1),
                   # For binarised image use sigmoid function to get the probability
                   nn.Sigmoid(),
               )

#         self.forward_hook_handle  = self.register_forward_hook(forward_hook)
#         self.backward_hook_handle = self.register_backward_hook(backward_hook)
      
    def reparameterize(self, mean, logvar):
        sd  = torch.exp(0.5 * logvar)   # Standard deviation
        # We'll assume the posterior is a multivariate Gaussian
        eps = torch.randn_like(sd)
        z   = eps.mul(sd).add(mean)
        return z
    
    def forward(self, x):
        enc    = self.encoder(x)
        mean   = self.fc_mean(enc)
        logvar = self.fc_var(enc)
        z      = self.reparameterize(mean, logvar)
        dec    = self.decoder(z)
        
        return dec, LogMoments(mean, logvar)
    
    def generate_img(self, z):
        dec = self.decoder(z)

        return dec

###############################
# Loss function
###############################

# The Evidence Lower Bound (ELBO) gives the negative loss, so
# minimising the loss maximises the ELBO
def loss_fn(x_original, x_recon, z_post, z_prior, beta=1):   
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
    KL_loss = -0.5 * torch.sum(1 + z_post.logvar - z_post.mean ** 2 - z_post.logvar.exp())
    
	# REVISIT: Try using Wasserstein distance as regulariser instead of KL divergence
    # Wasserstein distance
    # 2-Wasserstein distance for diagonal Gaussians is
    #
    #   W2 = sqrt( ||mean_1 - mean_2||^2 + ||sqrt(covar_1) - sqrt(covar_2)||^2 )
    #
    # where ||.|| is the L2 norm and Frobenius norm of the mean vectors and
    # covarance matrices respectively.
    # To avoid the outer square root we'll use the squared W2 as is commonly done.
#    mean_term  = torch.sum((z_post.mean - z_prior.mean) ** 2)
#    # Square root of diagonal matrix is just the square roots of the diagonal entries
#    diff_sqrt  = torch.sqrt(z_post.logvar.exp()) - torch.sqrt(z_prior.logvar.exp())
#    covar_term = torch.sum(diff_sqrt ** 2)
#    W2 = mean_term + covar_term
    W2 = torch.tensor(0.)
    
    total_loss = recon_loss + beta * KL_loss + W2
    
    return total_loss, recon_loss, KL_loss, W2

###############################
# Main
###############################

model     = ConvVAE(h_len, z_len).to(device)
#torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
optimiser = optim.Adam(model.parameters(), lr=lr)
# Learning rate annealing
scheduler = optim.lr_scheduler.StepLR(optimiser, step_size = 20, gamma = 0.5)
    
def train(epoch, beta):
    loss_accum     = 0
    loss_avg       = DecayAverage()
    KL_loss_avg    = DecayAverage()
    recon_loss_avg = DecayAverage()
    W2_avg         = DecayAverage()

    start_time = time.time()
    model.train()
    for batch, (x, class_idx) in enumerate(train_loader):
        z_prior = 0
        
        with torch.autograd.detect_anomaly():
            x        = x.to(device)
            recon, z = model(x)
            loss, recon_loss, KL_loss, W2 = loss_fn(x, recon, z, z_prior, beta)
            loss.backward()
            
            loss_accum += loss.item()
            loss_avg.update(loss.item())
            recon_loss_avg.update(recon_loss.item())
            KL_loss_avg.update(KL_loss.item())
            W2_avg.update(W2.item())

        optimiser.step()
        optimiser.zero_grad()

    epoch_time = time.time() - start_time
    print("Time Taken for Epoch {}: {:.2f}s".format(epoch, epoch_time))
    print('Epoch {} avg. training loss: {:.3f}'.format(epoch, loss_accum / NUM_TRAIN))
    print('         recon loss: {}, KLD = {}, W2 = {}'.format(recon_loss, KL_loss, W2))
    
    if epoch % 10 == 0:
        print("Epoch {} reconstruction:".format(epoch))
        imgs_numpy = recon.detach().to('cpu').numpy()
        fig = show_images(imgs_numpy[0:25], colour_img)
        fig.savefig('{}/train_{}.png'.format(output_dir, str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)
        
    return loss_avg.value, recon_loss_avg.value, KL_loss_avg.value, W2_avg.value

def validation(epoch, beta):
    loss_accum     = 0
    loss_avg       = RunningAverage()
    KL_loss_avg    = RunningAverage()
    recon_loss_avg = RunningAverage()
    W2_avg         = RunningAverage()

    model.eval()
    with torch.no_grad():
        for batch, (x, class_idx) in enumerate(val_loader):
            z_prior = 0

            x        = x.to(device)
            recon, z = model(x)
            loss, recon_loss, KL_loss, W2 = loss_fn(x, recon, z, z_prior, beta)
            
            loss_accum += loss.item()
            loss_avg.update(loss.item())
            recon_loss_avg.update(recon_loss.item())
            KL_loss_avg.update(KL_loss.item())
            W2_avg.update(W2.item())

        print('Epoch {} validation loss: {:.3f}'.format(epoch, loss_accum / NUM_VAL))
        print('         recon loss: {}, KLD = {}, W2 = {}'.format(recon_loss, KL_loss, W2))
            
        if epoch % 10 == 0:
            print("Epoch {} reconstruction:".format(epoch))
            imgs_numpy = recon.detach().to('cpu').numpy()
            fig = show_images(imgs_numpy[0:25], colour_img)
            fig.savefig('{}/val_{}.png'.format(output_dir, str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)

    return loss_avg.value, recon_loss_avg.value, KL_loss_avg.value, W2_avg.value


for epoch in range(1, num_epochs + 1):
    if (epoch % 15 == 0):
        beta = beta * 0.9
    
    scheduler.step()
    train_loss, train_recon_loss, train_KL_loss, train_W2 = train(epoch, beta)
    val_loss, val_recon_loss, val_KL_loss, val_W2         = validation(epoch, beta)
    
    train_losses.total.append(train_loss)
    train_losses.recon.append(train_recon_loss)
    train_losses.KLD.append(train_KL_loss)
    train_losses.Wasserstein.append(train_W2)
    val_losses.total.append(val_loss)
    val_losses.recon.append(val_recon_loss)
    val_losses.KLD.append(val_KL_loss)    
    val_losses.Wasserstein.append(val_W2)

    # Plot losses
    if in_colab:
        plot_loss(grid, 1, epoch, train_losses, val_losses)

    if epoch % 50 == 0:
        # Save checkpoints
        torch.save({
            'model'      : model.state_dict(),
            'optimiser'  : optimiser.state_dict(),
            'hyperparams': {'h_len'      : h_len,
                            'z_len'      : z_len,
                            'num_epochs' : num_epochs}
        }, '{}/epoch_{}.pth'.format(chkpt_dir, epoch))

torch.save({
    'model'      : model.state_dict(),
    'optimiser'  : optimiser.state_dict(),
    'hyperparams': {'h_len'      : h_len,
                    'z_len'      : z_len,
                    'num_epochs' : num_epochs}
}, '{}/epoch_{}.pth'.format(chkpt_dir, num_epochs))
