import os
import math
import torch
import numpy as np
import pickle
import argparse
import torchvision.utils as vutils
from diffusion_utils.utils import add_parent_path
import torchvision
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Data
add_parent_path(level=1)
from datasets.data import get_data, get_data_id, add_data_args, get_plot_transform

# Model
from model import get_model, get_model_id, add_model_args
from diffusion_utils.base import DataParallelDistribution


###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--originals', type=int, default=1)
parser.add_argument('--samples', type=int, default=3)
parser.add_argument('--nrow', type=int, default=8)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--double', type=eval, default=False)


add_model_args(parser)
eval_args = parser.parse_args()

path_args = '{}/args.pickle'.format(eval_args.model)
path_check = '{}/check/checkpoint.pt'.format(eval_args.model)

torch.manual_seed(eval_args.seed)

###############
## Load args ##
###############

with open(path_args, 'rb') as f:
    args = pickle.load(f)

args.loss_weighted=False

##################
## Specify data ##
##################

train_loader, _, data_shape, num_classes = get_data(args)
minibatch_data = None
for minibatch_data in train_loader:
    break
assert minibatch_data is not None

###################
## Specify model ##
###################

model = get_model(args, data_shape=data_shape)
if args.parallel == 'dp':
    model = DataParallelDistribution(model)

if torch.cuda.is_available():
    checkpoint = torch.load(path_check)
else:
    checkpoint = torch.load(path_check, map_location='cpu')
model.load_state_dict(checkpoint['model'])

if torch.cuda.is_available():
    model = model.cuda()

print('Loaded weights for model at {}/{} epochs'.format(checkpoint['current_epoch'], args.epochs))


############
## Sample ##
############

#Custom grey-level colormap for plotting
K = model.num_classes
gray_shades = [(1/(K-1) * k, 1/(K-1) * k, 1/(K-1) * k) for k in range(K)]
gcmap = ListedColormap(gray_shades)

#getting the x0s to inpaint
x0s = []
originals = eval_args.originals
for batch in train_loader:
    for i in range(batch.size(0)):
        if len(x0s) < originals:
            x0s.append(batch[i])
        else:
            break
    if len(x0s) == originals:
        break

samples = eval_args.samples
fig, axs = plt.subplots(len(x0s), samples+1, figsize=((samples+1)* 4, len(x0s) * 4 + 2))
for i, x0 in enumerate(x0s):
    x0 = x0.unsqueeze(0)
    mask = torch.where((x0 == 1) | (x0 == 2), 0, 1)
    categories = [1, 2]
    samples = eval_args.samples
    with torch.no_grad():
        samples = model.sample_masked(samples, x0, mask, categories).cpu().numpy()

    axs[i, 0].imshow(x0[0].cpu().numpy(), cmap = gcmap, vmin=0, vmax=K-1)
    axs[i, 0].axis('off')
    axs[i, 0].set_title('Original')
    for j, sample in enumerate(samples):
        axs[i, j+1].imshow(sample[0], cmap = gcmap, vmin=0, vmax=K-1)
        axs[i, j+1].axis('off')
        axs[i, j+1].set_title('Sample {}'.format(j+1))

plt.savefig('inpainted_samples.png')