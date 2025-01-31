import sys
sys.path.append("drive/MyDrive/MM_Diffusion/multinomial_diffusion")

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

#Constraint
from constraint import no_constraint
from constraint import dummy_score_white, dummy_score_black
from constraint import test_score
from constraint import circle_constraint


###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--samples', type=int, default=64)
parser.add_argument('--nrow', type=int, default=8)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--double', type=eval, default=False)
parser.add_argument('--out_dir', type=str, default=None)
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


""" path_samples = 'samples/sample_ep{}_s{}.png'.format(checkpoint['current_epoch'], eval_args.seed)
if not os.path.exists(os.path.dirname(path_samples)):
    os.mkdir(os.path.dirname(path_samples)) """

#deciding whether to put a constraint or not: 
#ToDo
constraint = None
chain_samples = eval_args.samples

with torch.no_grad():
    samples_chain, starts = model.sample_chain_start(chain_samples, constraint=constraint)

print('shape:, ', samples_chain.shape)
print('shape starts:', starts.shape)

#Custom grey-level colormap for plotting
K = model.num_classes
gray_shades = [(1/(K-1) * k, 1/(K-1) * k, 1/(K-1) * k) for k in range(K)]
gcmap = ListedColormap(gray_shades)

""" T = len(samples_chain)
h = 100
N = T // h
fig, axs = plt.subplots(2, N+1, figsize = ((N+1) * 4, 2 * 4 + 1))
for i in range(N+1):
    axs[0, i].imshow(samples_chain[max(i * h - 1, 0)][0][0].cpu().numpy(), cmap = gcmap, vmin=0, vmax=K-1)
    axs[0, i].axis('off')
    axs[0, i].set_title('t = {}'.format(max(i * h - 1, 0)+1))

    axs[1, i].imshow(starts[max(i * h - 1, 0)][0][0].cpu().numpy(), cmap = gcmap, vmin=0, vmax=K-1)
    axs[1, i].axis('off')
    axs[1, i].set_title('t = {}'.format(max(i * h - 1, 0)+1))
plt.savefig('chain_sample_start.png')
 """

""" fig, axs = plt.subplots(2, N, figsize = (N * 4, 2 * 4 + 1))
for i in range(start, end+1):
    axs[0, i-start].imshow(samples_chain[i][1][0].cpu().numpy(), cmap = gcmap, vmin=0, vmax=K-1)
    axs[0, i-start].axis('off') 
    axs[0, i-start].set_title('t = {}'.format(i+1))

    axs[1, i-start].imshow(starts[i][1][0].cpu().numpy(), cmap = gcmap, vmin=0, vmax=K-1)
    axs[1, i-start].axis('off')
    axs[1, i-start].set_title('t = {}'.format(i+1))
plt.savefig('chain_sample_start_fsteps.png') """

start, end = 950, 999
N = end-start+1
for j in range(chain_samples):
    fig, axs = plt.subplots(2, N, figsize = (N * 4, 2 * 4 + 1))
    for i in range(start, end+1):
        axs[0, i-start].imshow(samples_chain[i][j][0].cpu().numpy(), cmap = gcmap, vmin=0, vmax=K-1)
        axs[0, i-start].axis('off') 
        axs[0, i-start].set_title('t = {}'.format(i+1))

        axs[1, i-start].imshow(starts[i][j][0].cpu().numpy(), cmap = gcmap, vmin=0, vmax=K-1)
        axs[1, i-start].axis('off')
        axs[1, i-start].set_title('t = {}'.format(i+1))
    plt.savefig(os.path.join(eval_args.out_dir, 'chain_sample_start_fsteps_{}.png'.format(j+1)))

""" #save the samples_chain in the folder with pickle
with open(path_samples[:-4] + '.pickle', 'wb') as f:
    pickle.dump(samples_chain, f) """

""" images = []
for samples_i in samples_chain:
    grid = batch_samples_to_grid(samples_i)
    images.append(grid)

images = list(reversed(images))


def chain_linspace(chain, num_steps=150, repeat_last=10):
    out = []
    for i in np.linspace(0, len(chain)-1, num_steps):
        idx = int(i)
        if idx >= len(chain):
            print('index too big')
            idx = idx - 1
        out.append(chain[idx])

    # So that the animation stalls at the final output.
    for i in range(repeat_last):
        out.append(chain[-1])
    return out


images = chain_linspace(images)

# images.extend([images[-1], images[-1], images[-1], images[-1], images[-1]])

# images = np.array(images)
# images = images[np.arange(0, len(images), 10)]


imageio.mimsave(path_samples[:-4] + '_chain.gif', images)
imageio.imsave(path_samples, images[-1])

# from pygifsicle import optimize
# optimize(path_samples[:-4] + "_chain.gif")
 """