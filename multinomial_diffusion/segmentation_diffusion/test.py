#a test file to test some implemented functions
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import numpy as np
import torch
import SimpleITK as sitk
from torchvision.utils import make_grid

gray_shades = [(0.0, 0.0, 0.0),   # Black
            (0.33, 0.33, 0.33),  # Dark Gray
            (0.67, 0.67, 0.67),  # Light Gray
            (1.0, 1.0, 1.0)]     # White
gcmap = ListedColormap(gray_shades)

""" from constraint import dummy_score

#a fully black binary image
log_x = torch.zeros((1, 2, 3, 3), dtype=float)
log_x[:,1,...]=-1


#calculate the dummy score of this image
scores = dummy_score(log_x,1)

print(scores[0]) """

chain_samples = 5
batch = torch.randint(0, 4, (5, 1, 320, 320))
grid = make_grid(
    batch, nrow=5, padding=2, normalize=False, pad_value = 3)

grid = grid.permute(1, 2, 0)
grid = grid[:, :, 0] #only take one channel because make_grid make copies when dealing with single-channel images


# update .png

#plotting parameters
nsamples = 10
h_on_w = grid.shape[0] / grid.shape[1]
height_inches = 6
width_inches = height_inches / h_on_w * 11/10 #add a margin to not overlap subplots

#plot
fig, axes = plt.subplots(nsamples, 1, figsize=(width_inches, height_inches * nsamples))
print(height_inches, h_on_w, width_inches, width_inches * nsamples)
titles = ['Epoch {}'.format(i+1) for i in range(nsamples)]
for i, (ax, title) in enumerate(zip(axes, titles)):
    ax.imshow(grid, cmap = gcmap, vmin=0, vmax=3)
    ax.set_title(title)

plt.tight_layout()
plt.savefig('test_grid')