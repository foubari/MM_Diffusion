from typing import List
import torch.nn as nn
import torch
from src.models.vae_base import VAE
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.getcwd())

from multinomial_diffusion.segmentation_diffusion.layers.layers import SegmentationUnet

class MultiModalSegmentationUnet(nn.Module):

    def __init__(self, seg_unet: SegmentationUnet, modalities_vaes: List[VAE]):

        self.seg_unet = seg_unet
        self.n_modalities = len(modalities_vaes)
        self.modalities_vaes = modalities_vaes

    def forward(self, x: torch.Tensor):
        
        print("------------------")
        print("------------------")
        print("------------------")
        print(x.shape)
        # separate x into modalities
        x_modals = separate_into_modalities(x)

        #z_modals = [_vae(_x) for _vae, _x, zip(self.modalities_vaes, x_modals)]