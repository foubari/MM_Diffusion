from typing import List
import torch.nn as nn
import torch
from src.models.vae_base import VAE
import sys
import os
import random
from typing import Optional
from tqdm import tqdm

from multinomial_diffusion.segmentation_diffusion.layers.layers import SegmentationUnet

class MultiModalSegmentationUnet(nn.Module):
    def __init__(self, seg_unet: SegmentationUnet, modalities_vaes: List[VAE], strategy: str = "mixture_of_experts"):
        
        assert strategy in ["mixture_of_experts", "concatenation"]

        super().__init__()

        # Register the segmentation UNet
        self.seg_unet = seg_unet
        self.add_module("seg_unet", seg_unet)
        self.strategy = strategy

        # Register VAE modalities
        self.n_modalities = len(modalities_vaes)
        self.modalities_vaes = nn.ModuleList(modalities_vaes)

    def progress_bar(self, total):
        return tqdm(total=total)

    def encode(self, x: torch.Tensor):
        print("------------------")
        print("------------------")
        print("------------------")
        print(x.shape, x[:, 0, :, :].shape, type(x[:, 0, :, :]))

        if self.strategy.lower() == "mixture_of_experts":

            # encode
            modality_idx = random.randint(0, self.n_modalities-1)
            vae = self.modalities_vaes[modality_idx]
            qz_x_params = vae.encode(x[:, modal_idx, :, :])
            qz_x_dist = vae.get_encoding_dist()

            # sample
            z = qz_x_dist(*qz_params)

            return z, modality_idx

        if self.strategy.lower() == "concatenation":

            # independently encode all the modalities
            qz_params_lst = []
            qz_x_dist_lst = []
            z_lst = []
            for modal_idx, vae in enumerate(self.modalities_vaes):

                _qz_x_params = vae.encode(x[:, modal_idx, :, :])
                _qz_x_dist = vae.get_encoding_dist()
                _z = _qz_x_dist(*_qz_x_params)

                qz_params_lst.append(_qz_x_params)
                qz_x_dist_lst.append(_qz_x_dist)
                z_lst.append(_z)
            
            z = torch.concatenation(z_lst)

            print(z.shape)

            return z, None

    def generate(self, z: Optional[torch.Tensor] = None, diffusion_steps: int = 10):        
        return self.seg_unet(z)
