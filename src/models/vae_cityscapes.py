import os
import json
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from utils import Constants
from .vae_base import CatVAE
from .encoder_decoder.cityscapes import Enc, Dec



class CS_unimodal(CatVAE):
    """ Unimodal VAE subclass Cityscapes EXP """

    def __init__(self, params):
        super(CS_unimodal, self).__init__(
            dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,         # prior
            dist.Categorical,  # likelihood
            dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,        # posterior
            enc=Enc(params.latent_dim, dist=params.priorposterior),      # Encoder model
            dec=Dec(params.latent_dim),                                   # Decoder model
            params=params)        
        # Params (args passed to main)
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=True)  # It is important that this log-variance vector is learnable (see paper)
        ])

        self.modelName = f'cityscapes m_{params.modality}'
        self.dataSize = torch.Size([1, 32, 64])
        self.params = params

    @property
    def pz_params(self):
        """

        Returns: Parameters of prior auxiliary distribution for modality-specific latent code

        """
        if self.params.priorposterior == 'Normal':
            return self._pz_params[0], F.softplus(self._pz_params[1]) + Constants.eta
        else:
            return self._pz_params[0], F.softmax(self._pz_params[1], dim=-1) * self._pz_params[1].size(-1) + Constants.eta