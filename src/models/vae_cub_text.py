import os
import json
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from utils import Constants
from .vae_base import VAE
from .encoder_decoder.cnn_cub_text import Enc, Dec



# Constants
maxSentLen = 32  # max length of any description for birds dataset
minOccur = 3
embeddingDim = 128
lenWindow = 3
fBase = 32
vocabSize = 1590


class CUB_Sentence(VAE):
    """ Unimodal VAE subclass for Text modality CUBICC experiment """

    def __init__(self, params):
        super(CUB_Sentence, self).__init__(
            prior_dist=dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,      # prior
            likelihood_dist=dist.OneHotCategorical,                                             # likelihood
            post_dist=dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,       # posterior
            enc=Enc(params.latent_dim, dist=params.priorposterior),      # Encoder model
            dec=Dec(params.latent_dim),                                   # Decoder model
            params=params)        
        # Params (args passed to main)
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=True)  # It is important that this log-variance vector is learnable (see paper)
        ])

        self.modelName = 'cub text'
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






    