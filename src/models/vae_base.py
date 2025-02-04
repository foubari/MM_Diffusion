# Base VAE class definition

# Imports
import torch
import torch.nn as nn
from utils import get_mean


class VAE(nn.Module):
    """
    Unimodal VAE class. M unimodal VAEs are then used to construct a mixture-of-experts multimodal VAE.
    """
    def __init__(self, prior_dist, likelihood_dist, post_dist, enc, dec, params):
        super(VAE, self).__init__()
        self.pz = prior_dist # Prior distribution class (private latent)
        self.px_z = likelihood_dist # Likelihood distribution class
        self.qz_x = post_dist # Posterior distribution class
        self.enc = enc # Encoder object
        self.dec = dec # Decoder object
        self.modelName = None # Model name : defined in subclass
        self.params = params # Parameters (i.e. args passed to the main script)
        self._pz_params = None # defined in subclass
        self._qz_x_params = None  # Parameters of posterior distributions: populated in forward



    @property
    def qz_x_params(self):
        """Get encoding distribution parameters (already adapted for the specific distribution at the end of the Encoder class)"""
        if self._qz_x_params is None:
            raise NameError("qz_x params not initalised yet!")
        return self._qz_x_params

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        # handle merging individual datasets appropriately in sub-class
        raise NotImplementedError

    def forward(self, x, K=1):
        """
        Forward function
        Returns:
            Encoding dist, latents, decoding dist

        """
        self._qz_x_params = self.enc(x) # Get encoding distribution params from encoder
        qz_x = self.qz_x(*self._qz_x_params) # Encoding distribution
        zs = qz_x.rsample(torch.Size([K])) # K-sample reparameterization trick
        px_z = self.px_z(*self.dec(zs)) # Get decoding distribution
        return qz_x, px_z, zs

    def generate(self, N, K=1):
        self.eval()
        with torch.no_grad():
            pz = self.pz(*self.pz_params)
            latents = pz.sample(torch.Size([N]))
            px_z = self.px_z(*self.dec(latents))
            data = px_z.sample(torch.Size([K]))
        return data#data.view(-1, *data.size()[3:])
    
    def reconstruct(self, data):
        """
        Test-time reconstruction.
        """
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(data))
            latents = qz_x.rsample(torch.Size([1]))  # no dim expansion
            px_z = self.px_z(*self.dec(latents))
            recon = get_mean(px_z)
        return recon

class CatVAE(nn.Module):
    """
    Unimodal CatVAE class. M unimodal CatVAEs are then used to construct a mixture-of-experts multimodal CatVAE.
    """

    ### This VAE class is for handling categorical data
    ### The differences are 1) an embedding layer in the encoder 2) a softmax at the end of the decoder

    def __init__(self, prior_dist, likelihood_dist, post_dist, enc, dec, params):
        super(CatVAE, self).__init__()
        self.pz = prior_dist # Prior distribution class (private latent)
        self.px_z = likelihood_dist # Likelihood distribution class
        self.qz_x = post_dist # Posterior distribution class
        self.enc = enc # Encoder object
        self.dec = dec # Decoder object
        self.modelName = None # Model name : defined in subclass
        self.params = params # Parameters (i.e. args passed to the main script)
        self._pz_params = None # defined in subclass
        self._qz_x_params = None  # Parameters of posterior distributions: populated in forward

        total_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())
        print(f"Total number of parameters: {total_params}")

    @property
    def qz_x_params(self):
        """Get encoding distribution parameters (already adapted for the specific distribution at the end of the Encoder class)"""
        if self._qz_x_params is None:
            raise NameError("qz_x params not initalised yet!")
        return self._qz_x_params

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        # handle merging individual datasets appropriately in sub-class
        raise NotImplementedError

    def forward(self, x, K=1):
        """
        Forward function
        Returns:
            Encoding dist, latents, decoding dist

        """

        if K != 1:
            raise NotImplementedError(f"Only sample size K=1 is currently supported")

        self._qz_x_params = self.enc(x) # Get encoding distribution params from encoder
        qz_x = self.qz_x(*self._qz_x_params) # Encoding distribution
        zs = qz_x.rsample(torch.Size([K])) # K-sample reparameterization trick
        px_z = self.px_z(logits=self.dec(zs)) # Get decoding distribution
        return qz_x, px_z, zs

    def encode(self, x):
        return self.enc(x)

    def get_encoding_dist(self):
        return self.qz_x

    def generate(self, N, K=1):
        self.eval()
        with torch.no_grad():
            pz = self.pz(*self.pz_params)
            latents = pz.sample(torch.Size([N]))
            
            decs = self.dec(latents)
            px_z = self.px_z(logits=decs)
            data = px_z.sample()
            #data = decs.argmax(dim=-1)
        return data#data.view(-1, *data.size()[3:])
    
    def reconstruct(self, data):
        """
        Test-time reconstruction.
        """
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(data))
            latents = qz_x.rsample(torch.Size([1]))  # no dim expansion
            px_z = self.px_z(*self.dec(latents))
            recon = get_mean(px_z)
        return recon