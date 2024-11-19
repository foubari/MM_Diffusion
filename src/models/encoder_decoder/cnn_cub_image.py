import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import Constants

# Constants
imgChans = 3
fBase = 64


# ResNet Block specifications

def actvn(x):
    out = torch.nn.functional.leaky_relu(x, 2e-1)
    return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


# Classes
class Enc(nn.Module):
    """Encoder for images with full covariance matrix."""
    def __init__(self, latent_dim=16, dist = 'Normal'):
        super().__init__()
        self.dist = dist
        s0 = self.s0 = 2
        nf = self.nf = 80  # Reduced to decrease parameter count
        nf_max = self.nf_max = 320
        size = 64

        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2 ** nlayers)

        # Single pathway for z
        blocks = [ResnetBlock(nf, nf)]
        for i in range(nlayers):
            nf0_i = min(nf * 2 ** i, nf_max)
            nf1_i = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0_i, nf1_i),
            ]

        self.conv_img = nn.Conv2d(3, nf, 3, 1, 1)
        self.resnet = nn.Sequential(*blocks)
        self.fc_mu = nn.Linear(self.nf0 * s0 * s0, latent_dim)
        self.fc_logvar = nn.Linear(self.nf0 * s0 * s0, latent_dim)

    def forward(self, x):
        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        
        if self.dist == 'Normal':
            return mu, F.softplus(logvar) + Constants.eta
        else:
            return  mu, F.softmax(logvar, dim=-1)* logvar.size(-1) + Constants.eta


class Dec(nn.Module):
    """Decoder for images from latent space."""
    def __init__(self, latent_dim=16):
        super().__init__()
        s0 = self.s0 = 2
        nf = self.nf = 80  # Match nf with the encoder
        nf_max = self.nf_max = 320
        size = 64

        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2 ** nlayers)
        self.fc = nn.Linear(latent_dim, self.nf0 * s0 * s0)

        blocks = []
        for i in range(nlayers):
            nf0_i = min(nf * 2 ** (nlayers - i), nf_max)
            nf1_i = min(nf * 2 ** (nlayers - i - 1), nf_max)
            blocks += [
                ResnetBlock(nf0_i, nf1_i),
                nn.Upsample(scale_factor=2)
            ]
        blocks += [ResnetBlock(nf, nf)]
        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 3, 3, 1, 1)

    def forward(self, z):
        out = self.fc(z).view(-1, self.nf0, self.s0, self.s0)
        out = self.resnet(out)
        out = self.conv_img(actvn(out))
        return out, torch.tensor(0.01).to(z.device) 