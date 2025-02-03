import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import Constants

# Constants
dataSize = torch.Size([1, 32, 64])

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
    """ Encoder Image Cityscapes Resnet """

    def __init__(self, latent_dim=16, embedding_dim=8, dist = 'Normal'):
        super().__init__()
        self.dist = dist
        s0 = self.s0 = 8  # kwargs['s0']
        nf = self.nf = 64  # nfilter
        nf_max = self.nf_max = 1024  # nfilter_max
        size = 32

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)



        blocks_z = [
            ResnetBlock(nf, nf)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            
            blocks_z += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.num_classes = 2
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.num_classes, self.embedding_dim)        

        self.conv_img_z = nn.Conv2d(self.embedding_dim, 1 * nf, 3, padding=1)
        self.resnet_z = nn.Sequential(*blocks_z)
        self.fc_mu_z = nn.Linear(self.nf0 * 8 * 16, latent_dim) #TODO: replace 8, 16 by a more general shape
        self.fc_lv_z = nn.Linear(self.nf0 * 8 * 16, latent_dim) #TODO: replace 8, 16 by a more general shape

    def forward(self, x):

        #x_shape = x.size()[1:]

        B, C, H, W = x.size()
        x = self.embedding(x)

        assert x.shape == (B, C, H, W, self.embedding_dim)

        x = x.permute(0, 1, 4, 2, 3)

        assert x.shape == (B, C, self.embedding_dim, H, W)

        x = x.reshape(B, C * self.embedding_dim, H, W)

        out_z = self.conv_img_z(x)
        out_z = self.resnet_z(out_z)
        out_z = out_z.reshape(out_z.size()[0], self.nf0 * 8 * 16) #TODO: replace 8, 16 by a more general shape
        logvar = self.fc_lv_z(out_z)
        mu = self.fc_mu_z(out_z)
    
        if self.dist == 'Normal':
            return mu, F.softplus(logvar) + Constants.eta
        else:
            return  mu, F.softmax(logvar, dim=-1)* logvar.size(-1) + Constants.eta



class Dec(nn.Module):
    """ Decoder Cityscapes Resnet """

    def __init__(self, ndim):
        super().__init__()

        # NOTE: I've set below variables according to Kieran's suggestions
        s0 = self.s0 = 8  # kwargs['s0']
        nf = self.nf = 64  # nfilter
        nf_max = self.nf_max = 512  # nfilter_max
        size = 32

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        self.fc = nn.Linear(ndim, self.nf0*8*16) #TODO: replace 8, 16 by a more general shape

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2**(nlayers-i), nf_max)
            nf1 = min(nf * 2**(nlayers-i-1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2)
            ]

        blocks += [
            ResnetBlock(nf, nf),
        ]

        self.resnet = nn.Sequential(*blocks)
        self.num_classes = 2
        self.conv_img = nn.Conv2d(nf, self.num_classes, 3, padding=1)

    def forward(self, z):
        out = self.fc(z).reshape(-1, self.nf0, 8, 16)
        out = self.resnet(out)
        out = self.conv_img(actvn(out))
        # out = out.view(*u.size()[:2], *out.size()[1:])
        # consider also predicting the length scale
        #return out, torch.tensor(0.75).to(z.device)  # mean, length scale

        #softmax needed in the categorical case
        log_pred = F.log_softmax(out, dim=1) #(B, num_classes=2, H, W)
        log_pred = log_pred.permute(0, 2, 3, 1)
        log_pred.unsqueeze(1)
        return log_pred # torch.tensor(0.75).to(z.device)