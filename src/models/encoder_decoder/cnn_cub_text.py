import torch
from torch import nn
import torch.nn.functional as F
from utils import Constants


# Constants
maxSentLen = 32  # max length of any description for birds dataset
minOccur = 3
embeddingDim = 128
lenWindow = 3
fBase = 32
vocabSize = 1590


class Enc(nn.Module):
    """Generate latent parameters for sentence data with a single latent space."""

    def __init__(self, latent_dim=16, dist = 'Normal'):
        super(Enc, self).__init__()
        self.dist = dist
        self.embedding = nn.Linear(vocabSize, embeddingDim)

        fBase = 32  # Base number of filters
        nf = fBase
        nf_mult = 1.5  # Increase filters to balance parameter count

        # Single encoding pathway
        self.enc = nn.Sequential(
            # Input size: 1 x 32 x 128
            nn.Conv2d(1, int(nf * nf_mult), 4, 2, 1, bias=True),
            nn.BatchNorm2d(int(nf * nf_mult)),
            nn.ReLU(True),
            # Size: (fBase * nf_mult) x 16 x 64
            nn.Conv2d(int(nf * nf_mult), int(nf * 2 * nf_mult), 4, 2, 1, bias=True),
            nn.BatchNorm2d(int(nf * 2 * nf_mult)),
            nn.ReLU(True),
            # Size: (fBase * 2 * nf_mult) x 8 x 32
            nn.Conv2d(int(nf * 2 * nf_mult), int(nf * 4 * nf_mult), 4, 2, 1, bias=True),
            nn.BatchNorm2d(int(nf * 4 * nf_mult)),
            nn.ReLU(True),
            # Size: (fBase * 4 * nf_mult) x 4 x 16
            nn.Conv2d(int(nf * 4 * nf_mult), int(nf * 8 * nf_mult), (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(int(nf * 8 * nf_mult)),
            nn.ReLU(True),
            # Size: (fBase * 8 * nf_mult) x 4 x 8
            nn.Conv2d(int(nf * 8 * nf_mult), int(nf * 16 * nf_mult), (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(int(nf * 16 * nf_mult)),
            nn.ReLU(True),
            # Size: (fBase * 16 * nf_mult) x 4 x 4
        )

        # Compute the size of the flattened feature map
        conv_out_dim = int(fBase * 16 * nf_mult) * 4 * 4  # Assuming output size is [batch_size, channels, 4, 4]
        self.fc_mu = nn.Linear(conv_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(conv_out_dim, latent_dim)

    def forward(self, x):
        x_emb = self.embedding(x).unsqueeze(1)  # Shape: [batch_size, 1, maxSentLen, embeddingDim]
        e = self.enc(x_emb)  # Output size: [batch_size, channels, 4, 4]
        e_flat = e.view(e.size(0), -1)  # Flatten the feature map
        mu = self.fc_mu(e_flat)
        logvar = self.fc_logvar(e_flat)    
        if self.dist == 'Normal':
            return mu, F.softplus(logvar) + Constants.eta
        else:
            return  mu, F.softmax(logvar, dim=-1)* logvar.size(-1) + Constants.eta


class Dec(nn.Module):
    """Generate a sentence given a sample from the latent space."""

    def __init__(self, latent_dim=16, use_softmax = True):
        super(Dec, self).__init__()
        fBase = 32
        nf = fBase
        nf_mult = 1.0  # Reduced from 1.5 to 1.0 to decrease parameter count
        self.use_softmax = use_softmax
        # Single decoder pathway
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, int(nf * 16 * nf_mult), 4, 1, 0, bias=True),
            nn.BatchNorm2d(int(nf * 16 * nf_mult)),
            nn.ReLU(True),
            # Size: (fBase * 16 * nf_mult) x 4 x 4
            nn.ConvTranspose2d(int(nf * 16 * nf_mult), int(nf * 8 * nf_mult), (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(int(nf * 8 * nf_mult)),
            nn.ReLU(True),
            # Size: (fBase * 8 * nf_mult) x 4 x 8
            nn.ConvTranspose2d(int(nf * 8 * nf_mult), int(nf * 8 * nf_mult), 3, 1, 1, bias=True),
            nn.BatchNorm2d(int(nf * 8 * nf_mult)),
            nn.ReLU(True),
            # Size: (fBase * 8 * nf_mult) x 4 x 8
            nn.ConvTranspose2d(int(nf * 8 * nf_mult), int(nf * 4 * nf_mult), (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(int(nf * 4 * nf_mult)),
            nn.ReLU(True),
        )

        self.dec_h = nn.Sequential(
            nn.ConvTranspose2d(int(nf * 4 * nf_mult), int(nf * 4 * nf_mult), 3, 1, 1, bias=True),
            nn.BatchNorm2d(int(nf * 4 * nf_mult)),
            nn.ReLU(True),
            # Size: (fBase * 4 * nf_mult) x 8 x 32
            nn.ConvTranspose2d(int(nf * 4 * nf_mult), int(nf * 2 * nf_mult), 4, 2, 1, bias=True),
            nn.BatchNorm2d(int(nf * 2 * nf_mult)),
            nn.ReLU(True),
            # Size: (fBase * 2 * nf_mult) x 16 x 64
            nn.ConvTranspose2d(int(nf * 2 * nf_mult), int(nf * nf_mult), 4, 2, 1, bias=True),
            nn.BatchNorm2d(int(nf * nf_mult)),
            nn.ReLU(True),
            # Size: (fBase * nf_mult) x 32 x 128
            nn.ConvTranspose2d(int(nf * nf_mult), 1, 4, 2, 1, bias=True),
            nn.ReLU(True)
            # Output size: 1 x 64 x 256
        )

        # Inverts the 'embedding' module up to one-hotness
        self.toVocabSize = nn.Linear(embeddingDim, vocabSize)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)
        h = self.dec(z.view(-1, *z.size()[-3:]))
        out = self.dec_h(h)
        out = out.view(-1, embeddingDim)
        # The softmax is key for this to work
        ret = self.toVocabSize(out).view(-1, maxSentLen, vocabSize) #softmax included in F.crossentropy
        if self.use_softmax:
            ret = self.softmax(ret)
        return [ret]
    
    