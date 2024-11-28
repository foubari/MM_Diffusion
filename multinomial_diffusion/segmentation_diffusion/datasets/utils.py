import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.colors import ListedColormap
import pandas as pd

def flat_to_square_index(i, w):
    return (i // w, i % w)


def draw_exp_zeros(N, d, seed):
    rng = np.random.default_rng(seed=seed)
    cs = rng.random(N)

    zeros = np.zeros(N,dtype=np.int64)
    a = 1/(2 - 2**(-d))
    for i, c in enumerate(cs):
        zero = d
        s = a
        while s <= c:
            s += 0.5 * a
            zero -= 1
        zeros[i] = zero
    return zeros

def draw_circle(h,w,r,thickness=0.5):
    x0,y0 = (h-1)/2, (w-1)/2
    sample = torch.zeros((h,w),dtype=torch.int64)
    for i in range(h):
        for j in range(w):
            if abs(np.sqrt((i-x0)**2 + (j-y0)**2) - r) <= thickness:
                sample[i][j] = 1
    return sample