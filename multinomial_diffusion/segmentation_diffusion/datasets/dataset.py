from __future__ import print_function

import errno

import torch
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
import torchvision
from torchvision.datasets.utils import download_url
import torchvision.transforms as transforms
import torchvision.transforms.functional as F_transforms
from .utils import draw_exp_zeros, flat_to_square_index
from .utils import draw_circle

import numpy as np

import os
from PIL import Image

import pickle

HOME = '/export/fhome2/denis/bct_generation_data_and_results/multinomial_diffusion/data/'
#go to ama15 if not already there
import subprocess
hostname = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()
if not 'ama15' in hostname:
    HOME = '/net/10.215.25.15' + HOME

def fn_to_tensor(img):
    img = np.array(img)

    # Add channel to grayscale images.
    if len(img.shape) == 2:
        img = img[:, :, None]
    img = np.array(img).transpose(2, 0, 1)
    return torch.from_numpy(img)


class toTensor:
    def __init__(self):
        pass

    def __call__(self, img):
        return fn_to_tensor(img)

class Binary12(torch.utils.data.Dataset):
    def __init__(self, N=1000, p=[0.25, 0.25, 0.25, 0.25], transform=None):
        self.p = p
        self.transform = transform

        #generate the data at init
        self.data = [self.generate_sample() for _ in range(N)]
    
    def generate_sample(self):
        #generate a sample
        cat = np.random.choice([0, 1, 2, 3], p=self.p)
        if cat == 0:
            return torch.tensor([1, 0], dtype=torch.int64).reshape(1, 1, 2)
        elif cat == 1:
            return torch.tensor([0, 1], dtype=torch.int64).reshape(1, 1, 2)
        elif cat == 2:
            return torch.tensor([1, 1], dtype=torch.int64).reshape(1, 1, 2)
        else:
            return torch.tensor([0, 0], dtype=torch.int64).reshape(1, 1, 2)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

class MultiClassImagesV1(torch.utils.data.Dataset):
    def __init__(self, generate_data=False, N=100, folder_path='multi_class_images_v1', transform=None):
        
        folder_path = os.path.join(HOME, folder_path)
        #retrieve the data from the folder
        self.data = []
        if not generate_data:
            # List all files in the folder
            files = os.listdir(folder_path)

            png_files = [file for file in files if file.endswith('.png')]

            for file in png_files:
                img = Image.open(os.path.join(folder_path, file))
                #convert the PIL image to a tensor of dtype torch.int64
                img = fn_to_tensor(img)
                img = img.to(torch.int64)
                self.data.append(img)
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class MultiClassImagesV2(torch.utils.data.Dataset):
    def __init__(self, generate_data=False, N=100, folder_path='../images/multi_class_images_v2', transform=None):

        folder_path = os.path.join(HOME, folder_path)
        #retrieve the data from the folder
        self.data = []
        if not generate_data:
            # List all files in the folder
            files = os.listdir(folder_path)

            png_files = [file for file in files if file.endswith('.png')]

            for file in png_files:
                img = Image.open(os.path.join(folder_path, file))
                #convert the PIL image to a tensor of dtype torch.int64
                img = fn_to_tensor(img)
                img = img.to(torch.int64)
                self.data.append(img)
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

class ExpZeros(torch.utils.data.Dataset):
    def __init__(self, generate_data=False, h=4, w=4, N=100, file_path='exp_zeros/data.pkl', transform=None):

        #retrieve the data from the folder
        self.data = []
        if not generate_data:
            file_path = os.path.join(HOME, file_path)
            self.load_data(file_path)

        if generate_data:
            self.generate_data(h,w,N)
        
        self.transform = transform
    
    def generate_data(self, h, w, N):
        #draw the number of zeros
        seed = 0
        d = h*w
        zeros = draw_exp_zeros(N, d, seed)

        for i, zero in enumerate(zeros):
            #add (d - zero) randomly chosen white pixels to a black image
            sample = torch.zeros((h,w), dtype=torch.int64)
            wp = [torch.tensor(flat_to_square_index(i,w)) for i in np.random.choice(d, d-zero, replace=False)]
            rows = [p[0] for p in wp]
            cols = [p[1] for p in wp]
            sample[rows,cols] = 1

            #add it to the data
            self.data.append(sample)

    def save_data(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"Dataset saved to {save_path}")

    def load_data(self, file_path):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"Dataset loaded from {file_path}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

class Circles(torch.utils.data.Dataset):
    def __init__(self, generate_data=False, h=64, w=64, rmin=8,rmax=30,file_path='circles/data.pkl', transform=None):

        #retrieve the data from the folder
        self.data = []
        if not generate_data:
            file_path = os.path.join(HOME, file_path)
            self.load_data(file_path)

        if generate_data:
            self.generate_data(h,w,rmin,rmax)
        
        self.transform = transform
    
    def generate_data(self, h, w, rmin, rmax):
        self.data = [draw_circle(h,w,r) for r in range(rmin, rmax+1)]

    def save_data(self, save_path):
        save_path = os.path.join(HOME, save_path)
        with open(save_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"Dataset saved to {save_path}")

    def load_data(self, file_path):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"Dataset loaded from {file_path}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample