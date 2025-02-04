import numpy as np
import torch
import random

from torch.utils.data import Dataset
from PIL import Image

import matplotlib.pyplot as plt

from collections import namedtuple

CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                 'has_instances', 'ignore_in_eval', 'color'])
classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

map_id_to_category_id = [x.category_id for x in classes]
map_id_to_category_id = torch.tensor(map_id_to_category_id)
category_ids = set([x.category_id for x in classes])


class CityscapesDataset(Dataset):
    """Cityscapes Dataset."""

    #NOTE: This class is basically the CityscapesFast class (in multinomial_diffusion/) adapted to the multimodal VAE training

    def __init__(self, npy_datapath, modality=0, transform=None,
                 target_transform=None):
        
        super().__init__()
        self.npy_datapath = npy_datapath
        self.modality = modality
        self.transform = transform
        self.target_transform = target_transform
        self.data = torch.from_numpy(np.load(npy_datapath))

    def __getitem__(self, index):
        
        img = self.data[index]
        img = img.long()
        img = map_id_to_category_id[img] #shape (1, H, W)
        img = (img == self.modality).long()

        if self.transform:
            assert img.size(0) == 1
            img = img[0]
            img = Image.fromarray(img.numpy().astype('uint8'))
            img = self.transform(img)
            img = np.array(img)
            img = torch.tensor(img).long()

        return img, 0

    def __len__(self):
        return len(self.data)


class CityscapesMultiModalDataset(Dataset):
    """Cityscapes Dataset."""

    #NOTE: This class is basically the CityscapesFast class (in multinomial_diffusion/) adapted to the multimodal VAE training

    def __init__(self, npy_datapath, pre_transform=None, post_transform=None, target_transform=None, subset_percentage=1.0):
        
        super().__init__()
        self.npy_datapath = npy_datapath
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.target_transform = target_transform
        self.subset_percentage = subset_percentage

        self.data = torch.from_numpy(np.load(npy_datapath))

        # Calculate how many samples to keep based on the percentage
        num_samples = len(self.data)
        num_samples_to_keep = int(num_samples * (self.subset_percentage / 100))
        
        # Randomly sample the data indices
        indices = random.sample(range(num_samples), num_samples_to_keep)
        
        # Create a subset of the data based on the selected indices
        self.data = self.data[indices]

    def __getitem__(self, index):

        img = self.data[index]
        print(img.shape)
        img = img.long()
        img = map_id_to_category_id[img] #shape (1, H, W)

        if self.pre_transform:
            assert img.size(0) == 1
            img = img[0]
            img = Image.fromarray(img.numpy().astype('uint8'))
            img = self.pre_transform(img)
            img = np.array(img)
            img = torch.tensor(img).long()
        
        #img = img.squeeze()  # Remove singleton dimensions
        C, H, W = img.shape  # C should be 1, so we preserve it
        meta_img = []
        print(img.shape)  # Should still be (1, H, W)

        for meta_idx in category_ids:
            _modality_img = torch.zeros((C, H, W), dtype=torch.uint8)  # Maintain (1, H, W)
            _modality_img[img == meta_idx] = 1  # Apply mask
            _modality_img = _modality_img.unsqueeze(1)
            meta_img.append(_modality_img)  # Already has (1, H, W)

        meta_img = torch.cat(meta_img, dim=0).long()  # Concatenate along the first dimension
        print(img.shape, meta_img.shape)

        if self.post_transform:
            raise NotImplementedError("Post tranforms are still not implemented in CityscapesMultiModalDataset")

        print(img.shape, meta_img.shape)

        return meta_img

    def __len__(self):
        return len(self.data)