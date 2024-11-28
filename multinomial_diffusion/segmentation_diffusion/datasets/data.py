import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import RandomHorizontalFlip, Pad, RandomAffine, \
    CenterCrop, RandomCrop, Compose, ToPILImage, ToTensor

from datasets.dataset import Binary12
from datasets.dataset import MultiClassImagesV1, MultiClassImagesV2
from datasets.dataset import ExpZeros
from datasets.dataset import Circles
from datasets.dataset import CTFirstCoronal, CTMiddleCoronal, CT25pctsCoronal
from datasets.dataset import Skin25pctsCoronal, Skin25pctsCoronalWI
from datasets.dataset import CondCT25pctsCoronal
from datasets.dataset import CTChosenCoronal

dataset_choices = {
    'binary12', 
    'multi_class_images_v1', 'multi_class_images_v2',
    'exp_zeros',
    'circles',
    'ct_first_coronal', 'ct_middle_coronal', 'ct_25pcts_coronal',
    'skin_25pcts_coronal', 'skin_25pcts_coronal_wi',
    'cond_ct_25pcts_coronal',
    'ct_chosen_coronal'
    }


p1, p2, p3, p4 = 0.1, 0.2, 0.3, 0.4
def add_data_args(parser):

    # Data params
    parser.add_argument('--dataset', type=str, default='binary12',
                        choices=dataset_choices)

    # Train params
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=eval, default=True)
    parser.add_argument('--augmentation', type=str, default=None)


def get_plot_transform(args):
    def identity(x):
        return x
    return identity


def get_data_id(args):
    return '{}'.format(args.dataset)


def get_data(args):
    assert args.dataset in dataset_choices

    # Dataset
    # data_shape = get_data_shape(args.dataset)
    
    #add the binary12 dataset
    if args.dataset == 'binary12':
        data_shape = (1, 1, 2)
        num_classes = 2
        train = Binary12(N=1000,p=[p1, p2, p3, p4])
        test = Binary12(N=100,p=[p1, p2, p3, p4])

    #add the MultiClassImages datasets
    elif args.dataset == 'multi_class_images_v1':
        data_shape = (1, 128, 128)
        num_classes = 3
        train = MultiClassImagesV1()
        test = MultiClassImagesV1()
    
    elif args.dataset == 'multi_class_images_v2':
        data_shape = (1, 128, 128)
        num_classes = 5
        train = MultiClassImagesV2()
        test = MultiClassImagesV2()
    
    elif args.dataset == 'exp_zeros':
        data_shape = (1, 4, 4)
        num_classes = 2
        train = ExpZeros()
        test = ExpZeros()
    
    elif args.dataset == 'circles':
        data_shape = (1, 64, 64)
        num_classes = 2
        train = Circles()
        test = Circles()
    
    elif args.dataset == 'ct_first_coronal':
        data_shape = (1, 512, 512)
        num_classes = 5
        train = CTFirstCoronal()
        test = CTFirstCoronal()
    
    elif args.dataset == 'ct_middle_coronal':
        data_shape = (1, 320, 320)
        num_classes = 4
        train = CTMiddleCoronal()
        test = CTMiddleCoronal()
        
    elif args.dataset == 'ct_25pcts_coronal':
        data_shape = (1, 512, 512)
        num_classes = 4
        train = CT25pctsCoronal(split='train')
        test = CT25pctsCoronal(split='test')
    elif args.dataset == 'skin_25pcts_coronal':
        data_shape = (1, 512, 512)
        num_classes = 2
        train = Skin25pctsCoronal(split='train')
        test = Skin25pctsCoronal(split='test')
    elif args.dataset == 'skin_25pcts_coronal_wi':
        data_shape = (1, 512, 512)
        num_classes = 3
        train = Skin25pctsCoronalWI(split='train')
        test = Skin25pctsCoronalWI(split='test')
    elif args.dataset == 'cond_ct_25pcts_coronal':
        data_shape = (1, 512, 512)
        num_classes = 4
        train = CondCT25pctsCoronal(split='train')
        test = CondCT25pctsCoronal(split='test')
    elif args.dataset == 'ct_chosen_coronal':
        data_shape = (1, 512, 512)
        num_classes = 4
        train = CTChosenCoronal(split='train')
        test = CTChosenCoronal(split='test')
    else:
        raise ValueError

    # Data Loader
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    eval_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    return train_loader, eval_loader, data_shape, num_classes


def get_augmentation(augmentation, dataset, data_shape):
    h, w = data_shape
    if augmentation is None:
        pil_transforms = []
    elif augmentation == 'horizontal_flip':
        pil_transforms = [RandomHorizontalFlip(p=0.5)]
    # torchvision.transforms.s
    return pil_transforms


def get_data_shape(dataset):
    if dataset == 'bmnist':
        return (28, 28)

    elif dataset == 'mnist_1bit':
        return (28, 28)

    elif dataset == 'binary12':
        return (1, 2)
    elif dataset == 'exp_zeros':
        return (4, 4)
    elif dataset == 'circles':
        return (64, 64)
    elif dataset == 'ct_first_coronal':
        return (512, 512)
    elif dataset == 'ct_middle_coronal':
        return (320, 320)
    elif dataset == 'ct_25pcts_coronal':
        return (512, 512)
    elif dataset == 'skin_25pcts_coronal':
        return (512, 512)