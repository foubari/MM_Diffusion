import math
import torch
import torch.nn.functional as F
import torch.nn as nn

from multinomial_diffusion.diffusion_utils.diffusion_multinomial import MultinomialDiffusion, CondMultinomialDiffusion, PartialDiffusion
from multinomial_diffusion.segmentation_diffusion.layers.layers import SegmentationUnet, CondSegmentationUnet

def add_model_args(parser):
    # Model params
    parser.add_argument('--conditional', type=eval, default=False)
    parser.add_argument('--partial', type=eval, default=False)
    parser.add_argument('--loss_type', type=str, default='vb_stochastic')
    parser.add_argument('--loss_weighted', type=eval, default=False)
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--diffusion_dim', type=int, default=32)
    parser.add_argument('--dp_rate', type=float, default=0.)
    parser.add_argument('--param', type=str, default='x0')


def get_model_id(args):
    if args.conditional:
        return 'cond_multinomial_diffusion'
    if args.partial:
        return 'partial_diffusion'
    return 'multinomial_diffusion'


def get_model(args, data_shape):
    print('Data shape:', data_shape)
    data_shape = torch.Size(data_shape)

    current_shape = data_shape

    #binary12 case
    if data_shape[1] == 1 and data_shape[2] == 2:
        dim_mults = (1, 2)
    #4*4 images
    elif data_shape[1] == 4 and data_shape[2] == 4:
        dim_mults = (1, 2, 4)
    elif (data_shape[-1] // 4) % 2 == 0:
        dim_mults = (1, 2, 4, 8)
    else:
        dim_mults = (1, 4, 8)
    
    if args.conditional:
        dynamics = CondSegmentationUnet(
            num_classes=args.num_classes,
            dim=args.diffusion_dim,
            num_steps=args.diffusion_steps,
            dim_mults=dim_mults,
            dropout=args.dp_rate
        )

        base_dist = CondMultinomialDiffusion(
            args.num_classes, current_shape, dynamics, timesteps=args.diffusion_steps,
            loss_type=args.loss_type, loss_weighted=args.loss_weighted, parametrization=args.param)
        
        return base_dist

    dynamics = SegmentationUnet(
        num_classes=args.num_classes,
        dim=args.diffusion_dim,
        num_steps=args.diffusion_steps,
        dim_mults=dim_mults,
        dropout=args.dp_rate
    )
    
    if args.partial:
        base_dist = PartialDiffusion(
            args.num_classes, [0, 3], current_shape, dynamics, timesteps=args.diffusion_steps,
            loss_type=args.loss_type, loss_weighted=args.loss_weighted, parametrization=args.param)
        
        return base_dist

    base_dist = MultinomialDiffusion(
        args.num_classes, current_shape, dynamics, timesteps=args.diffusion_steps,
        loss_type=args.loss_type, loss_weighted=args.loss_weighted, parametrization=args.param)

    return base_dist
