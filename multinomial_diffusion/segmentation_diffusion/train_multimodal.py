import sys
sys.path.append("drive/MyDrive/MM_Diffusion/multinomial_diffusion")

import torch
import argparse
from diffusion_utils.utils import add_parent_path, set_seeds
from src.models.vae_cityscapes import CS_unimodal
from src.dataset_manipulation.dataloaders import DataLoaderFactory

# Exp
from experiment import Experiment, CondExperiment, add_exp_args

# Data
add_parent_path(level=1)
from datasets.data import get_data, get_data_id, add_data_args

# Model
from model import get_model, get_model_id, add_model_args

# Optim
from diffusion_utils.optim.multistep import get_optim, get_optim_id, add_optim_args

################################
## Print the available device ##
################################


if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print(f"Available : GPU: {torch.cuda.get_device_name(device)}")
else:
    print("Working on CPU")
    device = "cpu"

###########
## Setup ##
###########

NUM_CLASSES = 8
DATASET_NAME = 'cityscapes_multimodal'
SUBSET_RATIO = 1.0 # use a subset of the data for debuging
DATA_PATH = 'data'
BATCH_SIZE = 16

class ParamsVAE:
    def __init__(self, dataset_name, device=device, modality=None):
        self.priorposterior = 'Normal'
        if dataset_name == 'polymnist_mulimodal':
            self.latent_dim = 64
        elif dataset_name == 'cityscapes_multimodal':
            self.latent_dim = 128
        else:
            raise ValueError(f"dataset name {dataset_name} is not supported")
        self.modality = modality            
        self.device = device

parser = argparse.ArgumentParser()
add_exp_args(parser)
add_data_args(parser)
add_model_args(parser)
add_optim_args(parser)

# add addtional arguments for multimodality
args = parser.parse_args()
args.multimodal = True
params_vaes = [ParamsVAE(dataset_name=DATASET_NAME, device=device, modality=mod) for mod in range(NUM_CLASSES)]
args.modalities_vaes = [CS_unimodal(params) for params in params_vaes]
args.strategy = 'mixture_of_experts'

set_seeds(args.seed)

##################
## Specify data ##
##################
dlf = DataLoaderFactory(datadir=DATA_PATH, num_workers=1, pin_memory=True)
train_loader, eval_loader = dlf.get_dataloader(DATASET_NAME, BATCH_SIZE, subset_percentage=SUBSET_RATIO)
data_shape = (1, 32, 64)
args.num_classes = NUM_CLASSES
data_id = get_data_id(args)

###################
## Specify model ##
###################

model = get_model(args, data_shape=data_shape)
#print the number of parameters in the model
print('Number of parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
model_id = get_model_id(args)

#######################
## Specify optimizer ##
#######################

optimizer, scheduler_iter, scheduler_epoch = get_optim(args, model)
optim_id = get_optim_id(args)

##############
## Training ##
##############

exp_class = Experiment
if args.conditional:
    exp_class = CondExperiment
exp = exp_class(args=args,
                 data_id=data_id,
                 model_id=model_id,
                 optim_id=optim_id,
                 train_loader=train_loader,
                 eval_loader=eval_loader,
                 model=model,
                 optimizer=optimizer,
                 scheduler_iter=scheduler_iter,
                 scheduler_epoch=scheduler_epoch)

exp.run()
