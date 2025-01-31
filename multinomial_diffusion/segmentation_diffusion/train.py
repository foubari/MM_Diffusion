import sys
sys.path.append("drive/MyDrive/MM_Diffusion/multinomial_diffusion")

import torch
import argparse
from diffusion_utils.utils import add_parent_path, set_seeds

# Exp
from experiment import Experiment, CondExperiment, add_exp_args

# Data
add_parent_path(level=1)
from datasets.data import get_data, get_data_id, add_data_args

# Model
from model import get_model, get_model_id, add_model_args

# Optim
from diffusion_utils.optim.multistep import get_optim, get_optim_id, add_optim_args

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
add_exp_args(parser)
add_data_args(parser)
add_model_args(parser)
add_optim_args(parser)
args = parser.parse_args()
set_seeds(args.seed)

##################
## Specify data ##
##################

train_loader, eval_loader, data_shape, num_classes = get_data(args)
args.num_classes = num_classes
data_id = get_data_id(args)

###################
## Specify model ##
###################

model = get_model(args, data_shape=data_shape)
#print the number of parameters in the model
print('Number of parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
model_id = get_model_id(args)

################################
## Print the available device ##
################################


if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print(f"Available : GPU: {torch.cuda.get_device_name(device)}")
else:
    print("Working on CPU")

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
