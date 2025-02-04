import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from dataset_manipulation.dataloaders import DataLoaderFactory
from models.vae_cityscapes import CS_unimodal
import sys
import os
import argparse

from multinomial_diffusion.segmentation_diffusion.multimodal.model import MultiModalSegmentationUnet
from multinomial_diffusion.segmentation_diffusion.model import get_model as get_model_seg_unet
from multinomial_diffusion.segmentation_diffusion.experiment import Experiment, CondExperiment, add_exp_args
from datasets.data import get_data, get_data_id, add_data_args
from model import get_model, get_model_id, add_model_args
from diffusion_utils.optim.multistep import get_optim, get_optim_id, add_optim_args


print("started traoning!")

### DEVICE ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### SEED ###
torch.manual_seed(0)

### DATA ###
data_path = 'data'
SUBSET_RATIO = 1.0
N_MODALITIES = 8
dlf = DataLoaderFactory(datadir=data_path, num_workers=1, pin_memory=True)

dataset_name = 'cityscapes_multimodal'

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
add_exp_args(parser)
add_data_args(parser)
add_model_args(parser)
add_optim_args(parser)
parser.data_root = data_path
args = parser.parse_args()
args.data_root = data_path

##################
## Specify data ##
##################

args.num_classes = 8

### UTILS ###
def plot_training_curves(mean_losses, mean_neg_loglikelihood,  mean_divs):
    epochs = len(mean_losses)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(epochs), mean_losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    plt.subplot(1, 3, 2)
    plt.plot(range(epochs), mean_neg_loglikelihood, label='Mean neg_loglikelihood')
    plt.xlabel('Epochs')
    plt.ylabel('Mean neg_loglikelihood')
    plt.title('Mean neg_loglikelihood')

    plt.subplot(1, 3, 3)
    plt.plot(range(epochs), mean_divs, label='KL Divergence')
    plt.xlabel('Epochs')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence')

    plt.tight_layout()
    #plt.show()
    plt.savefig('training_curves')

def plot_images(data, n_samples, modality = 1):
    #n_rows =  1
    #n_cols = n_samples
    n_rows, n_cols = 3, 3
    # Set up the subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(n_cols, n_rows))
    
    for i in range(n_cols):
        for j in range(n_rows):
            ax = axes[i, j] #if n_images > 1 else axes[j]
            img = data[i].detach().cpu().numpy()

            ax.imshow(img)
            ax.axis('off')  # Turn off axis
    
    plt.tight_layout()
    #plt.show()
    plt.savefig('training_samples')


def get_objective(obj):
    if obj == 'elbo':
        from objectives import elbo 
        return elbo
    elif obj == 'iwae':
        from objectives import iwae 
        return iwae
    elif obj ==  'dreg':
        from objectives import  dreg 
        return dreg
    else:
        raise ValueError(f"Dataset '{obj}' is not supported. Choose from 'elbo', 'cat_elbo', 'iwae', 'dreg'.")

def get_model(dataset_name, params_diff, params_vaes):

    params_diff["modalities_vaes"] = [CS_unimodal(params) for params in params_vaes]

    if dataset_name == 'cityscapes_multimodal':
        return MultiModalSegmentationUnet(**params_diff)
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported. Choose from 'cityscapes_multimodal'.")
    
def create_log_dir(epochs, beta, latent_dim, dist, obj='elbo', dataset_name=dataset_name, modality=None):
    if dataset_name in ['polymnist_mulimodal', 'cityscapes_multimodal']:
        dataset_name+='_'+str(modality)
    # Generate a unique directory name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_and_model_dir = f'runs/{dataset_name}/vae/ep_{epochs}_beta_{beta}_lt_{latent_dim}_dist_{dist}_{obj}_{timestamp}'        
    os.makedirs(log_and_model_dir, exist_ok=True)
    # Use this directory for both TensorBoard logs and saving models
    return log_and_model_dir

def train(model, train_loader, optimizer, epochs, beta, log_path, objective,  log_interval=1, K=1, plot_freq = 5):
    
    model.train()
    
    writer = SummaryWriter(log_path)
    # Lists to store the mean values of metrics per epoch
    mean_losses = []
    mean_neg_loglikelihood= []
    mean_divs = []
    

    for epoch in tqdm(range(epochs)):
        # Lists to store metrics per batch in the current epoch
        batch_losses = []
        batch_neg_loglikelihood= []
        batch_divs = []

        for i, (data,_) in enumerate(train_loader):
            #if dataset_name == 'polymnist_mulimodal':
            #    data = data[0]
            print(f"data shape: {data.shape}")
            data = data.to(device)
            optimizer.zero_grad()
            elbo, neg_logliklihood, div = objective(model, data, beta, K=K)
            loss = - elbo
            loss.backward()
            optimizer.step()
            # Store the metrics for the current batch
            batch_losses.append(loss.item())
            batch_neg_loglikelihood.append(neg_logliklihood.item())
            batch_divs.append(div.item())

        # Compute the mean values for the current epoch and store them
        mean_losses.append(sum(batch_losses) / len(batch_losses))
        mean_neg_loglikelihood.append(sum(batch_neg_loglikelihood) / len(batch_neg_loglikelihood))
        mean_divs.append(sum(batch_divs) / len(batch_divs))
        
        # Log metrics to TensorBoard
        if epoch % log_interval == 0:  # Log every 'log_interval' epochs
            writer.add_scalar('Total Loss/train', mean_losses[-1], epoch)
            writer.add_scalar('Negative LogLikelihood/train', mean_neg_loglikelihood[-1], epoch)
            writer.add_scalar('Div/train', mean_divs[-1], epoch)


        #TO DO
        if (epoch+1)%plot_freq ==0 or epoch==epochs-1:
                N = 9
                generations = model.generate(N)
                plot_images(generations, N)
                #plot_generations(generations, N)
        
        # Optional: print the mean metrics for the current epoch
        print(f"Epoch {epoch+1}/{epochs}, Loss: {mean_losses[-1]:.4f}, log p(x|z): {mean_neg_loglikelihood[-1]:.4f}, DIV: {mean_divs[-1]:.4f} ")
       # Save the model
        model_save_path = os.path.join(log_path, 'model_state_dict.pt')
        torch.save(model.state_dict(), model_save_path)
    writer.close()
    
    return mean_losses, mean_neg_loglikelihood,  mean_divs

"""
You can define the model hyper parameters here
"""

class Params:
    def __init__(self, dataset_name, device=device):
        self.priorposterior = 'Normal'
        if dataset_name == 'polymnist_mulimodal':
            self.latent_dim = 64
        elif dataset_name == 'cityscapes_multimodal':
            self.latent_dim = 128
        else:
            raise ValueError(f"dataset name {dataset_name} is not supported")
        self.device = device

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

"""
Training parameters
"""
batch_size = 16
epochs = 1000
obj = 'elbo'
objective = get_objective(obj)
beta = 0.1
K=1
plot_freq = 5
lr = 1e-3

params_vaes = [ParamsVAE(dataset_name=dataset_name, device=device, modality=mod) for mod in range(N_MODALITIES)]
params_diff = {"seg_unet": get_model_seg_unet(args, data_shape=(1, 32, 64))}
model = get_model(dataset_name, params_diff, params_vaes).to(device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, amsgrad=True)

### TRAIN ###

train_loader, test_loader = dlf.get_dataloader(dataset_name, batch_size, subset_percentage=SUBSET_RATIO)
log_path = create_log_dir(epochs, beta, params_vaes[0].latent_dim, params_vaes[0].priorposterior, obj=obj)
mean_losses, mean_neg_loglikelihood,  mean_divs = train(model, train_loader, optimizer, epochs, beta, log_path,
          log_interval=1, K=K, objective=objective)

plot_training_curves(mean_losses, mean_neg_loglikelihood,  mean_divs)