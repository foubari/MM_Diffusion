import torch
from diffusion_utils.utils import get_args_table, clean_dict

# Path
import os
import time
from shutil import copy2
import pathlib

import subprocess
hostname = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()

# Experiment
from diffusion_utils import BaseExperiment
from diffusion_utils.base import DataParallelDistribution

#  Logging framework
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

gray_shades = [(0.0, 0.0, 0.0),   # Black
            (0.33, 0.33, 0.33),  # Dark Gray
            (0.67, 0.67, 0.67),  # Light Gray
            (1.0, 1.0, 1.0)]     # White
gcmap = ListedColormap(gray_shades)

from lpips import LPIPS


def add_exp_args(parser):

    # Train params
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--parallel', type=str, default=None, choices={'dp'})
    parser.add_argument('--resume', type=str, default=None)

    # Logging params
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--eval_every', type=int, default=None)
    parser.add_argument('--check_every', type=int, default=None)
    parser.add_argument('--k_best', type=int, default=None)
    parser.add_argument('--eval_sample_every', type=int, default=None)
    parser.add_argument('--log_tb', type=eval, default=True)
    parser.add_argument('--log_home', type=str, default=None)


class DiffusionExperiment(BaseExperiment):
    no_log_keys = ['project', 'name',
                   'log_tb',
                   'check_every', 'kbest', 'eval_every', 'eval_sample_every',
                   'device', 'parallel'
                   'pin_memory', 'num_workers']

    def __init__(self, args,
                 data_id, model_id, optim_id,
                 train_loader, eval_loader,
                 model, optimizer, scheduler_iter, scheduler_epoch):
        if args.log_home is None:
            self.log_base = 'log/'
        else:
            self.log_base = args.log_home

        # Edit args
        if args.eval_every is None:
            args.eval_every = args.epochs
        if args.check_every is None:
            args.check_every = args.epochs
        if args.eval_sample_every is None:
            args.eval_sample_every = args.epochs + 1 #by default we never sample from the model
        if args.name is None:
            args.name = time.strftime("%Y-%m-%d_%H-%M-%S")
        if args.project is None:
            args.project = '_'.join([data_id, model_id])

        # Move model
        model = model.to(args.device)
        if args.parallel == 'dp':
            model = DataParallelDistribution(model)

        # Init parent
        super(DiffusionExperiment, self).__init__(model=model,
                                                  optimizer=optimizer,
                                                  scheduler_iter=scheduler_iter,
                                                  scheduler_epoch=scheduler_epoch,
                                                  log_path=os.path.join(self.log_base, data_id, model_id, optim_id, args.name),
                                                  eval_every=args.eval_every,
                                                  check_every=args.check_every,
                                                  k_best = args.k_best
                                                  )

        # Store args
        self.create_folders()
        self.save_args(args)
        self.args = args

        # Store interval (not necessarily an argument of a BaseExperiment)
        self.eval_sample_every = args.eval_sample_every

        # Store the experiment note
        if args.note_exp:
            #Copy the note_experiment.txt file in the log/ directory
            copy2('note_experiment.txt', self.log_path)

        # Store IDs
        self.data_id = data_id
        self.model_id = model_id
        self.optim_id = optim_id

        # Store data loaders
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # Init logging
        args_dict = clean_dict(vars(args), keys=self.no_log_keys)
        if args.log_tb:
            self.writer = SummaryWriter(os.path.join(self.log_path, 'tb'))
            self.writer.add_text("args", get_args_table(args_dict).get_html_string(), global_step=0)

        #Custom grey-level colormap for plotting
        K = self.model.num_classes
        gray_shades = [(1/(K-1) * k, 1/(K-1) * k, 1/(K-1) * k) for k in range(K)]
        self.gcmap = ListedColormap(gray_shades)

        #For partial diffusion, we can store 3 samples for the start of the sampling chain
        if self.model_id == 'partial_diffusion':
            for x in self.train_loader:
                a = x[0]
            for x in self.eval_loader:
                b, c = x[0], x[1]
            self.eval_starting_samples = torch.stack([a, b, c], dim=0).to(args.device)

    def log_fn(self, epoch, train_dict, eval_dict, samples):

        # Tensorboard
        if self.args.log_tb:
            for metric_name, metric_value in train_dict.items():
                self.writer.add_scalar('base/{}'.format(metric_name), metric_value, global_step=epoch+1)
            if eval_dict:
                for metric_name, metric_value in eval_dict.items():
                    self.writer.add_scalar('eval/{}'.format(metric_name), metric_value, global_step=epoch+1)
            if samples is not None:
                K = self.model.num_classes
                gcmap = self.gcmap
                B, C, H, W = samples.shape
                #self.writer.add_images('Eval Samples', samples, epoch, dataformats='NCHW')
                fig, axs = plt.subplots(3, B, figsize=(B*4, 3*4 + 2))
                for i, sample in enumerate(samples):
                    axs[0, i].imshow(sample[0], cmap = gcmap, vmin=0, vmax=K-1)
                    axs[0, i].axis('off')
                
                #moreover, we want to check how similar are the samples to the database
                closest_elements_list = self.find_closest_elements(samples, 1, dist='perceptual') #(B, list of len t=1)

                for i, closest_elements in enumerate(closest_elements_list):
                    distance, closest_element = closest_elements[0]
                    axs[1, i].imshow(closest_element, cmap = gcmap, vmin=0, vmax=K-1)
                    axs[1, i].axis('off')
                    axs[1, i].set_title('Lpips = {}'.format(distance))
                
                closest_elements_list = self.find_closest_elements(samples, 1, dist='l0') #(B, list of len t=1)

                for i, closest_elements in enumerate(closest_elements_list):
                    distance, closest_element = closest_elements[0]
                    axs[2, i].imshow(closest_element, cmap = gcmap, vmin=0, vmax=K-1)
                    axs[2, i].axis('off')
                    axs[2, i].set_title('l0 distance = {}'.format(distance/(H*W*C)))

                self.writer.add_figure('Eval samples', fig, epoch+1)
    
    def resume(self):
        resume_path = os.path.join(self.log_base, self.data_id, self.model_id, self.optim_id, self.args.name, 'check/checkpoint.pt')
        self.checkpoint_load(resume_path)
        for epoch in range(self.current_epoch):
            train_dict = {}
            for metric_name, metric_values in self.train_metrics.items():
                train_dict[metric_name] = metric_values[epoch]
            if epoch in self.eval_epochs:
                eval_dict = {}
                for metric_name, metric_values in self.eval_metrics.items():
                    eval_dict[metric_name] = metric_values[self.eval_epochs.index(epoch)]
            else: eval_dict = None
            self.log_fn(epoch, train_dict=train_dict, eval_dict=eval_dict, samples=None)

    def run(self):
        if self.args.resume: self.resume()
        super(DiffusionExperiment, self).run(epochs=self.args.epochs)
    
    def find_closest_elements(self, samples, t, dist='l0'):
        """
        Finds the closest `t` elements in self.train_loader to `samples` using the L0 norm.
        
        Parameters:
        samples: torch.Tensor of shape (B, C, H, W)
        t: int, the number of closest elements to find.
        
        Returns:
        A list of the closest `t` elements from the train_loader.
        """

        closest_elements = [[] for i in range(len(samples))]
        if dist == 'perceptual':
            loss_fn = LPIPS(net='squeeze', pnet_rand=True, model_path='../diffusion_utils/squeezenet.pth')
        
        for data in self.train_loader:
            if isinstance(data, list) and len(data) == 2: #to encompass the case of a conditional exp
                data = data[0]
            for data_sample in data:
                for i, sample in enumerate(samples):
                    if dist == 'l0':
                        distance = torch.sum((sample != data_sample).int()).item()
                    elif dist == 'perceptual':
                        distance = loss_fn(sample.float(), data_sample.float()).item()
                    closest_elements[i].append((distance, data_sample))

        for i in range(len(closest_elements)):
            closest_elements[i].sort(key=lambda x: x[0])

        return [element[:t] for element in closest_elements] #(B, t)
