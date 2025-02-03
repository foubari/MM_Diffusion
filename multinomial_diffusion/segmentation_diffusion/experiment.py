import torch
from torchvision.utils import make_grid
from diffusion_utils.loss import elbo_bpd, cond_elbo_bpd
from diffusion_utils.utils import add_parent_path

add_parent_path(level=2)
from diffusion_utils.experiment import DiffusionExperiment
from diffusion_utils.experiment import add_exp_args as add_exp_args_parent

# for saving the samples at training
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import h5py
#Todo: put this variable in the right class
max_epochs_over_eval_sample_every = 50

def add_exp_args(parser):
    add_exp_args_parent(parser)
    parser.add_argument('--clip_value', type=float, default=None)
    parser.add_argument('--clip_norm', type=float, default=None)
    parser.add_argument('--note_exp', type=bool, default=False)

gray_shades = [(0.0, 0.0, 0.0),   # Black
            (0.33, 0.33, 0.33),  # Dark Gray
            (0.67, 0.67, 0.67),  # Light Gray
            (1.0, 1.0, 1.0)]     # White
gcmap = ListedColormap(gray_shades)

class Experiment(DiffusionExperiment):

    def train_fn(self, epoch):
        self.model.train()
        loss_sum = 0.0
        loss_count = 0
        for x in self.train_loader:
            self.optimizer.zero_grad()
            loss = elbo_bpd(self.model, x.to(self.args.device))
            loss.backward()
            if self.args.clip_value: torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.clip_value)
            if self.args.clip_norm: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)
            self.optimizer.step()
            if self.scheduler_iter: self.scheduler_iter.step()
            loss_sum += loss.detach().cpu().item() * len(x)
            loss_count += len(x)
            print('Training. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, self.args.epochs, loss_count, len(self.train_loader.dataset), loss_sum/loss_count), end='\r')
        print('')
        if self.scheduler_epoch: self.scheduler_epoch.step()
        return {'bpd': loss_sum/loss_count}

    def eval_fn(self, epoch):
        self.model.eval()

        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            for x in self.train_loader:
                loss = elbo_bpd(self.model, x.to(self.args.device))
                loss_sum += loss.detach().cpu().item() * len(x)
                loss_count += len(x)
                print('Train evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, self.args.epochs, loss_count, len(self.train_loader.dataset), loss_sum/loss_count), end='\r')
            print('')

        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            for x in self.eval_loader:
                loss = elbo_bpd(self.model, x.to(self.args.device))
                loss_sum += loss.detach().cpu().item() * len(x)
                loss_count += len(x)
                print('     Evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, self.args.epochs, loss_count, len(self.eval_loader.dataset), loss_sum/loss_count), end='\r')
            print('')
        
        #draw some samples
        samples = None
        if (epoch+1) % self.eval_sample_every == 0:
            samples = self.eval_sample_fn(epoch)
        return {'bpd': loss_sum/loss_count}, samples


    def eval_sample_fn(self, epoch):
        num_samples = 3
        if self.model_id == 'partial_diffusion':
            x = self.eval_starting_samples
            return self.model.sample(num_samples, x).cpu()
        with torch.no_grad():
            return self.model.sample(num_samples).cpu()

    def eval_sample_fn_bis(self, epoch):

        #TO BE CAREFUL WITH THIS METHOD:
        #there is a maximum image size by the renderer: 2^16 ~ 65536 in each dimension.
        #For example I got 155059 pixels for 100 images.
        #A simple solution can be to give a maximum number of rows,
        #and then plot in another column for the next samples

        chain_samples = 5
        with torch.no_grad():
            samples_chain = self.model.sample_chain(chain_samples)
        
        batch = samples_chain[0] #take the last timestep of the reverse diffusion
        grid = make_grid(
            batch, nrow=5, padding=2, normalize=False, pad_value = 3)

        grid = grid.permute(1, 2, 0)
        grid = grid[:, :, 0] #only take one channel because make_grid make copies when dealing with single-channel images


        path = os.path.join(self.log_path, 'samples_at_training')
        os.makedirs(path, exist_ok=True)
        # append to the file and update the .png
        with h5py.File(os.path.join(path, 'samples.h5'), 'a') as f:
            # append
            if 'grids' not in f:
                grid_shape = grid.shape
                f.create_dataset('grids', shape=(0, *grid_shape), maxshape=(max_epochs_over_eval_sample_every, *grid_shape))

            i = (epoch+1)//self.eval_sample_every - 1
            if f['grids'].shape[0] <= i:
                f['grids'].resize(i+1,axis=0)
            f['grids'][i] = grid

            # update .png

            #plotting parameters
            nsamples = i+1
            h_on_w = grid.shape[0] / grid.shape[1]
            height_inches = 7
            width_inches = height_inches / h_on_w * 11/10 #add a margin to not overlap subplots

            #plot
            fig, axes = plt.subplots(nsamples, 1, figsize=(width_inches, height_inches* nsamples))
            titles = ['Epoch {}'.format(self.eval_sample_every * (i+1)) for i in range(nsamples)]
            for i, (ax, title) in enumerate(zip(axes, titles)):
                ax.imshow(f['grids'][i], cmap = gcmap, vmin=0, vmax=3)
                ax.set_title(title)

            plt.tight_layout()

            plt.savefig(os.path.join(path, 'grid_samples.png'))
            print('Grid of samples at epoch {} updated.'.format(epoch+1))
        
class CondExperiment(DiffusionExperiment):

    def train_fn(self, epoch):
        self.model.train()
        loss_sum = 0.0
        loss_count = 0
        for x, y in self.train_loader:  #difference of the conditioning
            self.optimizer.zero_grad()
            loss = cond_elbo_bpd(self.model, x.to(self.args.device), y.to(self.args.device))
            loss.backward()
            if self.args.clip_value: torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.clip_value)
            if self.args.clip_norm: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)
            self.optimizer.step()
            if self.scheduler_iter: self.scheduler_iter.step()
            loss_sum += loss.detach().cpu().item() * len(x)
            loss_count += len(x)
            print('Training. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, self.args.epochs, loss_count, len(self.train_loader.dataset), loss_sum/loss_count), end='\r')
        print('')
        if self.scheduler_epoch: self.scheduler_epoch.step()
        return {'bpd': loss_sum/loss_count}

    def eval_fn(self, epoch):
        self.model.eval()

        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            for x, y in self.train_loader:
                loss = cond_elbo_bpd(self.model, x.to(self.args.device), y.to(self.args.device))
                loss_sum += loss.detach().cpu().item() * len(x)
                loss_count += len(x)
                print('Train evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, self.args.epochs, loss_count, len(self.train_loader.dataset), loss_sum/loss_count), end='\r')
            print('')

        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            for x, y in self.eval_loader:
                loss = cond_elbo_bpd(self.model, x.to(self.args.device), y.to(self.args.device))
                loss_sum += loss.detach().cpu().item() * len(x)
                loss_count += len(x)
                print('     Evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, self.args.epochs, loss_count, len(self.eval_loader.dataset), loss_sum/loss_count), end='\r')
            print('')
        
        #draw some samples
        samples = None
        if (epoch+1) % self.eval_sample_every == 0:
            samples = self.eval_sample_fn(epoch)
        return {'bpd': loss_sum/loss_count}, samples


    def eval_sample_fn(self, epoch):
        num_samples = 3
        _, y = next(iter(self.train_loader)) #to be fixed in case batch_size < num_samples
        y = y[:num_samples]
        with torch.no_grad():
            return self.model.sample(num_samples, y.to(self.args.device)).cpu()

class MultiModalExperiment(Experiment):

    def train_fn(self, epoch):
        self.model.train()
        loss_sum = 0.0
        loss_count = 0
        for x in self.train_loader:
            self.optimizer.zero_grad()
            loss = elbo_bpd(self.model, x.to(self.args.device))
            loss.backward()
            if self.args.clip_value: torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.clip_value)
            if self.args.clip_norm: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)
            self.optimizer.step()
            if self.scheduler_iter: self.scheduler_iter.step()
            loss_sum += loss.detach().cpu().item() * len(x)
            loss_count += len(x)
            print('Training. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, self.args.epochs, loss_count, len(self.train_loader.dataset), loss_sum/loss_count), end='\r')
        print('')
        if self.scheduler_epoch: self.scheduler_epoch.step()
        return {'bpd': loss_sum/loss_count}