import os
import pickle
import torch
from prettytable import PrettyTable


def get_args_table(args_dict):
    table = PrettyTable(['Arg', 'Value'])
    for arg, val in args_dict.items():
        table.add_row([arg, val])
    return table


def get_metric_table(metric_dict, epochs):
    table = PrettyTable()
    table.add_column('Epoch', epochs)
    if len(metric_dict)>0:
        for metric_name, metric_values in metric_dict.items():
            table.add_column(metric_name, metric_values)
    return table

class BaseExperiment(object):

    def __init__(self, model, optimizer, scheduler_iter, scheduler_epoch,
                 log_path, eval_every, check_every, k_best):

        # Objects
        self.model = model
        self.optimizer = optimizer
        self.scheduler_iter = scheduler_iter
        self.scheduler_epoch = scheduler_epoch

        # Paths
        self.log_path = log_path
        self.check_path = os.path.join(log_path, 'check')

        # Intervals
        self.eval_every = eval_every
        self.check_every = check_every
        self.k_best = k_best

        # Initialize
        self.current_epoch = 0
        self.train_metrics = {}
        self.eval_metrics = {}
        self.eval_epochs = []

    def train_fn(self, epoch):
        raise NotImplementedError()

    def eval_fn(self, epoch):
        raise NotImplementedError()

    def log_fn(self, epoch, train_dict, eval_dict):
        raise NotImplementedError()

    def log_train_metrics(self, train_dict):
        if len(self.train_metrics)==0:
            for metric_name, metric_value in train_dict.items():
                self.train_metrics[metric_name] = [metric_value]
        else:
            for metric_name, metric_value in train_dict.items():
                self.train_metrics[metric_name].append(metric_value)

    def log_eval_metrics(self, eval_dict):
        if len(self.eval_metrics)==0:
            for metric_name, metric_value in eval_dict.items():
                self.eval_metrics[metric_name] = [metric_value]
        else:
            for metric_name, metric_value in eval_dict.items():
                self.eval_metrics[metric_name].append(metric_value)

    def create_folders(self):

        # Create log folder
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        print("Storing logs in:", self.log_path)

        # Create check folder
        if self.check_every is not None:
            if not os.path.exists(self.check_path):
                os.makedirs(self.check_path)
            print("Storing checkpoints in:", self.check_path)

    def save_args(self, args):

        # Save args
        with open(os.path.join(self.log_path, 'args.pickle'), "wb") as f:
            pickle.dump(args, f)

        # Save args table
        args_table = get_args_table(vars(args))
        with open(os.path.join(self.log_path,'args_table.txt'), "w") as f:
            f.write(str(args_table))

    def save_metrics(self):

        # Save metrics
        with open(os.path.join(self.log_path,'metrics_train.pickle'), 'wb') as f:
            pickle.dump(self.train_metrics, f)
        with open(os.path.join(self.log_path,'metrics_eval.pickle'), 'wb') as f:
            pickle.dump(self.eval_metrics, f)

        # Save metrics table
        metric_table = get_metric_table(self.train_metrics, epochs=list(range(1, self.current_epoch+2)))
        with open(os.path.join(self.log_path,'metrics_train.txt'), "w") as f:
            f.write(str(metric_table))
        metric_table = get_metric_table(self.eval_metrics, epochs=[e+1 for e in self.eval_epochs])
        with open(os.path.join(self.log_path,'metrics_eval.txt'), "w") as f:
            f.write(str(metric_table))

    def checkpoint_save(self, name='checkpoint.pt'):
        checkpoint = {'current_epoch': self.current_epoch,
                      'train_metrics': self.train_metrics,
                      'eval_metrics': self.eval_metrics,
                      'eval_epochs': self.eval_epochs,
                      'model': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'scheduler_iter': self.scheduler_iter.state_dict() if self.scheduler_iter else None,
                      'scheduler_epoch': self.scheduler_epoch.state_dict() if self.scheduler_epoch else None}
        torch.save(checkpoint, os.path.join(self.check_path, name))

    def checkpoint_load(self, check_path):
        checkpoint = torch.load(check_path)
        self.current_epoch = checkpoint['current_epoch']
        self.train_metrics = checkpoint['train_metrics']
        self.eval_metrics = checkpoint['eval_metrics']
        self.eval_epochs = checkpoint['eval_epochs']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler_iter: self.scheduler_iter.load_state_dict(checkpoint['scheduler_iter'])
        if self.scheduler_epoch: self.scheduler_epoch.load_state_dict(checkpoint['scheduler_epoch'])
    
    def checkpoint_remove(self, name):
        path_to_remove = os.path.join(self.check_path, name)
        if os.path.exists(path_to_remove):
            os.remove(path_to_remove)

    def run(self, epochs):

        for epoch in range(self.current_epoch, epochs):

            # Train
            train_dict = self.train_fn(epoch)
            self.log_train_metrics(train_dict)

            # Eval
            if (epoch+1) % self.eval_every == 0:
                eval_dict, samples = self.eval_fn(epoch)
                self.log_eval_metrics(eval_dict)
                self.eval_epochs.append(epoch)
            else:
                eval_dict = None
                samples = None

            # Log
            self.save_metrics()
            self.log_fn(epoch, train_dict, eval_dict, samples)

            # Checkpoint
            self.current_epoch += 1
            if (epoch+1) % self.check_every == 0:
                if self.k_best is None:
                    self.checkpoint_save()
                else:
                    #we save only if the bpd is in the k best bpds
                    name = 'checkpoint' + str(epoch+1) + '.pt'
                    if len(self.eval_epochs) <= self.k_best:
                        self.checkpoint_save(name)
                    else:
                        eval_epochs = self.eval_epochs
                        bpds = self.eval_metrics['bpd']

                        best_epochs_values = sorted(zip(eval_epochs, bpds), key=lambda x: x[1])[:self.k_best+1]
                        epoch_check_to_remove = best_epochs_values[-1][0]
                        name_to_remove = 'checkpoint' + str(epoch_check_to_remove+1) + '.pt'
                        self.checkpoint_remove(name_to_remove)
                        
                        if any(epoch == t[0] for t in best_epochs_values[:-1]):
                            self.checkpoint_save(name)
                


class DataParallelDistribution(torch.nn.DataParallel):
    """
    A DataParallel wrapper for Distribution.
    To be used instead of nn.DataParallel for Distribution objects.
    """

    def log_prob(self, *args, **kwargs):
        return self.forward(*args, mode='log_prob', **kwargs)

    def sample(self, *args, **kwargs):
        return self.module.sample(*args, **kwargs)

    def sample_with_log_prob(self, *args, **kwargs):
        return self.module.sample_with_log_prob(*args, **kwargs)
