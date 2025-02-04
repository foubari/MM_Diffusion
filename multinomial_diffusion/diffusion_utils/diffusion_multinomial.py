import torch
import torch.nn.functional as F
import numpy as np
from inspect import isfunction
from typing import List
from src.models.vae_base import VAE
import random

"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8


def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)

    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def exists(x):
    return x is not None


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    #print(x.shape, x)

    #this function assumes a batch dimension
    x_onehot = F.one_hot(x, num_classes)

    permute_order = (0, -1) + tuple(range(1, len(x.size())))

    x_onehot = x_onehot.permute(permute_order)

    log_x = torch.log(x_onehot.float().clamp(min=1e-30))

    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)

def log_sum_exp(log_x,dims=(1,)):
    return torch.log(log_x.exp().sum(dims, keepdim=True).clamp(min=1e-30))

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas


class MultinomialDiffusion(torch.nn.Module):
    def __init__(self, num_classes, shape, denoise_fn, timesteps=1000,
                 loss_type='vb_stochastic', loss_weighted=False, parametrization='x0'):
        super(MultinomialDiffusion, self).__init__()
        assert loss_type in ('vb_stochastic', 'vb_all')
        assert parametrization in ('x0', 'direct')

        if loss_type == 'vb_all':
            print('Computing the loss using the bound on _all_ timesteps.'
                  ' This is expensive both in terms of memory and computation.')

        self.num_classes = num_classes
        self._denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.loss_weighted = loss_weighted
        self.shape = shape
        self.num_timesteps = timesteps
        self.parametrization = parametrization

        alphas = cosine_beta_schedule(timesteps)

        alphas = torch.tensor(alphas.astype('float64'))
        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        # Convert to float32 and register buffers.
        self.register_buffer('log_alpha', log_alpha.float())
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())

        self.register_buffer('Lt_history', torch.zeros(timesteps))
        self.register_buffer('Lt_count', torch.zeros(timesteps))

    def multinomial_kl(self, log_prob1, log_prob2, weights=None):
        if weights is None:
            weights = torch.ones_like(log_prob1)
        kl = (weights * log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )

        return log_probs

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)

        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )

        return log_probs

    def predict_start(self, log_x_t, t):
        x_t = log_onehot_to_index(log_x_t)

        out = self._denoise_fn(t, x_t)

        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes
        assert out.size()[2:] == x_t.size()[1:]
        log_pred = F.log_softmax(out, dim=1)
        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t):
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0)


        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = \
            unnormed_logprobs \
            - torch.logsumexp(unnormed_logprobs, dim=1, keepdim=True)

        return log_EV_xtmin_given_xt_given_xstart

    def p_pred(self, log_x, t):
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(log_x, t=t)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, t=t)
        else:
            raise ValueError
        return log_model_pred
    
    
    def log_prob_to_log_score(self,log_prob,log_x,t):
        log_a = log_prob - self.q_pred_one_timestep(log_x,t)
        log_a_T_1 = log_sum_exp(log_a)
        log_score = log_a_T_1 + self.q_pred_one_timestep(log_a - log_a_T_1, t)
        return log_score
    
    def log_score_to_log_prob(self,log_score,log_x,t):
        #inverse the score
        log_alpha_t = extract(self.log_alpha, t, log_x.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x.shape)

        #we cannot use the log representation in case there are negative components
        log_score_T_1 = log_sum_exp(log_score)
        u = log_score.exp() - (log_1_min_alpha_t - np.log(self.num_classes) + log_score_T_1).exp()
        
        #threshold the negative values to zero
        u = u.clamp(min=1e-30)
        log_u = torch.log(u)
        #renormalize
        log_u = log_u - log_sum_exp(log_u)        

        #compute log_score from the formula
        log_v = self.q_pred_one_timestep(log_x,t)
        log_prob = log_v + log_u - log_alpha_t

        return log_prob

    @torch.no_grad()
    def p_sample(self, log_x, t, constraint=None, categories=None):
        model_log_prob = self.p_pred(log_x=log_x, t=t)
        if constraint is not None:
            log_score = self.log_prob_to_log_score(model_log_prob,log_x,t)
            new_log_score = log_score + constraint(log_x,t)
            new_model_log_prob = self.log_score_to_log_prob(new_log_score,log_x,t)
            model_log_prob = new_model_log_prob
        out = self.log_sample_categorical(model_log_prob, categories)
        return out

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.log_alpha.device

        b = shape[0]
        # start with random normal image.
        img = torch.randn(shape, device=device)

        for i in reversed(range(1, self.num_timesteps)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img
    
    @torch.no_grad()
    def _sample(self, image_size, batch_size = 16):
        return self.p_sample_loop((batch_size, 3, image_size, image_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in reversed(range(0, t)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def log_sample_categorical(self, logits, categories=None):
        if categories is not None:
            exclude_categories = [k for k in range(self.num_classes) if not k in categories]
            logits[:, exclude_categories,...] = np.log(1e-30) #set the logit to minus infinity
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)

        #we should have that we have sampled in the 'categories'
        if categories is not None:
            device = self.log_alpha.device
            values = torch.tensor(categories).to(device)
            assert torch.isin(sample, values).all()
        
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def nll(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        loss = 0
        for t in range(0, self.num_timesteps):
            t_array = (torch.ones(b, device=device) * t).long()

            kl = self.compute_Lt(
                log_x_start=log_x_start,
                log_x_t=self.q_sample(log_x_start=log_x_start, t=t_array),
                t=t_array)

            loss += kl

        loss += self.kl_prior(log_x_start)

        return loss

    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)

    def compute_Lt(self, log_x_start, log_x_t, t, detach_mean=False, weighted=False):
        log_true_prob = self.q_posterior(
            log_x_start=log_x_start, log_x_t=log_x_t, t=t)

        log_model_prob = self.p_pred(log_x=log_x_t, t=t)

        if detach_mean:
            log_model_prob = log_model_prob.detach()
        
        weights = torch.ones_like(log_model_prob)
        #reweighting the KL
        if weighted:
            weights = torch.where(log_x_start == 0., 10.0, 1.0).float()

        kl = self.multinomial_kl(log_true_prob, log_model_prob, weights)
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl

        return loss

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def _train_loss(self, x):
        b, device = x.size(0), x.device

        if self.loss_type == 'vb_stochastic':
            x_start = x

            t, pt = self.sample_time(b, device, 'importance')

            log_x_start = index_to_log_onehot(x_start, self.num_classes)

            kl = self.compute_Lt(
                log_x_start, self.q_sample(log_x_start=log_x_start, t=t), t, weighted=self.loss_weighted)

            Lt2 = kl.pow(2)
            Lt2_prev = self.Lt_history.gather(dim=0, index=t)
            new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
            self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
            self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

            kl_prior = self.kl_prior(log_x_start)

            # Upweigh loss term of the kl
            vb_loss = kl / pt + kl_prior

            return -vb_loss

        elif self.loss_type == 'vb_all':
            # Expensive, dont do it ;).
            return -self.nll(x)
        else:
            raise ValueError()

    def log_prob(self, x):
        b, device = x.size(0), x.device
        if self.training:
            return self._train_loss(x)

        else:
            log_x_start = index_to_log_onehot(x, self.num_classes)

            t, pt = self.sample_time(b, device, 'importance')

            kl = self.compute_Lt(
                log_x_start, self.q_sample(log_x_start=log_x_start, t=t), t)

            kl_prior = self.kl_prior(log_x_start)

            # Upweigh loss term of the kl
            loss = kl / pt + kl_prior

            return -loss

    def sample(self, num_samples):
        #set always the same seed for the samples, so as to compare 
        #the samples between epochs
        
        rng_state = torch.get_rng_state()
        torch.manual_seed(8)
        
        b = num_samples
        device = self.log_alpha.device
        uniform_logits = torch.zeros((b, self.num_classes) + self.shape, device=device)
        log_z = self.log_sample_categorical(uniform_logits)
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Sample timestep {i:4d}', end='\r')

            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t)
        
        #set back the random state before sampling
        torch.set_rng_state(rng_state)
        return log_onehot_to_index(log_z)

    def sample_masked(self, num_samples, x0, mask, categories):
        #set always the same seed for the samples, so as to compare 
        #the samples between epochs
        
        rng_state = torch.get_rng_state()
        torch.manual_seed(8)

        #x0 should be of shape self.shape
        assert x0.shape == self.shape

        #mask should be of shape self.shape (and only contains 0 and 1)
        assert mask.shape == self.shape

        b = num_samples
        device = self.log_alpha.device

        #transform x0 into tensor, batched then log_onehot
        x0 = torch.tensor(x0, dtype=torch.int64)
        x0 = x0.unsqueeze(0).repeat(b, *[1]*len(x0.shape)).to(device)
        log_x0 = index_to_log_onehot(x0, self.num_classes)

        #transform the mask into a tensor and duplicate for all classes, then batch
        mask = torch.tensor(mask, dtype=torch.float32)
        mask = mask.unsqueeze(0).repeat(self.num_classes, *[1]*len(mask.shape))
        mask = mask.unsqueeze(0).repeat(b, *[1]*len(mask.shape)).to(device)

        uniform_logits = torch.zeros((b, self.num_classes) + self.shape, device=device)

        #sample and apply the mask
        log_z = self.log_sample_categorical(uniform_logits, categories)
        log_z = log_z * (1. - mask) + log_x0 * mask
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Sample timestep {i:4d}', end='\r')

            t = torch.full((b,), i, device=device, dtype=torch.long)

            #sample and apply the mask
            log_z = self.p_sample(log_z, t, categories=categories)
            log_z = log_z * (1. - mask) + log_x0 * mask
        
        #set back the random state before sampling
        torch.set_rng_state(rng_state)
        return log_onehot_to_index(log_z)

    def sample_chain(self, num_samples,constraint=None):
        b = num_samples
        device = self.log_alpha.device
        uniform_logits = torch.zeros(
            (b, self.num_classes) + self.shape, device=device)

        zs = torch.zeros((self.num_timesteps, b) + self.shape).long()

        log_z = self.log_sample_categorical(uniform_logits)
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Chain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t,constraint=constraint)

            zs[i] = log_onehot_to_index(log_z)
        print()
        return zs

    def p_pred_start(self, log_x, t):
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(log_x, t=t)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, t=t)
        else:
            raise ValueError
        return log_model_pred, log_x_recon

    def sample_masked_chain(self, num_samples, x0, mask, categories):
        #set always the same seed for the samples, so as to compare 
        #the samples between epochs
        
        rng_state = torch.get_rng_state()
        torch.manual_seed(8)

        #x0 should be of shape self.shape
        assert x0.shape == self.shape

        #mask should be of shape self.shape (and only contains 0 and 1)
        assert mask.shape == self.shape

        b = num_samples
        device = self.log_alpha.device

        #transform x0 into tensor, batched then log_onehot
        x0 = torch.tensor(x0, dtype=torch.int64)
        x0 = x0.unsqueeze(0).repeat(b, *[1]*len(x0.shape)).to(device)
        log_x0 = index_to_log_onehot(x0, self.num_classes)

        #transform the mask into a tensor and duplicate for all classes, then batch
        mask = torch.tensor(mask, dtype=torch.float32)
        mask = mask.unsqueeze(0).repeat(self.num_classes, *[1]*len(mask.shape))
        mask = mask.unsqueeze(0).repeat(b, *[1]*len(mask.shape)).to(device)

        uniform_logits = torch.zeros((b, self.num_classes) + self.shape, device=device)

        #sample and apply the mask
        log_z = self.log_sample_categorical(uniform_logits, categories)
        log_z = log_z * (1. - mask) + log_x0 * mask

        zs = torch.zeros((self.num_timesteps, b) + self.shape).long()
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Sample timestep {i:4d}', end='\r')

            t = torch.full((b,), i, device=device, dtype=torch.long)

            #sample and apply the mask
            log_z = self.p_sample(log_z, t, categories=categories)
            log_z = log_z * (1. - mask) + log_x0 * mask

            zs[i] = log_onehot_to_index(log_z)
        
        #set back the random state before sampling
        torch.set_rng_state(rng_state)
        return zs

    @torch.no_grad()
    def p_sample_start(self, log_x, t, constraint=None, categories=None):
        model_log_prob, pred_start = self.p_pred_start(log_x=log_x, t=t)
        if constraint is not None:
            log_score = self.log_prob_to_log_score(model_log_prob,log_x,t)
            new_log_score = log_score + constraint(log_x,t)
            new_model_log_prob = self.log_score_to_log_prob(new_log_score,log_x,t)
            model_log_prob = new_model_log_prob
        out = self.log_sample_categorical(model_log_prob, categories)
        return out, pred_start
    
    def sample_chain_start(self, num_samples,constraint=None):
        b = num_samples
        device = self.log_alpha.device
        uniform_logits = torch.zeros(
            (b, self.num_classes) + self.shape, device=device)

        zs = torch.zeros((self.num_timesteps, b) + self.shape).long()
        starts = torch.zeros((self.num_timesteps, b) + self.shape).long()

        log_z = self.log_sample_categorical(uniform_logits)
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Chain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z, pred_start = self.p_sample_start(log_z, t,constraint=constraint)

            zs[i] = log_onehot_to_index(log_z)
            starts[i] = log_onehot_to_index(pred_start)
        print()
        return zs, starts

class CondMultinomialDiffusion(torch.nn.Module):
    def __init__(self, num_classes, shape, denoise_fn, timesteps=1000,
                 loss_type='vb_stochastic', loss_weighted=False, parametrization='x0'):
        super(CondMultinomialDiffusion, self).__init__()
        assert loss_type in ('vb_stochastic', 'vb_all')
        assert parametrization in ('x0', 'direct')

        if loss_type == 'vb_all':
            print('Computing the loss using the bound on _all_ timesteps.'
                  ' This is expensive both in terms of memory and computation.')

        self.num_classes = num_classes
        self._denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.loss_weighted = loss_weighted
        self.shape = shape
        self.num_timesteps = timesteps
        self.parametrization = parametrization

        alphas = cosine_beta_schedule(timesteps)

        alphas = torch.tensor(alphas.astype('float64'))
        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        # Convert to float32 and register buffers.
        self.register_buffer('log_alpha', log_alpha.float())
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())

        self.register_buffer('Lt_history', torch.zeros(timesteps))
        self.register_buffer('Lt_count', torch.zeros(timesteps))

    def multinomial_kl(self, log_prob1, log_prob2, weights=None):
        if weights is None:
            weights = torch.ones_like(log_prob1)
        kl = (weights * log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )

        return log_probs

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)

        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )

        return log_probs

    def predict_start(self, log_x_t, y, t):
        x_t = log_onehot_to_index(log_x_t)

        out = self._denoise_fn(t, x_t, y)

        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes
        assert out.size()[2:] == x_t.size()[1:]
        log_pred = F.log_softmax(out, dim=1)
        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t):
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0)


        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = \
            unnormed_logprobs \
            - torch.logsumexp(unnormed_logprobs, dim=1, keepdim=True)

        return log_EV_xtmin_given_xt_given_xstart

    def p_pred(self, log_x, y, t):
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(log_x, y, t=t)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, t=t)
        else:
            raise ValueError
        return log_model_pred
    
    
    def log_prob_to_log_score(self,log_prob,log_x,t):
        log_a = log_prob - self.q_pred_one_timestep(log_x,t)
        log_a_T_1 = log_sum_exp(log_a)
        log_score = log_a_T_1 + self.q_pred_one_timestep(log_a - log_a_T_1, t)
        return log_score
    
    def log_score_to_log_prob(self,log_score,log_x,t):
        #inverse the score
        log_alpha_t = extract(self.log_alpha, t, log_x.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x.shape)

        #we cannot use the log representation in case there are negative components
        log_score_T_1 = log_sum_exp(log_score)
        u = log_score.exp() - (log_1_min_alpha_t - np.log(self.num_classes) + log_score_T_1).exp()
        
        #threshold the negative values to zero
        u = u.clamp(min=1e-30)
        log_u = torch.log(u)
        #renormalize
        log_u = log_u - log_sum_exp(log_u)        

        #compute log_score from the formula
        log_v = self.q_pred_one_timestep(log_x,t)
        log_prob = log_v + log_u - log_alpha_t

        return log_prob

    @torch.no_grad()
    def p_sample(self, log_x, y, t, constraint=None, categories=None):
        model_log_prob = self.p_pred(log_x=log_x, y=y, t=t)
        if constraint is not None:
            log_score = self.log_prob_to_log_score(model_log_prob,log_x,t)
            new_log_score = log_score + constraint(log_x,t)
            new_model_log_prob = self.log_score_to_log_prob(new_log_score,log_x,t)
            model_log_prob = new_model_log_prob
        out = self.log_sample_categorical(model_log_prob, categories)
        return out

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.log_alpha.device

        b = shape[0]
        # start with random normal image.
        img = torch.randn(shape, device=device)

        for i in reversed(range(1, self.num_timesteps)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img
    
    @torch.no_grad()
    def _sample(self, image_size, batch_size = 16):
        return self.p_sample_loop((batch_size, 3, image_size, image_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in reversed(range(0, t)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def log_sample_categorical(self, logits, categories=None):
        if categories is not None:
            exclude_categories = [k for k in range(self.num_classes) if not k in categories]
            logits[:, exclude_categories,...] = np.log(1e-30) #set the logit to minus infinity
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)

        #we should have that we have sampled in the 'categories'
        if categories is not None:
            device = self.log_alpha.device
            values = torch.tensor(categories).to(device)
            assert torch.isin(sample, values).all()
        
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def nll(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        loss = 0
        for t in range(0, self.num_timesteps):
            t_array = (torch.ones(b, device=device) * t).long()

            kl = self.compute_Lt(
                log_x_start=log_x_start,
                log_x_t=self.q_sample(log_x_start=log_x_start, t=t_array),
                t=t_array)

            loss += kl

        loss += self.kl_prior(log_x_start)

        return loss

    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)

    def compute_Lt(self, log_x_start, y, log_x_t, t, detach_mean=False, weighted=False):
        log_true_prob = self.q_posterior(
            log_x_start=log_x_start, log_x_t=log_x_t, t=t)

        log_model_prob = self.p_pred(log_x=log_x_t, y=y, t=t)

        if detach_mean:
            log_model_prob = log_model_prob.detach()
        
        weights = torch.ones_like(log_model_prob)
        #reweighting the KL
        if weighted:
            weights = torch.where(log_x_start == 0., 10.0, 1.0).float()

        kl = self.multinomial_kl(log_true_prob, log_model_prob, weights)
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl

        return loss

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def _train_loss(self, x, y):
        b, device = x.size(0), x.device

        if self.loss_type == 'vb_stochastic':
            x_start = x

            t, pt = self.sample_time(b, device, 'importance')

            log_x_start = index_to_log_onehot(x_start, self.num_classes)

            kl = self.compute_Lt(
                log_x_start, y, self.q_sample(log_x_start=log_x_start, t=t), t, weighted=self.loss_weighted)

            Lt2 = kl.pow(2)
            Lt2_prev = self.Lt_history.gather(dim=0, index=t)
            new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
            self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
            self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

            kl_prior = self.kl_prior(log_x_start)

            # Upweigh loss term of the kl
            vb_loss = kl / pt + kl_prior

            return -vb_loss

        elif self.loss_type == 'vb_all':
            # Expensive, dont do it ;).
            return -self.nll(x)
        else:
            raise ValueError()

    def log_prob(self, x, y):
        b, device = x.size(0), x.device
        if self.training:
            return self._train_loss(x, y)

        else:
            log_x_start = index_to_log_onehot(x, self.num_classes)

            t, pt = self.sample_time(b, device, 'importance')

            kl = self.compute_Lt(
                log_x_start, y, self.q_sample(log_x_start=log_x_start, t=t), t)

            kl_prior = self.kl_prior(log_x_start)

            # Upweigh loss term of the kl
            loss = kl / pt + kl_prior

            return -loss

    def sample(self, num_samples, y):

        # the number of samples should be the same as the batch_size of y
        assert num_samples == y.size(0)

        #set always the same seed for the samples, so as to compare 
        #the samples between epochs
        
        rng_state = torch.get_rng_state()
        torch.manual_seed(8)
        
        b = num_samples
        device = self.log_alpha.device
        uniform_logits = torch.zeros((b, self.num_classes) + self.shape, device=device)
        log_z = self.log_sample_categorical(uniform_logits)
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Sample timestep {i:4d}', end='\r')

            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, y, t)
        
        #set back the random state before sampling
        torch.set_rng_state(rng_state)
        return log_onehot_to_index(log_z)

    def sample_masked(self, num_samples, x0, mask, categories):
        #set always the same seed for the samples, so as to compare 
        #the samples between epochs
        
        rng_state = torch.get_rng_state()
        torch.manual_seed(8)

        #x0 should be of shape self.shape
        assert x0.shape == self.shape

        #mask should be of shape self.shape (and only contains 0 and 1)
        assert mask.shape == self.shape

        b = num_samples
        device = self.log_alpha.device

        #transform x0 into tensor, batched then log_onehot
        x0 = torch.tensor(x0, dtype=torch.int64)
        x0 = x0.unsqueeze(0).repeat(b, *[1]*len(x0.shape)).to(device)
        log_x0 = index_to_log_onehot(x0, self.num_classes)

        #transform the mask into a tensor and duplicate for all classes, then batch
        mask = torch.tensor(mask, dtype=torch.float32)
        mask = mask.unsqueeze(0).repeat(self.num_classes, *[1]*len(mask.shape))
        mask = mask.unsqueeze(0).repeat(b, *[1]*len(mask.shape)).to(device)

        uniform_logits = torch.zeros((b, self.num_classes) + self.shape, device=device)

        #sample and apply the mask
        log_z = self.log_sample_categorical(uniform_logits, categories)
        log_z = log_z * (1. - mask) + log_x0 * mask
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Sample timestep {i:4d}', end='\r')

            t = torch.full((b,), i, device=device, dtype=torch.long)

            #sample and apply the mask
            log_z = self.p_sample(log_z, t, categories=categories)
            log_z = log_z * (1. - mask) + log_x0 * mask
        
        #set back the random state before sampling
        torch.set_rng_state(rng_state)
        return log_onehot_to_index(log_z)

    def sample_chain(self, num_samples,constraint=None):
        b = num_samples
        device = self.log_alpha.device
        uniform_logits = torch.zeros(
            (b, self.num_classes) + self.shape, device=device)

        zs = torch.zeros((self.num_timesteps, b) + self.shape).long()

        log_z = self.log_sample_categorical(uniform_logits)
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Chain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t,constraint=constraint)

            zs[i] = log_onehot_to_index(log_z)
        print()
        return zs

    def p_pred_start(self, log_x, t):
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(log_x, t=t)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, t=t)
        else:
            raise ValueError
        return log_model_pred, log_x_recon

    @torch.no_grad()
    def p_sample_start(self, log_x, t, constraint=None, categories=None):
        model_log_prob, pred_start = self.p_pred_start(log_x=log_x, t=t)
        if constraint is not None:
            log_score = self.log_prob_to_log_score(model_log_prob,log_x,t)
            new_log_score = log_score + constraint(log_x,t)
            new_model_log_prob = self.log_score_to_log_prob(new_log_score,log_x,t)
            model_log_prob = new_model_log_prob
        out = self.log_sample_categorical(model_log_prob, categories)
        return out, pred_start
    
    def sample_chain_start(self, num_samples,constraint=None):
        b = num_samples
        device = self.log_alpha.device
        uniform_logits = torch.zeros(
            (b, self.num_classes) + self.shape, device=device)

        zs = torch.zeros((self.num_timesteps, b) + self.shape).long()
        starts = torch.zeros((self.num_timesteps, b) + self.shape).long()

        log_z = self.log_sample_categorical(uniform_logits)
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Chain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z, pred_start = self.p_sample_start(log_z, t,constraint=constraint)

            zs[i] = log_onehot_to_index(log_z)
            starts[i] = log_onehot_to_index(pred_start)
        print()
        return zs, starts

class PartialDiffusion(torch.nn.Module):
    def __init__(self, num_classes, inactive_classes, shape, denoise_fn, timesteps=1000,
                 loss_type='vb_stochastic', loss_weighted=False, parametrization='x0'):
        super(PartialDiffusion, self).__init__()
        assert loss_type in ('vb_stochastic', 'vb_all')
        assert parametrization in ('x0', 'direct')

        assert all(0 <= item < num_classes for item in inactive_classes)

        if loss_type == 'vb_all':
            print('Computing the loss using the bound on _all_ timesteps.'
                  ' This is expensive both in terms of memory and computation.')

        self.num_classes = num_classes
        self.num_inactive_classes = len(inactive_classes)
        self.num_active_classes = num_classes - len(inactive_classes)
        self.inactive_classes = inactive_classes
        self.active_classes = [cl for cl in range(num_classes) if not cl in inactive_classes]
        self._denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.loss_weighted = loss_weighted
        self.shape = shape
        self.num_timesteps = timesteps
        self.parametrization = parametrization

        alphas = cosine_beta_schedule(timesteps)

        alphas = torch.tensor(alphas.astype('float64'))
        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        # Convert to float32 and register buffers.
        self.register_buffer('log_alpha', log_alpha.float())
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())
        
        self.register_buffer('Lt_history', torch.zeros(timesteps))
        self.register_buffer('Lt_count', torch.zeros(timesteps))

        #for practicity
        active_classes_tensor = torch.tensor([int(cl in self.active_classes) for cl in range(self.num_classes)], dtype=torch.float32)
        self.register_buffer('active_classes_tensor', active_classes_tensor)

    def log_onehot_to_mask_active(self, log_x):
        #the shape of x is (B, ...)
        #we want it to be (B, self.num_classes, ...)
        x = log_onehot_to_index(log_x)
        m = torch.isin(x, torch.tensor(self.active_classes, dtype=torch.int64, device=self.log_alpha.device)).to(torch.float32)
        m = m.unsqueeze(1)
        m = m.repeat(1, self.num_classes, *([1] * (m.ndim - 2)))

        return m
        

    def multinomial_kl(self, log_prob1, log_prob2, weights=None):
        if weights is None:
            weights = torch.ones_like(log_prob1)
        kl = (weights * log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        mask_active = self.log_onehot_to_mask_active(log_x_t)
        log_probs_active = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_active_classes)
        )

        return log_probs_active * torch.einsum('bk...,k->bk...', mask_active, self.active_classes_tensor) + log_x_t * (1.0 - mask_active)

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)

        mask_active = self.log_onehot_to_mask_active(log_x_start)
        log_probs_active = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        
        return log_probs_active * torch.einsum('bk...,k->bk...', mask_active, self.active_classes_tensor) + log_x_start * (1.0 - mask_active)

    def predict_start(self, log_x_t, t):
        x_t = log_onehot_to_index(log_x_t)

        out = self._denoise_fn(t, x_t)

        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes
        assert out.size()[2:] == x_t.size()[1:]
        log_pred = F.log_softmax(out, dim=1)
        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t):
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0)


        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = \
            unnormed_logprobs \
            - torch.logsumexp(unnormed_logprobs, dim=1, keepdim=True)

        return log_EV_xtmin_given_xt_given_xstart

    def p_pred(self, log_x, t):
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(log_x, t=t)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, t=t)
        else:
            raise ValueError
        return log_model_pred
    
    
    def log_prob_to_log_score(self,log_prob,log_x,t):
        log_a = log_prob - self.q_pred_one_timestep(log_x,t)
        log_a_T_1 = log_sum_exp(log_a)
        log_score = log_a_T_1 + self.q_pred_one_timestep(log_a - log_a_T_1, t)
        return log_score
    
    def log_score_to_log_prob(self,log_score,log_x,t):
        #inverse the score
        log_alpha_t = extract(self.log_alpha, t, log_x.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x.shape)

        #we cannot use the log representation in case there are negative components
        log_score_T_1 = log_sum_exp(log_score)
        u = log_score.exp() - (log_1_min_alpha_t - np.log(self.num_classes) + log_score_T_1).exp()
        
        #threshold the negative values to zero
        u = u.clamp(min=1e-30)
        log_u = torch.log(u)
        #renormalize
        log_u = log_u - log_sum_exp(log_u)        

        #compute log_score from the formula
        log_v = self.q_pred_one_timestep(log_x,t)
        log_prob = log_v + log_u - log_alpha_t

        return log_prob

    @torch.no_grad()
    def p_sample(self, log_x, t, constraint=None, categories=None):
        model_log_prob = self.p_pred(log_x=log_x, t=t)
        if constraint is not None:
            log_score = self.log_prob_to_log_score(model_log_prob,log_x,t)
            new_log_score = log_score + constraint(log_x,t)
            new_model_log_prob = self.log_score_to_log_prob(new_log_score,log_x,t)
            model_log_prob = new_model_log_prob
        out = self.log_sample_categorical(model_log_prob, categories)
        return out

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.log_alpha.device

        b = shape[0]
        # start with random normal image.
        img = torch.randn(shape, device=device)

        for i in reversed(range(1, self.num_timesteps)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img
    
    @torch.no_grad()
    def _sample(self, image_size, batch_size = 16):
        return self.p_sample_loop((batch_size, 3, image_size, image_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in reversed(range(0, t)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def log_sample_categorical(self, logits, categories=None):
        if categories is not None:
            exclude_categories = [k for k in range(self.num_classes) if not k in categories]
            logits[:, exclude_categories,...] = np.log(1e-30) #set the logit to minus infinity
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)

        #we should have that we have sampled in the 'categories'
        if categories is not None:
            device = self.log_alpha.device
            values = torch.tensor(categories).to(device)
            assert torch.isin(sample, values).all()
        
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def nll(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        loss = 0
        for t in range(0, self.num_timesteps):
            t_array = (torch.ones(b, device=device) * t).long()

            kl = self.compute_Lt(
                log_x_start=log_x_start,
                log_x_t=self.q_sample(log_x_start=log_x_start, t=t_array),
                t=t_array)

            loss += kl

        loss += self.kl_prior(log_x_start)

        return loss

    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)

    def compute_Lt(self, log_x_start, log_x_t, t, detach_mean=False, weighted=False):
        log_true_prob = self.q_posterior(
            log_x_start=log_x_start, log_x_t=log_x_t, t=t)

        log_model_prob = self.p_pred(log_x=log_x_t, t=t)

        if detach_mean:
            log_model_prob = log_model_prob.detach()
        
        weights = torch.ones_like(log_model_prob)
        #reweighting the KL
        if weighted:
            weights = torch.where(log_x_start == 0., 10.0, 1.0).float()

        kl = self.multinomial_kl(log_true_prob, log_model_prob, weights)
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl

        return loss

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def _train_loss(self, x):
        b, device = x.size(0), x.device

        if self.loss_type == 'vb_stochastic':
            x_start = x

            t, pt = self.sample_time(b, device, 'importance')

            log_x_start = index_to_log_onehot(x_start, self.num_classes)

            kl = self.compute_Lt(
                log_x_start, self.q_sample(log_x_start=log_x_start, t=t), t, weighted=self.loss_weighted)

            Lt2 = kl.pow(2)
            Lt2_prev = self.Lt_history.gather(dim=0, index=t)
            new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
            self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
            self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

            kl_prior = self.kl_prior(log_x_start)

            # Upweigh loss term of the kl
            vb_loss = kl / pt + kl_prior

            return -vb_loss

        elif self.loss_type == 'vb_all':
            # Expensive, dont do it ;).
            return -self.nll(x)
        else:
            raise ValueError()

    def log_prob(self, x):
        b, device = x.size(0), x.device
        if self.training:
            return self._train_loss(x)

        else:
            log_x_start = index_to_log_onehot(x, self.num_classes)

            t, pt = self.sample_time(b, device, 'importance')

            kl = self.compute_Lt(
                log_x_start, self.q_sample(log_x_start=log_x_start, t=t), t)

            kl_prior = self.kl_prior(log_x_start)

            # Upweigh loss term of the kl
            loss = kl / pt + kl_prior

            return -loss

    def log_starting_noise(self, b, x):
        device = self.log_alpha.device
        uniform_logits = torch.zeros((b, self.num_classes) + self.shape, device=device)
        log_x = index_to_log_onehot(x, num_classes=self.num_classes)
        mask_active = self.log_onehot_to_mask_active(log_x)
        mask_active = mask_active.unsqueeze(2) #to match self.shape that has a 'channel' dimension
        log_x = log_x.unsqueeze(2) #to match self.shape that has a 'channel' dimension
        
        return mask_active * uniform_logits + (1.0 - mask_active) * log_x

    def sample(self, num_samples, x):
        #set always the same seed for the samples, so as to compare 
        #the samples between epochs
        
        rng_state = torch.get_rng_state()
        torch.manual_seed(8)
        
        #in Partial Diffusion we need to randomly draw inactive pixels. 
        #in order to do this, we are just gonna take samples of the database.
        #so we must have as much data samples as asked in 'num_samples'

        assert num_samples == x.size()[0]

        b = num_samples
        device = self.log_alpha.device
        log_z = self.log_starting_noise(b, x)
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Sample timestep {i:4d}', end='\r')

            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t)
        
        #set back the random state before sampling
        torch.set_rng_state(rng_state)
        return log_onehot_to_index(log_z)

    def sample_chain(self, num_samples, x, constraint=None):

        #in Partial Diffusion we need to randomly draw inactive pixels. 
        #in order to do this, we are just gonna take samples of the database.
        #so we must have as much data samples as asked in 'num_samples'

        assert num_samples == x.size()[0]

        b = num_samples
        device = self.log_alpha.device
        log_z = self.log_starting_noise(b, x)
        zs = torch.zeros((self.num_timesteps+1, b) + self.shape).long()
        zs[-1] = log_onehot_to_index(log_z)
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Chain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t,constraint=constraint)
            zs[i] = log_onehot_to_index(log_z)
        print()
        return zs

class MultimodalMultinomialDiffusion(MultinomialDiffusion):

    def __init__(self, num_classes, shape, denoise_fn, modalities_vaes: List[VAE], timesteps=1000,
                 loss_type='vb_stochastic', loss_weighted=False, parametrization='x0', strategy: str = "mixture_of_experts"):
        
        assert strategy in ["mixture_of_experts", "concatenation"]

        super().__init__(
            num_classes,
            shape,
            denoise_fn,
            timesteps=1000,
            loss_type='vb_stochastic', 
            loss_weighted=False, 
            parametrization='x0',
        )
        self.strategy = strategy
        self.n_modalities = len(modalities_vaes)
        self.modalities_vaes = torch.nn.ModuleList(modalities_vaes)

    def encode(self, x: torch.Tensor):
        print("------------------")
        print("------------------")
        print("------------------")
        print(x.shape, x[:, 0, :, :].shape, type(x[:, 0, :, :]))

        if self.strategy.lower() == "mixture_of_experts":

            # sample a modality index
            modality_idx = random.randint(0, self.n_modalities-1)

            # encode
            vae = self.modalities_vaes[modality_idx]
            qz_x, px_z, z = vae(x[:, modality_idx, :, :])

            return z, modality_idx

        if self.strategy.lower() == "concatenation":

            # independently encode all the modalities
            qz_params_lst = []
            qz_x_dist_lst = []
            z_lst = []
            for modal_idx, vae in enumerate(self.modalities_vaes):

                _qz_x, _px_z, _z = vae(x[:, modal_idx, :, :])

                qz_params_lst.append(_qz_x_params)
                qz_x_dist_lst.append(_qz_x_dist)
                z_lst.append(_z)
            
            # concatenate the encoded modalities
            z = torch.concatenation(z_lst)

            print(z.shape)

            return z, None

        if self.strategy.lower() == "product_of_experts":

            # independently encode all the modalities
            qz_params_lst = []
            qz_x_dist_lst = []
            z_lst = []
            for modal_idx, vae in enumerate(self.modalities_vaes):

                _qz_x_params = vae.encode(x[:, modal_idx, :, :])
                _qz_x_dist = vae.get_encoding_dist()
                _z = _qz_x_dist(*_qz_x_params)

                qz_params_lst.append(_qz_x_params)
                qz_x_dist_lst.append(_qz_x_dist)
                z_lst.append(_z)
            
            z = torch.concatenation(z_lst) # remplace with product of experts

            print(z.shape)

            return z, None

    def log_prob(self, x):
        b, device = x.size(0), x.device
        if self.training:
            return self._train_loss(x)

        else:
            log_x_start = index_to_log_onehot(x, self.num_classes)

            t, pt = self.sample_time(b, device, 'importance')

            kl = self.compute_Lt(
                log_x_start, self.q_sample(log_x_start=log_x_start, t=t), t)

            kl_prior = self.kl_prior(log_x_start)

            # Upweigh loss term of the kl
            loss = kl / pt + kl_prior

            return -loss

    def _train_loss(self, x):
        b, device = x.size(0), x.device

        # get latent variable
        z, _ = self.encode(x)

        if self.loss_type == 'vb_stochastic':
            z_start = z

            t, pt = self.sample_time(b, device, 'importance')
            print(z_start)
            log_z_start = index_to_log_onehot(z_start, self.num_classes) # if the latent space is continuos, we cannot use multinomial diffusion

            kl = self.compute_Lt(
                log_z_start, self.q_sample(log_x_start=log_x_start, t=t), t, weighted=self.loss_weighted)

            Lt2 = kl.pow(2)
            Lt2_prev = self.Lt_history.gather(dim=0, index=t)
            new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
            self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
            self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

            kl_prior = self.kl_prior(log_z_start)

            # Upweigh loss term of the kl
            vb_loss = kl / pt + kl_prior

            return -vb_loss

        elif self.loss_type == 'vb_all':
            # Expensive, dont do it ;).
            return -self.nll(z)
        else:
            raise ValueError()