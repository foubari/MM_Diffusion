# objectives of choice
import torch
from numpy import prod
from utils import kl_divergence, is_multidata, log_mean_exp


# Helper to vectorise computation
def compute_microbatch_split(x, K):
    """ Checks if batch needs to be broken down further to fit in memory. """
    B = x[0].size(0) if is_multidata(x) else x.size(0)
    S = sum([1.0 / (K * prod(_x.size()[1:])) for _x in x]) if is_multidata(x) \
        else 1.0 / (K * prod(x.size()[1:]))
    S = int(1e8 * S)  # float heuristic for 12Gb cuda memory
    assert (S > 0), "Cannot fit individual data in memory, consider smaller K"
    return min(B, S)

def elbo(model, x, beta, K=1):
    """Computes E_{p(x)}[ELBO] """
    qz_x, px_z, _ = model(x)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1).sum(-1) 
    kld = kl_divergence(qz_x, model.pz(*model.pz_params))
    return (lpx_z.sum(-1) - beta*kld.sum(-1)).mean(0).sum(), -lpx_z.sum(-1).mean(0).sum(), kld.sum(-1).mean(0).sum()


def iwae(model, x, beta, K): #_iwae
    """IWAE estimate for log p_\theta(x) -- fully vectorised."""
    qz_x, px_z, zs = model(x, K)
    lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1)
    lqz_x = qz_x.log_prob(zs).sum(-1)
    return lpx_z.sum(-1) + beta*(lpz - lqz_x), -lpx_z.sum(-1), lqz_x-lpz 



def dreg_helper(model, x, beta,  K): #_dreg
    """DREG estimate for log p_\theta(x) -- fully vectorised."""
    _, px_z, zs = model(x, K)
    lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1).sum(-1)
    qz_x = model.qz_x(*[p.detach() for p in model.qz_x_params])  # stop-grad for \phi
    lqz_x = qz_x.log_prob(zs).sum(-1)
    neg_loglikelihood = -lpx_z.sum(-1)
    kl = lqz_x - lpz 
    lw = -neg_loglikelihood - beta*kl
    return lw, zs, neg_loglikelihood, kl

def dreg(model, x, beta, K, regs=None):
    lw, zs, neg_loglikelihood, kl = dreg_helper(model, x, beta,  K)
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 0, keepdim=True)).exp()
        if zs.requires_grad:
            zs.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    return (grad_wt * lw).sum(), neg_loglikelihood, kl



# def iwae(model, x, beta , K):
#     """Computes an importance-weighted ELBO estimate for log p_\theta(x)
#     Iterates over the batch as necessary.
#     """
#     S = compute_microbatch_split(x, K)
#     lw = torch.cat([_iwae(model, _x, beta, K) for _x in x.split(S)], 1)  # concat on batch
#     return log_mean_exp(lw).sum()




# # Original implementation


# def _dreg(model, x, beta,  K): #_dreg
#     """DREG estimate for log p_\theta(x) -- fully vectorised."""
#     _, px_z, zs = model(x, K)
#     lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
#     lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1) 
#     qz_x = model.qz_x(*[p.detach() for p in model.qz_x_params])  # stop-grad for \phi
#     lqz_x = qz_x.log_prob(zs).sum(-1)
#     lw = lpx_z.sum(-1) + beta*(lpz  - lqz_x)
#     return lw, zs




# def dreg(model, x, beta, K, regs=None):
#     """Computes a doubly-reparameterised importance-weighted ELBO estimate for log p_\theta(x)
#     Iterates over the batch as necessary.
#     """
#     S = compute_microbatch_split(x, K)
#     lw, zs = zip(*[_dreg(model, _x,beta, K) for _x in x.split(S)])
#     lw = torch.cat(lw, 1)  # concat on batch
#     zs = torch.cat(zs, 1)  # concat on batch
#     with torch.no_grad():
#         grad_wt = (lw - torch.logsumexp(lw, 0, keepdim=True)).exp()
#         if zs.requires_grad:
#             zs.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
#     return (grad_wt * lw).sum()

