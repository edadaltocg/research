import numpy as np
import torch
from torch import nn


def langevin_sample(model: nn.Module, dim=2, n_steps=60, step_size=10.0, noise=0.005, n=256):
    device = next(model.parameters()).device
    # start from noise, roll downhill on the energy
    x = torch.rand(n, dim, device=device) * 4 - 2  # uniform in [-2,2]^dim
    x.requires_grad_(True)
    for _ in range(n_steps):
        e = model(x).sum()
        grad = torch.autograd.grad(e, x)[0]
        grad = grad.clamp(-1, 1)
        x = x - step_size * grad + noise * torch.randn_like(x)
        x = x.detach().requires_grad_(True)
    return x.detach()


def cd_loss(model, x_pos, x_neg, alpha=0.01):
    e_pos = model(x_pos)
    e_neg = model(x_neg)
    loss = e_pos.mean() - e_neg.mean()
    loss = loss + alpha * (e_pos**2 + e_neg**2).mean()
    return loss


def score_matching_loss(model, x):
    x = x.clone().requires_grad_(True)
    E = model(x)
    # score = -dE/dx
    grad = torch.autograd.grad(E.sum(), x, create_graph=True)[0]
    grad_norm = 0.5 * (grad**2).sum(dim=1)

    # trace of Hessian via exact per-dim autodiff (fine for 2D)
    trace = 0.0
    for i in range(x.shape[1]):
        d2 = torch.autograd.grad(grad[:, i].sum(), x, create_graph=True)[0][:, i]
        trace = trace + d2
    return (grad_norm + trace).mean()  # note: +trace (from the -(-lap))


def dsm_loss(model, x, sigma=0.1):
    noise = torch.randn_like(x) * sigma
    x_tilde = (x + noise).requires_grad_(True)
    E = model(x_tilde)
    grad = torch.autograd.grad(E.sum(), x_tilde, create_graph=True)[0]
    target = -noise / sigma**2  # true score of the noised data
    return 0.5 * ((grad - target) ** 2).sum(dim=1).mean()


def nce_loss(model, x_data, log_Z, noise_std=2.0):
    x_noise = torch.randn_like(x_data) * noise_std
    # log density of Gaussian noise
    log_pn = lambda x: (-0.5 * (x / noise_std) ** 2 - np.log(noise_std) - 0.5 * np.log(2 * np.pi)).sum(1)

    def logit(x):  # log p_theta - log p_n
        return (-model(x) - log_Z) - log_pn(x)

    loss_data = -torch.nn.functional.logsigmoid(logit(x_data)).mean()
    loss_noise = -torch.nn.functional.logsigmoid(-logit(x_noise)).mean()
    return loss_data + loss_noise


def margin_loss(model, x_pos, x_neg, margin=1.0):
    return torch.clamp(margin + model(x_pos) - model(x_neg), min=0).mean()


def preference_loss(model, x_pos, x_neg):
    # softplus version of the margin. Used in RLHF reward models
    return -torch.nn.functional.logsigmoid(model(x_neg) - model(x_pos)).mean()


def infonce_loss(model, x_pos, x_negs):  # x_negs: (batch, K, dim)
    e_pos = -model(x_pos).unsqueeze(1)  # (B,1)
    e_neg = -model(x_negs.reshape(-1, x_negs.shape[-1]))
    e_neg = e_neg.reshape(x_negs.shape[0], -1)  # (B,K)
    logits = torch.cat([e_pos, e_neg], dim=1)  # (B,1+K)
    labels = torch.zeros(len(x_pos), dtype=torch.long)  # positive is index 0
    return torch.nn.functional.cross_entropy(logits, labels)
