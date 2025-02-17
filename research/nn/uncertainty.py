import torch
import numpy as np


def gini_np(probs, axis=1):
    return 1 - np.sum(probs**2, axis=axis, keepdims=True)


def gini_torch(probs, axis=1):
    return 1 - torch.sum(probs**2, dim=axis, keepdim=True)


def gini_logits_np(logits, temperature):
    probs = np.exp(logits / temperature)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return gini_np(probs)


def gini_logits_torch(logits, temperature):
    probs = torch.softmax(logits / temperature, dim=1)
    return gini_torch(probs)


def doctor_np(probs, axis=1):
    return gini_np(probs, axis=axis) / (1 - gini_np(probs, axis=axis))


def doctor_torch(probs, axis=1):
    return gini_torch(probs, axis=axis) / (1 - gini_torch(probs, axis=axis))


def doctor_logits_np(logits, temperature):
    probs = np.exp(logits / temperature)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return doctor_np(probs)


def doctor_logits_torch(logits, temperature):
    probs = torch.softmax(logits / temperature, dim=1)
    return doctor_torch(probs)


def odin_np(probs, axis=1):
    return 1 - np.max(probs, axis=axis, keepdims=True)


def odin_torch(probs, axis=1):
    return 1 - torch.max(probs, dim=axis, keepdim=True)[0]


def odin_logits_np(logits, temperature):
    probs = np.exp(logits / temperature)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return odin_np(probs)


def odin_logits_torch(logits, temperature):
    probs = torch.softmax(logits / temperature, dim=1)
    return odin_torch(probs)


def msp_logits_np(logits):
    return odin_logits_np(logits, 1)


def msp_logits_torch(logits):
    return odin_logits_torch(logits, 1)


def energy_logits_np(logits, temperature):
    terms = np.exp(logits / temperature)
    return -temperature * np.log(np.sum(terms, axis=1, keepdims=True))


def energy_logits_torch(logits, temperature):
    terms = torch.exp(logits / temperature)
    return -temperature * torch.log(torch.sum(terms, dim=1, keepdim=True))


def shannon_entropy_np(probs, axis=1):
    return -np.sum(probs * np.log(probs), axis=axis, keepdims=True)


def shannon_entropy_torch(probs, axis=1):
    return -torch.sum(probs * torch.log(probs), dim=axis, keepdim=True)


def shannon_entropy_logits_np(logits, temperature):
    probs = np.exp(logits / temperature)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return shannon_entropy_np(probs)


def shannon_entropy_logits_torch(logits, temperature):
    probs = torch.softmax(logits / temperature, dim=1)
    return shannon_entropy_torch(probs)
