import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size, dim, n_classes = 128, 16, 3


class dummy:
    logits = torch.randn(batch_size, n_classes)
    target = torch.randint(n_classes, size=(batch_size,), dtype=torch.long)

    binary_logits = torch.randn(batch_size, 1)
    binary_targets = torch.randint(1, size=(batch_size,), dtype=torch.long)

    scalar_logits_train = torch.randn(batch_size, 1)
    scalar_logits_val = torch.randn(batch_size, 1)
    scalar_logits_test = torch.randn(batch_size, 1)

    scalar_target_train = torch.randn(batch_size, 1)
    scalar_target_val = torch.randn(batch_size, 1)
    scalar_target_test = torch.randn(batch_size, 1)

    x_vector_train = torch.randn(batch_size, dim)
    x_vector_val = torch.randn(batch_size, dim)
    x_vector_test = torch.randn(batch_size, dim)
