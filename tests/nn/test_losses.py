import torch
import torch.nn.functional as F
from research.nn.losses import mse_loss as mse_loss_
from research.nn.losses.mse import mat_mse_loss, mat_se
from testdata import dummy


def test_mse():
    l_ = mse_loss_(dummy.scalar_logits, dummy.scalar_target)
    lm = mat_mse_loss(dummy.scalar_logits, dummy.scalar_target)
    l = F.mse_loss(dummy.scalar_logits, dummy.scalar_target)
    assert torch.eq(l, l_) and torch.eq(l, lm)
