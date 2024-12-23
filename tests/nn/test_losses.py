import torch
import torch.nn.functional as F
from research.nn.activations.softmax import log_softmax_math
from research.nn.losses.mse import mse_loss as mse_loss_
from research.nn.losses.ce import ce, nll
from research.nn.losses.mse import matrix_mse_loss, mat_se
from ..testdata import dummy


def test_mse():
    l_ = mse_loss_(dummy.scalar_logits_train, dummy.scalar_logits_train)
    lm = matrix_mse_loss(dummy.scalar_logits_train, dummy.scalar_logits_train)
    l = F.mse_loss(dummy.scalar_logits_train, dummy.scalar_logits_train)
    assert torch.eq(l, l_) and torch.eq(l, lm)


def test_ce():
    l1 = nll(log_softmax_math(dummy.logits), dummy.target)
    l2 = F.nll_loss(F.log_softmax(dummy.logits, dim=-1), dummy.target)
    l3 = F.cross_entropy(dummy.logits, dummy.target)
    assert torch.eq(l1, l2) and torch.eq(l1, l3)
