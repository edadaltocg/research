import torch

from research.nn.evals.regression import r_squared_math
from ..testdata import dummy


def test_r_squared():
    r2 = r_squared_math(dummy.scalar_logits_test, dummy.scalar_target_test)
    assert torch.greater_equal(r2, torch.tensor(-torch.inf)) and torch.less_equal(r2, torch.tensor(1))
