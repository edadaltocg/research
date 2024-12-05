import torch
from research.nn.models.glm import LinearRegression
from ..testdata import dummy
import torch.nn.functional as F


def test_linear_regression():
    model = LinearRegression.analytical_solution(dummy.x_vector_train, dummy.scalar_target_train)
    logits = model(dummy.x_vector_test)
    error = F.mse_loss(logits, dummy.scalar_target_test)
    print(error)
