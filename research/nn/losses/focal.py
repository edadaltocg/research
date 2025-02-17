from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        weights=None,
        gamma=2.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.weights = weights
        self.gamma = gamma
        self.reduction = reduction
        self._reduction_op = reduction_ops[reduction]

    def forward(self, inputs, targets):
        # Apply softmax to get probabilities
        probs = F.softmax(inputs, dim=1)

        # Gather the probabilities of the true class
        targets = targets.view(-1, 1)
        probs = probs.gather(1, targets).squeeze(1)

        # Compute the focal loss components
        log_probs = torch.log(probs)
        focal_loss = -((1 - probs) ** self.gamma) * log_probs

        # Apply class weighting if alpha is provided
        if self.weights is not None:
            alpha = self.weights[targets]
            focal_loss = alpha * focal_loss
