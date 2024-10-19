
import torch
import torch.nn as nn

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def tversky_index(outputs:torch.Tensor, targets:torch.Tensor, alpha:float=0.5, beta:float=0.5, smooth:float=1e-6) -> float:
    """
    Calculate the Tversky index between the predicted outputs and the ground truth targets.

    Args:
        outputs (torch.Tensor): Predicted outputs after applying sigmoid, shape (N,).
        targets (torch.Tensor): Ground truth binary targets, shape (N,).
        alpha (float): Weight for false positives.
        beta (float): Weight for false negatives.
        smooth (float): Small constant to avoid division by zero.

    Returns:
        float: Tversky index value.
    """
    intersection = (outputs * targets).sum()
    false_positive = ((1 - targets) * outputs).sum()
    false_negative = (targets * (1 - outputs)).sum()

    return (intersection + smooth) / (intersection + alpha * false_positive + beta * false_negative + smooth)

class TverskyLoss(nn.Module):
    """
    Tversky Loss function, designed for imbalanced class segmentation tasks.

    Args:
        alpha (float): Weight for false positives.
        beta (float): Weight for false negatives.
        smooth (float): Small constant to avoid division by zero.
    """

    def __init__(self, alpha:float=0.5, beta:float=0.5, smooth:float=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, outputs:torch.Tensor, targets:torch.Tensor) -> float:
        # Apply sigmoid to the outputs if necessary
        outputs = torch.sigmoid(outputs)

        # Flatten the tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # Calculate the Tversky index using the external function
        tversky_idx = tversky_index(outputs, targets, self.alpha, self.beta, self.smooth)

        # Tversky Loss is 1 - Tversky index
        return 1 - tversky_idx

#-----------------------------------------------------------------------------------------------------------------------------------------------------