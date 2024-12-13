
import torch
import torch.nn as nn
from typing import Optional

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def focal_index(outputs: torch.Tensor,
                targets: torch.Tensor,
                gamma: Optional[float] = 2.0,
                delta: Optional[float] = 0.25,
                smooth: Optional[float] = 1e-6,
                ) -> float:
    """
    Calculate the Focal Loss index for each example (pixel) in the batch.

    Parameters:
        outputs (torch.Tensor): Predicted outputs after applying sigmoid, shape (N,) where N is the number of pixels.
        targets (torch.Tensor): Ground truth labels, shape (N,) for binary classification (0 for background, 1 for object).
        gamma (float, optional): Focusing parameter to focus on hard-to-classify examples, default is 2.0.
        delta (float, optional): Balancing factor (alpha) for the classes, default is 0.25.
        smooth (float, optional): Small constant to avoid log(0).

    Returns:
        (float): The mean Focal Loss in the batch.
    """
    p_t = outputs * targets + (1 - outputs) * (1 - targets)
    index = - delta * (1 - p_t) ** gamma * torch.log(p_t + smooth)
    return index.mean()

class FocalLoss(nn.Module):
    """
    Focal Loss function, designed to address class imbalance by focusing more on hard-to-classify examples and reducing
    the loss contribution from easy-to-classify examples.

    Parameters:
        gamma (float, optional): Focusing parameter to focus on hard-to-classify examples, default is 2.0.
        delta (float, optional): Balancing factor (alpha) for the classes, default is 0.25.
        smooth (float, optional): Small constant to avoid log(0).
    """
    def __init__(self, 
                gamma: Optional[float] = 2.0,
                delta: Optional[float] = 0.25,
                smooth: Optional[float] = 1e-6):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.delta = delta
        self.smooth = smooth

    def forward(self, outputs:torch.Tensor, targets:torch.Tensor) -> float:
        # Flatten the tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # Calculate the Focal index using the external function
        focal_idx = focal_index(outputs, targets, self.gamma, self.delta, self.smooth)

        # Return the mean focal loss over all pixels in the batch
        return focal_idx
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------