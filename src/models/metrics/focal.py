
import torch
import torch.nn as nn
from typing import Optional

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def focal_index(outputs: torch.Tensor,
                targets: torch.Tensor,
                alpha: Optional[float] = 0.25,
                gamma: Optional[float] = 2.0,
                smooth: Optional[float] = 1e-6,
                ) -> torch.Tensor:
    """
    Calculate the Focal Loss index for each example (pixel) in the batch.

    Parameters:
        outputs (torch.Tensor): Predicted outputs after applying sigmoid, shape (N,) where N is the number of pixels.
        targets (torch.Tensor): Ground truth labels, shape (N,) for binary classification (0 for background, 1 for object).
        alpha (float, optional): Balancing factor for the classes, default is 0.25.
        gamma (float, optional): Focusing parameter to focus on hard-to-classify examples, default is 2.0.
        smooth (float, optional): Small constant to avoid log(0).

    Returns:
        (torch.Tensor): The computed Focal Loss for each pixel in the batch.
    """
    p_t = outputs * targets + (1 - outputs) * (1 - targets)

    return - alpha * (1 - p_t) ** gamma * torch.log(p_t + smooth)

class FocalLoss(nn.Module):
    """
    Focal Loss function, designed to address class imbalance by focusing more on hard-to-classify examples and reducing
    the loss contribution from easy-to-classify examples.

    Parameters:
        alpha (float, optional): Balancing factor for the classes, default is 0.25.
        gamma (float, optional): Focusing parameter to focus on hard-to-classify examples, default is 2.0.
        smooth (float, optional): Small constant to avoid log(0).
    """
    def __init__(self, 
                alpha: Optional[float] = 1.5,
                gamma: Optional[float] = 2.0,
                smooth: Optional[float] = 1e-6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, outputs:torch.Tensor, targets:torch.Tensor) -> float:
        # Flatten the tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # Calculate the Focal index using the external function
        focal_idx = focal_index(outputs, targets, self.alpha, self.gamma, self.smooth)

        # Return the mean focal loss over all pixels in the batch
        return focal_idx.mean()
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------