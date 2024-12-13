
import torch
import torch.nn as nn
from typing import Optional

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def dice_index(outputs: torch.Tensor,
               targets: torch.Tensor,
               smooth: Optional[float] = 1e-6,
               ) -> float:
    """
    Calculate the DICE index between the predicted outputs and the ground truth targets.

    Parameters:
        outputs (torch.Tensor): Predicted outputs after applying sigmoid, shape (N,).
        targets (torch.Tensor): Ground truth binary targets, shape (N,).
        smooth (float, optional): Small constant to avoid division by zero.

    Intuition: The DICE index measures the overlap between the predicted positive region and the actual 
               ground truth positive region. It quantifies how well the model's predictions match the true positives. 
               A DICE index close to 1 indicates a high degree of overlap, while a value close to 0 suggests poor performance.

    Returns:
        (float): DICE index value.
    """
    intersection = (outputs * targets).sum()
    return (2. * intersection + smooth) / (outputs.sum() + targets.sum() + smooth)

class DICELoss(nn.Module):
    """
    DICE Loss function, designed for imbalanced class segmentation tasks.

    Parameters:
        smooth (float, optional): Small constant to avoid division by zero.
    """

    def __init__(self, smooth: Optional[float] = 1e-6):
        super(DICELoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs:torch.Tensor, targets:torch.Tensor) -> float:
        # Flatten the tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # Calculate the DICE index using the external function
        dice_idx = dice_index(outputs, targets, self.smooth)

        # DICE Loss is 1 - DICE index
        return 1 - dice_idx

#-----------------------------------------------------------------------------------------------------------------------------------------------------