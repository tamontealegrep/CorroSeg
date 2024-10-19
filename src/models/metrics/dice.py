
import torch
import torch.nn as nn

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def dice_index(outputs:torch.Tensor, targets:torch.Tensor, smooth:float=1e-6) -> float:
    """
    Calculate the DICE index between the predicted outputs and the ground truth targets.

    Args:
        outputs (torch.Tensor): Predicted outputs after applying sigmoid, shape (N,).
        targets (torch.Tensor): Ground truth binary targets, shape (N,).
        smooth (float): Small constant to avoid division by zero.

    Returns:
        float: DICE index value.
    """
    intersection = (outputs * targets).sum()
    return (2. * intersection + smooth) / (outputs.sum() + targets.sum() + smooth)

class DICELoss(nn.Module):
    """
    DICE Loss function, designed for imbalanced class segmentation tasks.

    Args:
        smooth (float): Small constant to avoid division by zero.
    """

    def __init__(self, smooth:float=1e-6):
        super(DICELoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs:torch.Tensor, targets:torch.Tensor) -> float:
        # Apply sigmoid to the outputs if necessary
        outputs = torch.sigmoid(outputs)

        # Flatten the tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # Calculate the DICE index using the external function
        dice_idx = dice_index(outputs, targets, self.smooth)

        # DICE Loss is 1 - DICE index
        return 1 - dice_idx

#-----------------------------------------------------------------------------------------------------------------------------------------------------