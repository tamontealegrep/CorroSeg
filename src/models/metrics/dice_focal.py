
import torch
import torch.nn as nn
from typing import Optional
from .dice import dice_index
from .focal import focal_index

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class DICEFocalLoss(nn.Module):
    """
    Combines DICE Loss and Focal Loss to address class imbalance in segmentation tasks,
    focusing on difficult examples with the Focal Loss focusing parameter and evaluating precision with DICE Loss.

    Parameters:
        dice_weight (float, optional): Weight for the DICE Loss component. Defaults to 1.0.
        focal_weight (float, optional): Weight for the Focal Loss component. Defaults to 1.0.
        gamma (float, optional): Focusing parameter for Focal Loss. Defaults to 2.0.
        delta (float, optional): Weighting factor (alpha) for false positives in Focal Loss. Defaults to 0.25.
        smooth (float, optional): Small constant to avoid log(0) and division by zero.
    """

    def __init__(self,
                 dice_weight: Optional[float] = 1.0,
                 focal_weight: Optional[float] = 1.0,
                 gamma: Optional[float] = 2.0,
                 delta: Optional[float] = 0.25,
                 smooth: Optional[float] = 1e-6):
        super(DICEFocalLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.delta = delta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        # Flatten the tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # Calculate the DICE index
        dice_idx = dice_index(outputs, targets, self.smooth)

        # Calculate the Focal index
        focal_idx = focal_index(outputs, targets, self.gamma, self.delta, self.smooth)

        # Combine DICE Loss and Focal Loss
        loss = (self.dice_weight * (1 - dice_idx) + self.focal_weight * focal_idx) / 2

        return loss

#-----------------------------------------------------------------------------------------------------------------------------------------------------