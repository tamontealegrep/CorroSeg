
import torch
import torch.nn as nn
from typing import Optional
from .mcc import mcc_index
from .focal import focal_index

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class MCCFocalLoss(nn.Module):
    """
    Combines MCC Loss and Focal Loss to address class imbalance in segmentation tasks,
    focusing on difficult examples with the Focal Loss focusing parameter and the overall confusion balance MCC Loss.

    Parameters:
        mcc_weight (float, optional): Weight factor for the MCC component of the loss.
        focal_weight (float, optional): Weight factor for the Focal Loss component of the loss.
        gamma (float, optional): Focusing parameter for Focal Loss.
        delta (float, optional): Balancing parameter (alpha) for false positives in Tversky/Focal Loss.
        smooth (float, optional): Small constant to avoid log(0) and division by zero.
    """

    def __init__(self, 
                 mcc_weight: float = 1.0,
                 focal_weight: float = 1.0,
                 gamma: float = 2.0,
                 delta: float = 0.25,
                 smooth: float = 1e-6):
        super(MCCFocalLoss, self).__init__()
        self.mcc_weight = mcc_weight
        self.focal_weight = focal_weight
        self.gamma = gamma
        self.delta = delta
        self.smooth = smooth

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Flatten outputs and targets
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # Calculate MCC component
        mcc_idx = mcc_index(outputs, targets, self.smooth)

        # Calculate Focal Loss
        focal_idx = focal_index(outputs, targets, self.gamma, self.delta, self.smooth)

        loss = (self.mcc_weight * (1 - mcc_idx) + self.focal_weight * focal_idx) / 2

        return loss

#-----------------------------------------------------------------------------------------------------------------------------------------------------