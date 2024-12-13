
import torch
import torch.nn as nn
from typing import Optional
from .tversky import tversky_index
from .focal import focal_index

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class TverskyFocalLoss(nn.Module):
    """
    Combines Tversky Loss and Focal Loss to address class imbalance in segmentation tasks,
    focusing on difficult examples with the Focal Loss focusing parameter and evaluating precision with Tversky Loss.

    Parameters:
        tversky_weight (float, optional): Weight for the Tversky Loss component. Defaults to 1.0.
        focal_weight (float, optional): Weight for the Focal Loss component. Defaults to 1.0.
        alpha (float, optional): Weighting factor for false positives in Tversky Loss. Defaults to 0.5.
        beta (float, optional): Weighting factor for false negatives in Tversky Loss. Defaults to 0.5.
        gamma (float, optional): Focusing parameter for Focal Loss. Defaults to 2.0.
        delta (float, optional): Balancing factor (alpha) for the calculation of Focal Loss. Defaults to 0.25.
        smooth (float, optional): Small constant to avoid log(0) and division by zero.
    """
    def __init__(self,
                 tversky_weight: Optional[float] = 1.0,
                 focal_weight: Optional[float] = 1.0,
                 alpha: Optional[float] = 0.5,
                 beta: Optional[float] = 0.5,
                 gamma: Optional[float] = 2.0,
                 delta: Optional[float] = 0.25,
                 smooth: Optional[float] = 1e-6):
        super(TverskyFocalLoss, self).__init__()
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.smooth = smooth

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        # Flatten the tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # Calculate the Tversky index
        tversky_idx = tversky_index(outputs, targets, self.alpha, self.beta, self.smooth)

        # Calculate the Focal index
        focal_idx = focal_index(outputs, targets, self.gamma, self.delta, self.smooth)

        loss = (self.tversky_weight * (1 - tversky_idx) + self.focal_weight * focal_idx) / 2

        return loss

#-----------------------------------------------------------------------------------------------------------------------------------------------------