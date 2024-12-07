
import torch
import torch.nn as nn
from typing import Optional
from .iou import iou_index
from .focal import focal_index

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class IoUFocalLoss(nn.Module):
    """
    Combines Intersection over Union (IoU) and Focal Loss to address class imbalance. This loss function emphasizes 
    hard-to-classify examples while reducing the contribution of easy-to-classify ones, and integrates IoU to improve
    the evaluation of segmentation accuracy.

    Parameters:
        iou_weight (float, optional): Weight factor for the IoU component of the loss. Defaults to 1.0.
        focal_weight (float, optional): Weight factor for the Focal Loss component. Defaults to 1.0.
        alpha (float, optional): Balancing factor for the classes, default is 0.25.
        gamma (float, optional): Focusing parameter to focus on hard-to-classify examples, default is 2.0.
        smooth (float, optional): Small constant to avoid log(0) and division by zero.
    """
    def __init__(self,
                iou_weight: Optional[float] = 1.0,
                focal_weight: Optional[float] = 1.0,
                alpha: Optional[float] = 1.5,
                gamma: Optional[float] = 2.0,
                smooth: Optional[float] = 1e-6):
        super(IoUFocalLoss, self).__init__()
        self.iou_weight = iou_weight
        self.focal_weight = focal_weight
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        
    def forward(self, outputs:torch.Tensor, targets:torch.Tensor) -> float:
        # Flatten the tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # Calculate the IoU index
        iou_idx = iou_index(outputs, targets, self.smooth)

        # Calculate the Focal index
        focal_idx = focal_index(outputs, targets, self.alpha, self.gamma, self.smooth)

        loss = (self.iou_weight * (1 - iou_idx) + self.focal_weight * focal_idx) / 2

        return loss

#-----------------------------------------------------------------------------------------------------------------------------------------------------