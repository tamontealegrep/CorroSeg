
import torch
import torch.nn as nn

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def iou_index(outputs:torch.Tensor, targets:torch.Tensor, smooth:float=1e-6) -> float:
    """
    Calculate the IoU (Intersection over Union) index between the predicted outputs and the ground truth targets.

    Args:
        outputs (torch.Tensor): Predicted outputs after applying sigmoid, shape (N,).
        targets (torch.Tensor): Ground truth binary targets, shape (N,).
        smooth (float): Small constant to avoid division by zero.

    Returns:
        float: IoU index value.
    """
    intersection = (outputs * targets).sum()
    total = outputs.sum() + targets.sum()  # Union

    return (intersection + smooth) / (total - intersection + smooth)

class IoULoss(nn.Module):
    """
    IoU (Intersection over Union) Loss function, designed for imbalanced class segmentation tasks.

    Args:
        smooth (float): Small constant to avoid division by zero.
    """

    def __init__(self, smooth:float=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs:torch.Tensor, targets:torch.Tensor) -> float:
        # Flatten the tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # Calculate the IoU index using the external function
        iou_idx = iou_index(outputs, targets, self.smooth)

        # IoU Loss is 1 - IoU index
        return 1 - iou_idx

#-----------------------------------------------------------------------------------------------------------------------------------------------------
