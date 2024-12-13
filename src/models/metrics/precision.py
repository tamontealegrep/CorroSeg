import torch
import torch.nn as nn
from typing import Optional

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def precision_index(outputs: torch.Tensor,
                    targets: torch.Tensor,
                    smooth: Optional[float] = 1e-6,
                    ) -> float:
    """
    Calculate the precision of the model, i.e., the proportion of true positive predictions 
    among all predicted positives.

    Intuition: Out of all the instances that were predicted as positive by the model, 
               how many were actually positive?

    Parameters:
        outputs (torch.Tensor): Predicted outputs, shape (N,).
        targets (torch.Tensor): Ground truth binary targets, shape (N,).
        smooth (float, optional): Small constant to avoid division by zero.

    Returns:
        (float): Precision value.
    """
    TP = (outputs * targets).sum()
    FP = ((1 - targets) * outputs).sum()
    return TP / (TP + FP + smooth)


def negative_precision_index(outputs: torch.Tensor,
                             targets: torch.Tensor,
                             smooth: Optional[float] = 1e-6,
                             ) -> float:
    """
    Calculate the negative precision, i.e., the proportion of true negatives 
    among all predicted negatives.

    Intuition: Out of all the instances that were classified as negative by the model, 
               how many were actually negative?

    Parameters:
        outputs (torch.Tensor): Predicted outputs, shape (N,).
        targets (torch.Tensor): Ground truth binary targets, shape (N,).
        smooth (float, optional): Small constant to avoid division by zero.

    Returns:
        (float): Negative precision value.
    """
    TN = ((1 - outputs) * (1 - targets)).sum()
    FN = ((1 - outputs) * targets).sum()
    return TN / (TN + FN + smooth) 

#-----------------------------------------------------------------------------------------------------------------------------------------------------