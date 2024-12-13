import torch
import torch.nn as nn
from typing import Optional

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def recall_index(outputs: torch.Tensor,
                 targets: torch.Tensor,
                 smooth: Optional[float] = 1e-6,
                 ) -> float:
    """
    Calculate the recall of the model, i.e., the proportion of true positive predictions 
    among all actual positives.

    Intuition: Out of all the instances that are actually positive, how many did the model 
               correctly identify as positive?

    Parameters:
        outputs (torch.Tensor): Predicted outputs, shape (N,).
        targets (torch.Tensor): Ground truth binary targets, shape (N,).
        smooth (float, optional): Small constant to avoid division by zero.

    Returns:
        (float): Recall value.
    """
    TP = (outputs * targets).sum().item()
    FN = ((1 - outputs) * targets).sum().item()
    return TP / (TP + FN + smooth)


def negative_recall_index(outputs: torch.Tensor,
                          targets: torch.Tensor,
                          smooth: Optional[float] = 1e-6,
                          ) -> float:
    """
    Calculate the negative recall, i.e., the proportion of true negatives among all actual negatives.

    Intuition: Out of all the instances that are actually negative, how many did the model 
               correctly identify as negative?

    Parameters:
        outputs (torch.Tensor): Predicted outputs, shape (N,).
        targets (torch.Tensor): Ground truth binary targets, shape (N,).
        smooth (float, optional): Small constant to avoid division by zero.

    Returns:
        (float): Negative recall value.
    """
    TN = ((1 - outputs) * (1 - targets)).sum().item()
    FP = (outputs * (1 - targets)).sum().item()
    return TN / (TN + FP + smooth)

#-----------------------------------------------------------------------------------------------------------------------------------------------------