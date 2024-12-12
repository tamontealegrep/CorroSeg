import torch
import torch.nn as nn

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def precision_index(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate the precision of the model, i.e., the proportion of true positive predictions 
    among all predicted positives.

    Intuition: Out of all the instances that were predicted as positive by the model, 
               how many were actually positive?

    Parameters:
        outputs (torch.Tensor): Predicted outputs, shape (N,).
        targets (torch.Tensor): Ground truth binary targets, shape (N,).

    Returns:
        (float): Precision value.
    """
    TP = (outputs * targets).sum().item()
    FP = ((1 - targets) * outputs).sum().item()
    return TP / (TP + FP) if TP + FP > 0 else 0


def negative_precision_index(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate the negative precision, i.e., the proportion of true negatives 
    among all predicted negatives.

    Intuition: Out of all the instances that were classified as negative by the model, 
               how many were actually negative?

    Parameters:
        outputs (torch.Tensor): Predicted outputs, shape (N,).
        targets (torch.Tensor): Ground truth binary targets, shape (N,).

    Returns:
        (float): Negative precision value.
    """
    TN = ((1 - outputs) * (1 - targets)).sum().item()
    FN = ((1 - outputs) * targets).sum().item()
    return TN / (TN + FN) if TN + FN > 0 else 0

#-----------------------------------------------------------------------------------------------------------------------------------------------------