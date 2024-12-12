import torch
import torch.nn as nn

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def recall_index(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate the recall of the model, i.e., the proportion of true positive predictions 
    among all actual positives.

    Intuition: Out of all the instances that are actually positive, how many did the model 
               correctly identify as positive?

    Parameters:
        outputs (torch.Tensor): Predicted outputs, shape (N,).
        targets (torch.Tensor): Ground truth binary targets, shape (N,).

    Returns:
        (float): Recall value.
    """
    TP = (outputs * targets).sum().item()
    FN = ((1 - outputs) * targets).sum().item()
    return TP / (TP + FN) if TP + FN > 0 else 0


def negative_recall_index(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate the negative recall, i.e., the proportion of true negatives among all actual negatives.

    Intuition: Out of all the instances that are actually negative, how many did the model 
               correctly identify as negative?

    Parameters:
        outputs (torch.Tensor): Predicted outputs, shape (N,).
        targets (torch.Tensor): Ground truth binary targets, shape (N,).

    Returns:
        (float): Negative recall value.
    """
    TN = ((1 - outputs) * (1 - targets)).sum().item()
    FP = (outputs * (1 - targets)).sum().item()
    return TN / (TN + FP) if TN + FP > 0 else 0

#-----------------------------------------------------------------------------------------------------------------------------------------------------