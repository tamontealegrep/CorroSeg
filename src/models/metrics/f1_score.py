import torch
import torch.nn as nn
from .precision import precision_index, negative_precision_index
from .recall import recall_index, negative_recall_index

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def f1_score_index(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate the F1-score of the model, i.e., the harmonic mean of precision and recall.

    Intuition: A balance between precision and recall, i.e., how well the model 
               performs in terms of both identifying positives correctly and minimizing false positives.

    Parameters:
        outputs (torch.Tensor): Predicted outputs, shape (N,).
        targets (torch.Tensor): Ground truth binary targets, shape (N,).

    Returns:
        (float): F1-score value.
    """
    precision_value = precision_index(outputs, targets)
    recall_value = recall_index(outputs, targets)
    return 2 * (precision_value * recall_value) / (precision_value + recall_value) if precision_value + recall_value > 0 else 0


def negative_f1_score_index(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate the negative F1-score, i.e., the harmonic mean of negative precision and negative recall.

    Intuition: A balance between negative precision and negative recall, i.e., how well the model 
               performs in terms of both identifying negatives correctly and minimizing false negatives.

    Parameters:
        outputs (torch.Tensor): Predicted outputs, shape (N,).
        targets (torch.Tensor): Ground truth binary targets, shape (N,).

    Returns:
        (float): Negative F1-score value.
    """
    negative_precision_value = negative_precision_index(outputs, targets)
    negative_recall_value = negative_recall_index(outputs, targets)
    return 2 * (negative_precision_value * negative_recall_value) / (negative_precision_value + negative_recall_value) if negative_precision_value + negative_recall_value > 0 else 0


#-----------------------------------------------------------------------------------------------------------------------------------------------------