import torch
import torch.nn as nn

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def mcc_index(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate the Matthews Correlation Coefficient (MCC), a balanced measure that considers all 
    four confusion matrix categories (True Positives, False Positives, True Negatives, False Negatives).

    Intuition: A balanced evaluation metric that accounts for true and false positives as well as 
               true and false negatives, providing a more holistic measure of model performance.

    Parameters:
        outputs (torch.Tensor): Predicted outputs, shape (N,).
        targets (torch.Tensor): Ground truth binary targets, shape (N,).

    Returns:
        (float): Matthews Correlation Coefficient value.
    """
    TP = (outputs * targets).sum().item()  
    TN = ((1 - outputs) * (1 - targets)).sum().item()
    FP = (outputs * (1 - targets)).sum().item()
    FN = ((1 - outputs) * targets).sum().item()

    numerator = TP * TN - FP * FN
    denominator = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5

    return numerator / denominator if denominator > 0 else 0

#-----------------------------------------------------------------------------------------------------------------------------------------------------