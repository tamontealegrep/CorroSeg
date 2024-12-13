import torch
import torch.nn as nn
from typing import Optional

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def mcc_index(outputs: torch.Tensor,
              targets: torch.Tensor,
              smooth: Optional[float] = 1e-6,
              ) -> float:
    """
    Calculate the Matthews Correlation Coefficient (MCC), a balanced measure that considers all 
    four confusion matrix categories (True Positives, False Positives, True Negatives, False Negatives).

    Intuition: A balanced evaluation metric that accounts for true and false positives as well as 
               true and false negatives, providing a more holistic measure of model performance, range [-1, 1].

    Parameters:
        outputs (torch.Tensor): Predicted outputs after applying sigmoid, shape (N,).
        targets (torch.Tensor): Ground truth binary targets, shape (N,).
        smooth (float, optional): Small constant to avoid division by zero.

    Returns:
        (float): Matthews Correlation Coefficient value.
    """
    TP = (outputs * targets).sum()
    TN = ((1 - outputs) * (1 - targets)).sum()
    FP = (outputs * (1 - targets)).sum()
    FN = ((1 - outputs) * targets).sum()

    numerator = TP * TN - FP * FN
    denominator = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5

    return numerator / (denominator + smooth) 

class MCCLoss(nn.Module):
    """
    Matthews Correlation Coefficient (MCC) Loss function, a balanced measure that considers all four 
    confusion matrix categories (True Positives, False Positives, True Negatives, False Negatives).

    MCC is a holistic metric that accounts for true positives, false positives, true negatives, and false negatives.

    Parameters:
        smooth (float, optional): Small constant to prevent division by zero. Defaults to 1e-6.
    """

    def __init__(self, smooth: Optional[float] = 1e-6):
        super(MCCLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs:torch.Tensor, targets:torch.Tensor) -> float:
        # Flatten the tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # Calculate the MCC index using the external function
        mcc_idx = mcc_index(outputs, targets, self.smooth)

        # MCC Loss is 1 - MCC index
        return 1 - mcc_idx
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------