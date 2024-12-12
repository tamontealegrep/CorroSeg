
import torch
import torch.nn as nn

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def accuracy_index(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate the accuracy of the model, i.e., the proportion of correct predictions.
    
    Intuition: Out of all the predictions made by the model, how many did the model get right?
    
    Parameters:
        outputs (torch.Tensor): Predicted outputs, shape (N,).
        targets (torch.Tensor): Ground truth binary targets, shape (N,).

    Returns:
        (float): Accuracy value.
    """
    correct = (outputs == targets).sum().item()
    total = targets.numel()
    return correct / total

#-----------------------------------------------------------------------------------------------------------------------------------------------------