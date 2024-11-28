
import torch
from torch import nn
import numpy as np
from typing import Tuple, List

#---------------------------------------------------------------------------------------------------

def _evaluation_step(model:nn.Module, eval_loader:torch.utils.data.DataLoader, criterion:nn.Module) -> float:
    """
    Perform a single evaluation step for the given model.

    This function sets the model to evaluation mode, iterates over the evaluation
    data provided by the DataLoader, computes the loss without updating the model's
    weights, and returns the average loss for the evaluation set.

    Parameters:
        model (torch.nn.Module): The model to be evaluated.
        eval_loader (torch.utils.data.DataLoader): DataLoader for the evaluation data.
        criterion (torch.nn.Module)): The loss function used to compute the evaluation loss.

    Returns:
        float: The average loss for the evaluation set, calculated as the total loss
        divided by the number of samples in the evaluation dataset.
    """
    model.eval()  # Set the model to evaluation mode

    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in eval_loader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

    epoch_loss = running_loss / len(eval_loader.dataset)

    return epoch_loss

def evaluate_model(model: nn.Module, eval_loader: torch.utils.data.DataLoader, criterion: nn.Module) -> Tuple[np.ndarray, List[float], float]:
    """
    Evaluate the model on the evaluation dataset.

    This function performs the evaluation by running the model in evaluation mode, 
    making predictions, computing the loss for each sample, and returning:
    - the predictions for each sample,
    - the individual losses for each sample, 
    - and the total average loss.

    Parameters:
        model (nn.Module): The model to be evaluated.
        eval_loader (DataLoader): DataLoader for the evaluation data.
        criterion (nn.Module): The loss function used to compute the loss.

    Returns:
        tuple: A tuple containing.
            outputs (numpy.ndarray): predictions for each sample.
            losses (list): List of individual losses for each sample.
            average_loss (float): Average loss for the evaluation dataset.
    """
    model.to(model.device)
    model.eval()  # Set the model to evaluation mode

    outputs = []
    losses = []
    running_loss = 0.0

    with torch.no_grad():
        for input, target in eval_loader:
            input, target = input.to(model.device), target.to(model.device)

            output = model(input)
            loss = criterion(output, target)

            outputs.extend(output.squeeze().cpu().numpy())
            losses.extend([loss.cpu().numpy() for _ in range(input.size(0))])

            running_loss += loss.item()

    average_loss = running_loss / len(eval_loader.dataset)

    return np.concatenate(outputs, axis=0), losses, average_loss

#---------------------------------------------------------------------------------------------------
