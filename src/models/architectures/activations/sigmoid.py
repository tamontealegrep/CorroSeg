
import torch

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Applies the Sigmoid activation function to the input tensor and scales the output to be between 0 and 1.

    Parameters:
        x (torch.Tensor): The input tensor to apply the Sigmoid activation.

    Returns:
        torch.Tensor: The transformed tensor in the range [0, 1].
    """
    x = torch.sigmoid(x)
    return x

#-----------------------------------------------------------------------------------------------------------------------------------------------------
