
import torch

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def tanh_normalized(x:torch.Tensor) -> torch.Tensor:
    """
    Applies the tanh function to the input x and scales the output to be between 0 and 1.

    Parameters:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The transformed and scaled tensor in the range [0, 1].
    """
    x = torch.tanh(x)
    x = (x + 1) / 2
    return x

#-----------------------------------------------------------------------------------------------------------------------------------------------------
