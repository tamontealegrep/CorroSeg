
import torch

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def relu_clipped(x:torch.Tensor) -> torch.Tensor:
    """
    Applies the ReLU activation function to the input tensor and then clips the output to the range [0, 1].

    Parameters:
        x (torch.Tensor): The input tensor to apply the ReLU activation and clipping.

    Returns:
        torch.Tensor: The tensor after applying ReLU and clipping it to the range [0, 1].
    """
    x = torch.relu(x)  
    return torch.clamp(x, min=0, max=1)

#-----------------------------------------------------------------------------------------------------------------------------------------------------
