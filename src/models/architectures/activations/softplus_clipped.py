
import torch

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def softplus_clipped(x:torch.Tensor) -> torch.Tensor:
    """
    Applies the Softplus activation function to the input tensor and then clips the output to the range [0, 1].

    The Softplus function is a smooth approximation of the ReLU activation, producing values in the range (0, ∞).
    Clipping is applied to restrict the output to the desired range [0, 1].

    Parameters:
        x (torch.Tensor): The input tensor to apply the Softplus activation and clipping.

    Returns:
        torch.Tensor: The tensor after applying Softplus and clipping it to the range [0, 1].
    """
    x = torch.nn.functional.softplus(x)  # Apply Softplus
    return torch.clamp(x, min=0, max=1)  # Clip the output to [0, 1]

#-----------------------------------------------------------------------------------------------------------------------------------------------------
