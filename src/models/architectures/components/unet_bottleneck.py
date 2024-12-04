
import torch
import torch.nn as nn
from typing import Type, Optional

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class UnetBottleneck(nn.Module):
    """
    A bottleneck for U-Net architecture that applies a specified convolutional block.

    Parameters:
        base_channels (int): Base number of output channels. 
                             The number of channels for the bottleneck is computed as:
                             - Input channels: base_channels * (2 ** (num_layers - 1))
                             - Output channels: base_channels * (2 ** num_layers)
        num_layers (int): Number of layers in the bottleneck. This determines the scaling of output channels.
        block_type (Type[nn.Module]): The type of block to use for the bottleneck (e.g., DoubleConvBlock).
        **kwargs: Additional arguments to pass to the block constructor.

    Attributes:
        blocks (nn.Module): The convolutional block for feature extraction in the bottleneck.

    Forward pass:
        The input tensor "x" is passed through the convolutional block.

    Returns:
        (torch.Tensor): The output tensor after applying the convolutional block.
    """

    def __init__(self,
                 base_channels: int,
                 num_layers: int, 
                 block_type: Type[nn.Module],
                 **kwargs):
        super(UnetBottleneck, self).__init__()

        in_ch = base_channels * (2 ** (num_layers - 1))
        out_ch = base_channels * (2 ** num_layers)
        self.blocks = block_type(in_ch, out_ch, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the bottleneck.

        Args:
            x (torch.Tensor): Input tensor from the encoder.

        Returns:
            torch.Tensor: The output tensor after applying the convolutional block.
        """
        return self.blocks(x)
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------    