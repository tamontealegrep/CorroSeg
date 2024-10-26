
import torch
import torch.nn as nn
from typing import Type, Optional, List, Tuple

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class UnetEncoder(nn.Module):
    """
    An encoder for U-Net architecture that applies specified convolutional blocks followed by max pooling.

    Args:
        in_channels (int): Number of input channels.
        base_channels (int): Base number of output channels, will be multiplied by powers of 2 for each layer.
        num_layers (int): Number of layers in the encoder.
        block_type (Type[nn.Module]): The type of block to use.
        **kwargs: Additional arguments to pass to the block constructor.
    
    Attributes:
        blocks (nn.ModuleList): List of encoder blocks.    
        pool (nn.MaxPool2d): Max pooling layer for downsampling.

    Forward pass:
        The input tensor "x" is passed through each encoder block in sequence, applying the specified 
        convolutional operations. After each block, the output is pooled using max pooling to reduce
        the spatial dimensions. The output tensor and a list of skip connections from each encoder block 
        are returned.

    Returns:
        tuple[torch.Tensor, List[torch.Tensor]: A tuple containing:
            x (torch.Tensor): The output tensor after the final encoder block and pooling.
            List[torch.Tensor]: List of skip connections from each encoder block.
    """

    def __init__(self,
                 in_channels: int,
                 base_channels: int,
                 num_layers: int, 
                 block_type: Type[nn.Module],
                 **kwargs):
        super(UnetEncoder, self).__init__()
        self.blocks = nn.ModuleList()
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_ch
            out_ch = base_channels * (2 ** i)
            self.blocks.append(block_type(in_ch, out_ch, **kwargs))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the encoder blocks.
        
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, List[torch.Tensor]: A tuple containing:
                x (torch.Tensor): The output tensor after the final encoder block and pooling.
                List[torch.Tensor]: List of skip connections from each encoder block.
        """
        skip_connections = []
        for block in self.blocks:
            x = block(x)
            skip_connections.append(x)  
            x = self.downsample(x)    
        return x, skip_connections
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------