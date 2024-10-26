
import torch
import torch.nn as nn
from typing import Type, Optional, List, Tuple

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class UnetDecoder(nn.Module):
    """
    A decoder for U-Net architecture that applies specified convolutional blocks after upsampling.

    Args:
        base_channels (int): Base number of output channels, will be divided by powers of 2 for each layer.
        num_layers (int): Number of layers in the decoder.
        block_type (Type[nn.Module]): The type of block to use.
        **kwargs: Additional arguments to pass to the block constructor.

    Attributes:
        upsample (nn.ModuleList): List of ConvTranspose2d layers used in the decoder.
        blocks (nn.ModuleList): List of convolutional blocks used in the decoder.    
    
    Forward pass:
        The input tensor "x" is passed through each decoder block in sequence. Each block performs
        upsampling on the input tensor, concatenates it with the corresponding skip connection from the
        encoder, and then applies the specified convolutional operations. The output tensor from the 
        final block is returned.

    Returns:
        torch.Tensor: The output tensor after the final decoder block.
    """

    def __init__(self, 
                 base_channels: int,
                 num_layers: int, 
                 block_type: Type[nn.Module],
                 **kwargs):
        super(UnetDecoder, self).__init__()
        self.upsample = nn.ModuleList()
        self.blocks = nn.ModuleList()

        for i in range(num_layers):
            in_ch = base_channels * (2 ** num_layers) if i == 0 else out_ch
            out_ch = base_channels * (2 ** (num_layers - (i + 1)))

            upconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            conv_block = block_type(out_ch * 2, out_ch, **kwargs)

            self.upsample.append(upconv)
            self.blocks.append(conv_block)

    def forward(self, x: torch.Tensor, skip_connections: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the decoder blocks.

        Args:
            x (torch.Tensor): Input tensor from the previous decoder block or the bottleneck.
            skip_connections (List[torch.Tensor]): List of skip connection tensors from the encoder.

        Returns:
            torch.Tensor: The output tensor after applying all decoder blocks.
        """
        for i in range(len(self.blocks)):
            upconv = self.upsample[i]
            conv_block = self.blocks[i]
            
            x = upconv(x)
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = conv_block(x)
        return x
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------