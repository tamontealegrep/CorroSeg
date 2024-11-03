
import torch
import torch.nn as nn
from typing import Type, Optional, List, Tuple
from src.models.architectures.layers.attention_gate import AttentionGate

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class UnetDecoder(nn.Module):
    """
    A decoder for the U-Net architecture that optionally incorporates attention mechanisms in its layers.

    This decoder applies specified convolutional blocks after upsampling, optionally enhancing feature representation 
    with attention gates that focus on relevant features from the skip connections.

    Args:
        base_channels (int): Base number of output channels, will be divided by powers of 2 for each layer.
        num_layers (int): Number of layers in the decoder.
        block_type (Type[nn.Module]): The type of block to use.
        attention_gates (bool, optional): If set to True, incorporates attention gates in the skip connections.
        cross_level_skip (float: optional): If set to True, allows the decoder to utilize skip connections from multiple levels of the encoder. Default False.
        **kwargs: Additional arguments to pass to the block constructor.

    Attributes:
        upsample (nn.ModuleList): List of ConvTranspose2d layers used in the decoder.
        blocks (nn.ModuleList): List of convolutional blocks used in the decoder.    
        attention_gates (nn.ModuleList): A list of attention gates to modulate the skip connections.

    Forward pass:
        The input tensor "x" is processed through each decoder block sequentially. For each layer, 
        the following steps occur:
        1. The input tensor is upsampled using a ConvTranspose2d layer.
        2. Optionally the corresponding skip connection from the encoder is processed through an attention gate.
        3. The upsampled tensor and the attention-modulated skip connection are concatenated.
        4. The concatenated tensor is passed through the specified convolutional block.

    Returns:
        torch.Tensor: The output tensor after the final decoder block.
    """

    def __init__(self, 
                 base_channels: int,
                 num_layers: int, 
                 block_type: Type[nn.Module],
                 attention_gates: Optional[float] = False,
                 cross_level_skip: Optional[float] = False,
                 **kwargs):
        super(UnetDecoder, self).__init__()
        self.upsample = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.attention_gates = nn.ModuleList() if attention_gates else None

        for i in range(num_layers):
            in_ch = base_channels * (2 ** num_layers) if i == 0 else out_ch
            out_ch = base_channels * (2 ** (num_layers - (i + 1)))

            upconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

            num_skip = i + 1 if cross_level_skip else 1

            if self.attention_gates is not None:
                att_gate = AttentionGate(out_ch * num_skip, out_ch, out_ch)
                self.attention_gates.append(att_gate)
            conv_block = block_type(out_ch * (num_skip + 1), out_ch, **kwargs)
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
            
            skip_connection = skip_connections.pop()
            if self.attention_gates is not None:
                att_gate = self.attention_gates[i]
                skip_connection = att_gate(skip_connection, x)
            
            x = torch.cat((x, skip_connection), dim=1)
            x = conv_block(x)
        return x
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------