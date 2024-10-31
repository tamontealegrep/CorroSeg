
import torch
import torch.nn as nn
from typing import Type, Optional, List, Tuple
from src.models.architectures.layers.attention_gate import AttentionGate

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class UnetDecoderAttention(nn.Module):
    """
    A decoder for the U-Net architecture that incorporates attention mechanisms in its layers. 
    This decoder applies specified convolutional blocks after upsampling, enhancing feature representation 
    with attention gates that focus on relevant features from the skip connections.

    Args:
        base_channels (int): The base number of output channels. This value is used to determine the number of output channels for each layer by dividing it by powers of 2.
        num_layers (int): The number of layers in the decoder, determining the depth of the network.
        block_type (Type[nn.Module]): The type of convolutional block to use in the decoder.
        **kwargs: Additional keyword arguments to pass to the block constructor (e.g., activation functions, normalization).

    Attributes:
        upsample (nn.ModuleList): A list of ConvTranspose2d layers used for upsampling in the decoder.
        attention_gates (nn.ModuleList): A list of attention gates to modulate the skip connections.
        blocks (nn.ModuleList): A list of convolutional blocks that process the concatenated output.

    Forward Pass:
        The input tensor "x" is processed through each decoder block sequentially. For each layer, 
        the following steps occur:
        1. The input tensor is upsampled using a ConvTranspose2d layer.
        2. The corresponding skip connection from the encoder is processed through an attention gate.
        3. The upsampled tensor and the attention-modulated skip connection are concatenated.
        4. The concatenated tensor is passed through the specified convolutional block.

    Returns:
        torch.Tensor: The output tensor after processing through all decoder blocks.
    """

    def __init__(self, 
                 base_channels: int,
                 num_layers: int, 
                 block_type: Type[nn.Module],
                 **kwargs):
        super(UnetDecoderAttention, self).__init__()
        self.upsample = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.blocks = nn.ModuleList()
        

        for i in range(num_layers):
            in_ch = base_channels * (2 ** num_layers) if i == 0 else out_ch
            out_ch = base_channels * (2 ** (num_layers - (i + 1)))

            upconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            att_gate = AttentionGate(out_ch, out_ch, out_ch)
            conv_block = block_type(out_ch * 2, out_ch, **kwargs)

            self.upsample.append(upconv)
            self.attention_gates.append(att_gate)
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
            att_gate = self.attention_gates[i]
            conv_block = self.blocks[i]
            
            x = upconv(x)
            skip_connection = att_gate(skip_connections.pop(), x)
            x = torch.cat((x, skip_connection), dim=1) 
            x = conv_block(x)
        return x
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------