
import torch
import torch.nn as nn
from typing import Type, Optional, List, Tuple
from src.models.architectures.layers.attention_gate import AttentionGate

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class UnetSkipConnections(nn.Module):
    """
    Sikp connections for U-Net architecture that applies specified convolutional blocks after upsampling.

    This class facilitates cross-level skip connections by allowing the combination of feature maps
    from different encoder layers during the decoding process.

    Args:
        base_channels (int): Base number of output channels; used to calculate the number of channels for each layer based on powers of 2.
        num_layers (int): Total number of layers in the U-Net.
        block_type (Type[nn.Module]): The type of block to use for convolutions in the decoder.
        attention_gates (bool, optional): If set to True, incorporates attention gates in the skip connections.
        **kwargs: Additional arguments to pass to the block constructor.

    Attributes:
        upsample (List[nn.ModuleList]): Lists of `ConvTranspose2d` layers for upsampling.
        blocks (List[nn.ModuleList]): Lists of convolutional blocks used in the skip connections.
        attention_gates List[nn.ModuleList]): Lists of attention gates to modulate the skip connections.
        base_channels (int): Base number of output channels.
        
    Forward Pass:
        The input list of tensors (skip connections) is processed through each block in sequence. Each block performs upsampling on the input tensor,
        concatenates it with the corresponding skip connection, applies the specified convolutional operations, and updates the skip connection with the resulting output by concatenation.
        Finally, the output tensor from the last block is returned.

    Returns:
        List[torch.Tensor]: A list of output tensors after applying all blocks, including the concatenated skip connections.
    """

    def __init__(self, 
                 base_channels: int,
                 num_layers: int, 
                 block_type: Type[nn.Module],
                 attention_gates: Optional[float] = False,
                 **kwargs):
        super(UnetSkipConnections, self).__init__()
        self.base_channels = base_channels
        self.upsample = nn.ModuleList()
        self.attention_gates = nn.ModuleList() if attention_gates else None
        self.blocks = nn.ModuleList()
        
        for i in range(num_layers - 1):
            up_module_list = nn.ModuleList()
            attention_module_list = nn.ModuleList() if self.attention_gates is not None else None
            conv_module_list = nn.ModuleList()
            for j in range(num_layers - 1 - i):
                in_ch = base_channels * (2 ** j) * (i + 2)
                up_in_ch = base_channels * (2 ** (j + 1))
                out_ch = base_channels * (2 ** j)

                upconv = nn.ConvTranspose2d(up_in_ch, out_ch, kernel_size=2, stride=2)
                
                if self.attention_gates is not None:
                    att_gate = AttentionGate(in_ch - out_ch, out_ch, out_ch)
                    attention_module_list.append(att_gate)

                conv_block = block_type(in_ch, out_ch, **kwargs)

                up_module_list.append(upconv)
                conv_module_list.append(conv_block)

            self.upsample.append(up_module_list)
            self.blocks.append(conv_module_list)
            if self.attention_gates is not None:
                self.attention_gates.append(attention_module_list)

    def forward(self, skip_connections: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through the blocks.

        This method processes the skip connections from the encoder and the upsampled features,
        applying the corresponding convolutional blocks to produce the output tensors.

        Args:
            skip_connections (List[torch.Tensor]): List of skip connection tensors from the encoder, each corresponding to a specific layer.

        Returns:
            List[torch.Tensor]: A list of output tensors after applying all blocks, where each tensor corresponds to a layer in the decoder.
        """
        for i in range(len(self.blocks)):
            for j in range(len(self.blocks[i])):
                upsampled_tensor = self.upsample[i][j](skip_connections[j + 1][:, -self.base_channels * (2 ** (j + 1)):, :, :])
                
                if self.attention_gates is not None:
                    attention_output = self.attention_gates[i][j](skip_connections[j], upsampled_tensor)
                else:
                    attention_output = skip_connections[j]
                
                concatenated_tensor = torch.cat((attention_output, upsampled_tensor), dim=1)
                
                skip_connection_output = self.blocks[i][j](concatenated_tensor)
                
                skip_connections[j] = torch.cat((skip_connections[j], skip_connection_output), dim=1)

        return skip_connections
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------