
import torch
import torch.nn as nn
from typing import Type, List, Tuple, Optional, Dict
from src.models import DEVICE 
from src.models.architectures.components.unet_encoder import UnetEncoder
from src.models.architectures.components.unet_bottleneck import UnetBottleneck
from src.models.architectures.components.unet_decoder import UnetDecoder
from src.models.architectures.components.unet_skip_connections import UnetSkipConnections

from src.models.train.train import train_model as tm
from src.models.train.train import train_validation_model as tvm

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class Unet(nn.Module):
    """
    U-Net architecture combining encoder, bottleneck, and decoder components.

    Args:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.
        base_channels (int): Base number of output channels, will be multiplied by powers of 2 for each layer.
        num_layers (int): Number of layers in the encoder and decoder.
        encoder_block_type (Type[nn.Module]): The type of block to use for the encoder.
        bottleneck_block_type (Type[nn.Module]): The type of block to use for the bottleneck.
        decoder_block_type (Type[nn.Module]): The type of block to use for the decoder.
        encoder_kwargs (Dict, optional): Additional arguments for the encoder blocks.
        bottleneck_kwargs (Dict, optional): Additional arguments for the bottleneck block.
        decoder_kwargs (Dict, optional): Additional arguments for the decoder blocks.
        device (str, optional): The device to perform calculations on. Default is DEVICE. Options: "cpu" and "cuda".

    Attributes:
        encoder (UnetEncoder): The encoder component of the U-Net.
        bottleneck (UnetBottleneck): The bottleneck component of the U-Net.
        decoder (UnetDecoder): The decoder component of the U-Net.
        final_conv (nn.Conv2d): Final convolution layer to produce output with the desired number of channels.

    Forward pass:
        The input tensor is passed through the encoder, bottleneck, and decoder sequentially.

    Returns:
        torch.Tensor: The output tensor after the final convolution layer.
    """

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 base_channels: int, 
                 num_layers: int,
                 encoder_block_type: Type[nn.Module], 
                 bottleneck_block_type: Type[nn.Module], 
                 decoder_block_type: Type[nn.Module],
                 skip_connections_block_type: Type[nn.Module] = None, 
                 encoder_kwargs: Optional[Dict] = None, 
                 bottleneck_kwargs: Optional[Dict] = None, 
                 decoder_kwargs: Optional[Dict] = None,
                 skip_connections_kwargs: Optional[Dict] = None,
                 attention_gates: Optional[float] = False,
                 device=DEVICE):
        super(Unet, self).__init__()
        self.device = device

        # Initialize kwargs if not provided
        encoder_kwargs = encoder_kwargs or {}
        bottleneck_kwargs = bottleneck_kwargs or {}
        decoder_kwargs = decoder_kwargs or {}
        skip_connections_kwargs = skip_connections_kwargs or {}
        
        self.encoder = UnetEncoder(input_channels, base_channels, num_layers, encoder_block_type, **encoder_kwargs)
        
        self.bottleneck = UnetBottleneck(base_channels, num_layers, bottleneck_block_type, **bottleneck_kwargs)
        
        if skip_connections_block_type is not None:
            self.skip_connections = UnetSkipConnections(base_channels, num_layers, skip_connections_block_type, attention_gates=attention_gates, **skip_connections_kwargs)
            self.decoder = UnetDecoder(base_channels, num_layers, decoder_block_type, attention_gates=attention_gates, cross_level_skip=True, **decoder_kwargs)
        else:
            self.skip_connections = None
            self.decoder = UnetDecoder(base_channels, num_layers, decoder_block_type, attention_gates=attention_gates, **decoder_kwargs)

            
        self.final_conv = nn.Conv2d(base_channels, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net architecture.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output tensor after processing through the U-Net.
        """
        # Encoder
        x, skip_connections = self.encoder(x)
        # Bottleneck
        x = self.bottleneck(x)
        # Skip Connections
        if self.skip_connections is not None:
            skip_connections = self.skip_connections(skip_connections)
        # Decoder
        x = self.decoder(x, skip_connections)
        # Final convolution
        x = self.final_conv(x)
        
        return x

    def train_model(self,train_loader, criterion, optimizer, num_epochs=10):
        tm(self,train_loader, criterion, optimizer, num_epochs)

    def train_validate_model(self,train_loader, val_loader, criterion, optimizer, num_epochs=10):
        tvm(self,train_loader, val_loader, criterion, optimizer, num_epochs)

#-----------------------------------------------------------------------------------------------------------------------------------------------------