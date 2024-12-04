
import torch
import torch.nn as nn
from typing import Type, Optional, Dict

from . import Network

from src.models.architectures.components.unet_encoder import UnetEncoder
from src.models.architectures.components.unet_bottleneck import UnetBottleneck
from src.models.architectures.components.unet_decoder import UnetDecoder
from src.models.architectures.components.unet_skip_connections import UnetSkipConnections
from src.models.architectures.activations import get_activation_function

from src.models.architectures.blocks import *

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class Unet(Network):
    """
    U-Net architecture combining encoder, bottleneck, and decoder components.

    Parameters:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.
        base_channels (int): Base number of output channels, will be multiplied by powers of 2 for each layer.
        num_layers (int): Number of layers in the encoder and decoder.
        encoder_block_type (Type[nn.Module]): The type of block to use for the encoder.
        bottleneck_block_type (Type[nn.Module]): The type of block to use for the bottleneck.
        decoder_block_type (Type[nn.Module]): The type of block to use for the decoder.
        skip_connections_block_type (Type[nn.Module]): The type of block to use for the skip connections.
        encoder_kwargs (Dict, optional): Additional arguments for the encoder blocks.
        bottleneck_kwargs (Dict, optional): Additional arguments for the bottleneck block.
        decoder_kwargs (Dict, optional): Additional arguments for the decoder blocks.
        skip_connections_kwargs Dict, optional): Additional arguments for the skip connection blocks.
        attention_gates (float, optional): Whether or not use attention gates in the skep connection blocks. Default False.
        output_activation (str, optional): The activation function to apply to the output of the network. 
            This function is applied after the final convolution layer. Default is "".

    Attributes:
        encoder (UnetEncoder): The encoder component of the U-Net.
        bottleneck (UnetBottleneck): The bottleneck component of the U-Net.
        decoder (UnetDecoder): The decoder component of the U-Net.
        final_conv (nn.Conv2d): Final convolution layer to produce output with the desired number of channels.
        output_activation (Callable, optional): The activation function applied to the output of the network after the final convolution layer.
    
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
                 output_activation: Optional[str] = ""):
        super(Unet, self).__init__()

        # Save config
        self.config = {
            "network_class": self.__class__.__name__, 
            "input_channels": input_channels,
            "output_channels": output_channels,
            "base_channels": base_channels,
            "num_layers": num_layers,
            "encoder_block_type": encoder_block_type.__name__,
            "bottleneck_block_type": bottleneck_block_type.__name__,
            "decoder_block_type": decoder_block_type.__name__,
            "skip_connections_block_type": skip_connections_block_type.__name__ if skip_connections_block_type else None,
            "encoder_kwargs": encoder_kwargs or {},
            "bottleneck_kwargs": bottleneck_kwargs or {},
            "decoder_kwargs": decoder_kwargs or {},
            "skip_connections_kwargs": skip_connections_kwargs or {},
            "attention_gates": attention_gates,
            "output_activation": output_activation,
        }

        # Initialize kwargs if not provided
        encoder_kwargs = encoder_kwargs or {}
        bottleneck_kwargs = bottleneck_kwargs or {}
        decoder_kwargs = decoder_kwargs or {}
        skip_connections_kwargs = skip_connections_kwargs or {}
        
        self.encoder = UnetEncoder(
            input_channels,
            base_channels,
            num_layers,
            encoder_block_type,
            **encoder_kwargs)
        
        self.bottleneck = UnetBottleneck(
            base_channels,
            num_layers,
            bottleneck_block_type,
            **bottleneck_kwargs)
        
        if skip_connections_block_type is not None:
            self.skip_connections = UnetSkipConnections(
                base_channels,
                num_layers,
                skip_connections_block_type,
                attention_gates=attention_gates,
                **skip_connections_kwargs)
            
            self.decoder = UnetDecoder(
                base_channels,
                num_layers,
                decoder_block_type,
                attention_gates=attention_gates,
                cross_level_skip=True,
                **decoder_kwargs)
        else:
            self.skip_connections = None
            self.decoder = UnetDecoder(
                base_channels,
                num_layers,
                decoder_block_type,
                attention_gates=attention_gates,
                **decoder_kwargs)

        self.final_conv = nn.Conv2d(base_channels, output_channels, kernel_size=1)

        self.output_activation = get_activation_function(output_activation) if output_activation != "" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net architecture.

        Parameters:
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
        # Output activation
        if self.output_activation is not None:
            x = self.output_activation(x)

        return x
    
    @staticmethod
    def from_dict(config_dict):
        """
        Creates a Unet model instance from the provided configuration dictionary.

        Parameters:
            config_dict (dict): A dictionary containing the model's configuration.

        Returns:
            Unet: An instance of the Unet model constructed from the dictionary.
        """
        return Unet(
            input_channels=config_dict["input_channels"],
            output_channels=config_dict["output_channels"],
            base_channels=config_dict["base_channels"],
            num_layers=config_dict["num_layers"],
            encoder_block_type=eval(config_dict["encoder_block_type"]),
            bottleneck_block_type=eval(config_dict["bottleneck_block_type"]),
            decoder_block_type=eval(config_dict["decoder_block_type"]),
            skip_connections_block_type=(
                eval(config_dict["skip_connections_block_type"]) 
                if config_dict["skip_connections_block_type"] 
                else None
                ),
            encoder_kwargs=config_dict["encoder_kwargs"],
            bottleneck_kwargs=config_dict["bottleneck_kwargs"],
            decoder_kwargs=config_dict["decoder_kwargs"],
            skip_connections_kwargs=config_dict["skip_connections_kwargs"],
            attention_gates=config_dict["attention_gates"],
            output_activation=config_dict["output_activation"],
        )

#-----------------------------------------------------------------------------------------------------------------------------------------------------