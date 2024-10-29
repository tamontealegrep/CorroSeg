
import torch
import torch.nn as nn
from typing import Type, List, Tuple, Optional, Dict
from src.models import DEVICE 
from src.models.architectures.components.vit_feature_extractor import ViTFeatureExtractor
from src.models.architectures.components.unet_encoder import UnetEncoder
from src.models.architectures.components.unet_bottleneck import UnetBottleneck
from src.models.architectures.components.unet_decoder import UnetDecoder
from src.models.train.train import train_model as tm
from src.models.train.train import train_validation_model as tvm

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class ViTUnet(nn.Module):
    """
    U-Net architecture combining encoder, bottleneck, and decoder components.

    Args:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.
        input_height (int): Height of the input tensors.
        input_width (int): Width of the input tensors.
        patch_height (int): Height of each patch to be extracted from the input.
        patch_width (int): Width of each patch to be extracted from the input.
        num_vit_layers (int): The number of Visual Transformer layers to stack.
        num_vit_heads (int): The number of attention heads in each transformer layer. 
        base_unet_channels (int): Base number of output channels, will be multiplied by powers of 2 for each layer.
        num_unet_layers (int): Number of layers in the encoder and decoder.
        extractor_block_type (Type[nn.Module]): The type of block to use for the feature extractor.
        encoder_block_type (Type[nn.Module]): The type of block to use for the encoder.
        bottleneck_block_type (Type[nn.Module]): The type of block to use for the bottleneck.
        decoder_block_type (Type[nn.Module]): The type of block to use for the decoder.
        extractor_kwargs (Dict, optional): Additional arguments for the feature extractor blocks.
        encoder_kwargs (Dict, optional): Additional arguments for the encoder blocks.
        bottleneck_kwargs (Dict, optional): Additional arguments for the bottleneck block.
        decoder_kwargs (Dict, optional): Additional arguments for the decoder blocks.
        device (str, optional): The device to perform calculations on. Default is DEVICE. Options: "cpu" and "cuda".

    Attributes:
        extractor (ViTFeatureExtractor): The feature extractor based on ViT.
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
                 out_channels: int,
                 input_height: int,
                 input_width: int,
                 patch_height: int,
                 patch_width: int,
                 num_vit_layers: int,
                 num_vit_heads: int,
                 base_unet_channels: int, 
                 num_unet_layers: int,
                 extractor_block_type: Type[nn.Module], 
                 encoder_block_type: Type[nn.Module], 
                 bottleneck_block_type: Type[nn.Module], 
                 decoder_block_type: Type[nn.Module],
                 extractor_kwargs: Optional[Dict] = None, 
                 encoder_kwargs: Optional[Dict] = None, 
                 bottleneck_kwargs: Optional[Dict] = None, 
                 decoder_kwargs: Optional[Dict] = None,
                 device=DEVICE):
        super(ViTUnet, self).__init__()
        self.device = device

        # Initialize kwargs if not provided
        extractor_kwargs = extractor_kwargs or {}
        encoder_kwargs = encoder_kwargs or {}
        bottleneck_kwargs = bottleneck_kwargs or {}
        decoder_kwargs = decoder_kwargs or {}

        self.extractor = ViTFeatureExtractor(input_channels, input_height, input_width, patch_height, patch_width, num_vit_layers, num_vit_heads, extractor_block_type, **extractor_kwargs)
        self.encoder = UnetEncoder(input_channels * 2, base_unet_channels, num_unet_layers, encoder_block_type, **encoder_kwargs)
        self.bottleneck = UnetBottleneck(base_unet_channels, num_unet_layers, bottleneck_block_type, **bottleneck_kwargs)
        self.decoder = UnetDecoder(base_unet_channels, num_unet_layers, decoder_block_type, **decoder_kwargs)
        self.final_conv = nn.Conv2d(base_unet_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extractor
        x, attention_scaler = self.extractor(x)
        # Encoder
        x, skip_connections = self.encoder(x)      
        # Bottleneck
        x = self.bottleneck(x)
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