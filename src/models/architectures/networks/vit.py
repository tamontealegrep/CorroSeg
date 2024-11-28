
import torch
import torch.nn as nn
from typing import Type, Optional, Dict, Union

from . import Network

from src.models import DEVICE 
from src.models.architectures.components.vit_feature_extractor import ViTFeatureExtractor
from src.models.architectures.components.unet_encoder import UnetEncoder
from src.models.architectures.components.unet_bottleneck import UnetBottleneck
from src.models.architectures.components.unet_decoder import UnetDecoder

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class ViT(Network):
    """
    ViT architecture.

    Args:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.
        input_height (int): Height of the input tensors.
        input_width (int): Width of the input tensors.
        patch_height (int): Height of each patch to be extracted from the input.
        patch_width (int): Width of each patch to be extracted from the input.
        embed_dim (int): Dimension of the output embeddings for each patch.
        num_layers (int): The number of Visual Transformer layers to stack.
        num_heads (int): The number of attention heads in each transformer layer. 
        extractor_block_type (Type[nn.Module]): The type of block to use for the feature extractor.
        extractor_kwargs (Dict, optional): Additional arguments for the feature extractor blocks.
        device (str, optional): The device to perform calculations on. Default is DEVICE. Options: "cpu" and "cuda".

    Attributes:
        extractor (ViTFeatureExtractor): The feature extractor based on ViT.
        final_conv (nn.Conv2d): Final convolution layer to produce output with the desired number of channels.

    Forward pass:
        The input tensor is passed through the encoder, bottleneck, and decoder sequentially.

    Returns:
        torch.Tensor: The output tensor after the ViT.
    """

    def __init__(self,
                 input_channels: int,
                 out_channels: int,
                 input_height: int,
                 input_width: int,
                 patch_height: int,
                 patch_width: int,
                 embed_dim: int,
                 num_layers: int,
                 num_heads: int,
                 extractor_block_type: Type[nn.Module], 
                 extractor_kwargs: Optional[Dict] = None, 
                 device=DEVICE):
        super(ViT, self).__init__()
        self.device = device

        # Initialize kwargs if not provided
        extractor_kwargs = extractor_kwargs or {}

        self.extractor = ViTFeatureExtractor(input_channels, input_height, input_width, patch_height, patch_width, embed_dim, num_layers, num_heads, extractor_block_type, **extractor_kwargs)
        self.final_conv = nn.Conv2d(input_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extractor
        x, _ = self.extractor(x)
        x = self.final_conv(x)
        return x

#-----------------------------------------------------------------------------------------------------------------------------------------------------