
import torch
import torch.nn as nn
from typing import Self, Type, Optional, Dict

from . import Network

from src.models.architectures.components.vit_feature_extractor import ViTFeatureExtractor
from src.models.architectures.activations import get_activation_function

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
        output_activation (str, optional): The activation function to apply to the output of the network. This function is applied after the final convolution layer. Default is "".

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
                 output_channels: int,
                 input_height: int,
                 input_width: int,
                 patch_height: int,
                 patch_width: int,
                 embed_dim: int,
                 num_layers: int,
                 num_heads: int,
                 extractor_block_type: Type[nn.Module], 
                 extractor_kwargs: Optional[Dict] = None,
                 output_activation: Optional[str] = ""):
        super(ViT, self).__init__()
        # Save config
        self.config = {
            "network_class": self.__class__.__name__, 
            "input_channels": input_channels,
            "output_channels": output_channels,
            "input_height": input_height,
            "input_width": input_width,
            "patch_height": patch_height,
            "patch_width": patch_width,
            "embed_dim": embed_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "extractor_block_type": extractor_block_type.__name__,
            "extractor_kwargs": extractor_kwargs or {},
            "output_activation": output_activation,
        }
        # Initialize kwargs if not provided
        extractor_kwargs = extractor_kwargs or {}

        self.extractor = ViTFeatureExtractor(
            input_channels,
            input_height,
            input_width,
            patch_height,
            patch_width,
            embed_dim,
            num_layers,
            num_heads,
            extractor_block_type,
            **extractor_kwargs)
        
        self.final_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.output_activation = get_activation_function(output_activation) if output_activation != "" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ViT architecture.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output tensor after processing through the U-Net.
        """
        # Extractor
        x, _ = self.extractor(x)
        # Final convolution
        x = self.final_conv(x)
        # Output activation
        if self.output_activation is not None:
            x = self.output_activation(x)

        return x
    
    @staticmethod
    def from_dict(config_dict) -> Self:
        """
        Creates a ViT model instance from the provided configuration dictionary.

        Args:
            config_dict (dict): A dictionary containing the model's configuration.

        Returns:
            ViT: An instance of the ViT model constructed from the dictionary.
        """
        return ViT(
            input_channels=config_dict["input_channels"],
            output_channels=config_dict["output_channels"],
            input_height=config_dict["input_height"],
            input_width=config_dict["input_width"],
            patch_height=config_dict["patch_height"],
            patch_width=config_dict["patch_width"],
            embed_dim=config_dict["embed_dim"],
            num_layers=config_dict["num_layers"],
            num_heads=config_dict["num_heads"],
            extractor_block_type=eval(config_dict["extractor_block_type"]),
            extractor_kwargs=config_dict["extractor_kwargs"],
            output_activation=config_dict["output_activation"],
        )

#-----------------------------------------------------------------------------------------------------------------------------------------------------