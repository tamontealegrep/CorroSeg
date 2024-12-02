
import torch
import torch.nn as nn
from typing import Type, Optional, Dict

from . import Network

from src.models.architectures.components.vit_feature_extractor import ViTFeatureExtractor
from src.models.architectures.components.unet_encoder import UnetEncoder
from src.models.architectures.components.unet_bottleneck import UnetBottleneck
from src.models.architectures.components.unet_decoder import UnetDecoder
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

        self.extractor = ViTFeatureExtractor(input_channels, input_height, input_width, patch_height, patch_width, embed_dim, num_layers, num_heads, extractor_block_type, **extractor_kwargs)
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
    def load_model(filename):
        """
        Loads a model from a file, reconstructing the architecture
        based on the saved configuration and loading the model weights.

        Args:
            filename (str): The path to the model file to load.

        Returns:
            Unet: The model with the loaded weights and configuration.
        """
        checkpoint = torch.load(filename)
        network_config = checkpoint["network_config"]
        network_state_dict = checkpoint["network_state_dict"]
        network_results = checkpoint["network_results"]

        model = ViT(
            input_channels=network_config["input_channels"],
            output_channels=network_config["output_channels"],
            input_height=network_config["input_height"],
            input_width=network_config["input_width"],
            patch_height=network_config["patch_height"],
            patch_width=network_config["patch_width"],
            embed_dim=network_config["embed_dim"],
            num_layers=network_config["num_layers"],
            num_heads=network_config["num_heads"],
            extractor_block_type=eval(network_config["extractor_block_type"]),
            extractor_kwargs=network_config["extractor_kwargs"],
            output_activation=network_config["output_activation"],
        )

        model.load_state_dict(network_state_dict)
        model.results = network_results

        return model

#-----------------------------------------------------------------------------------------------------------------------------------------------------