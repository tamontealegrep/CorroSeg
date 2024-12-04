
import torch
import torch.nn as nn
from typing import Type, Tuple

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class ViTFeatureExtractor(nn.Module):
    """
    A feature extractor based on the Vision Transformer architecture, designed to process
    input tensors and extract meaningful features through a series of transformer layers.

    Args:
        input_channels (int): Number of channels of the input tensors.
        input_height (int): Height of the input tensors.
        input_width (int): Width of the input tensors.
        patch_height (int): Height of each patch to be extracted from the input.
        patch_width (int): Width of each patch to be extracted from the input.
        embed_dim (int): Dimension of the output embeddings for each patch.
        num_layers (int): The number of Visual Transformer layers to stack.
        num_heads (int): The number of attention heads in each transformer layer.
        block_type (Type[nn.Module]): The type of block to use for the transformer layers.
        **kwargs: Additional arguments to pass to the block constructor.
    
    Attributes:
        blocks (nn.Module): An instance of the specified block type that processes the input
                            through the transformer layers.
    
    Forward pass:
        The input tensor "x" is processed through the transformer blocks, extracting features
        and generating an attention map. The original input tensor is then scaled by the attention
        map, and the scaled tensor is concatenated with the features extracted from the transformer.
        
    Returns:
        (tuple): A tuple containing.
            output (torch.Tensor): The concatenated tensor of scaled input and extracted features.
            attention_map (torch.Tensor): The attention map derived from the transformer blocks.
    """
    def __init__(self,
                 input_channels: int,
                 input_height: int,
                 input_width: int,
                 patch_height: int,
                 patch_width: int,
                 embed_dim: int,
                 num_layers: int,
                 num_heads: int,
                 block_type: Type[nn.Module],
                 **kwargs):
        super(ViTFeatureExtractor, self).__init__()
        self.blocks = block_type(input_channels,
                                 input_height,
                                 input_width,
                                 patch_height,
                                 patch_width,
                                 embed_dim,
                                 num_layers,
                                 num_heads,
                                 **kwargs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features_tensor, attention_map = self.blocks(x)
        output = features_tensor

        return output, attention_map
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------