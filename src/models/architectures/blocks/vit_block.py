
import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.models.architectures.layers.vit_emb import ViTEmbedding
from src.models.architectures.layers.vit_enc import ViTEncoder
from src.models.architectures.layers.vit_dec import ViTDecoder

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class ViTBlock(nn.Module):
    """
    A Vision Transformer block that consists of embedding, encoding, and decoding steps.

    This module processes input tensors through a series of layers that perform
    patch embedding, transformer encoding, and finally reconstructs an output tensor
    and an attention map.

    Args:
        input_channels (int): Number of channels of the input tensors.
        input_height (int): Height of the input tensors.
        input_width (int): Width of the input tensors.
        patch_height (int): Height of each patch.
        patch_width (int): Width of each patch.
        num_layers (int): The number of Visual Transformer layers to stack.
        num_heads (int): The number of attention heads in each transformer layer.
        dropout_prob (float, optional): Probability of dropout applied to the feed-forward layers. Default is None (no dropout).
        activation (str, optional): The activation function to use in the feed-forward layers. Default is "ReLU". Options include "ReLU", "LeakyReLU", and "GeLU".

    Attributes:
        embed_dim (int): Dimension of the output embeddings for each patch.

    Forward pass:
        The input tensor "x" is processed through the following sequence:
        1. Patch embedding to transform the input tensor into patch embeddings.
        2. Pass the patch embeddings through the transformer encoder, producing output embeddings and attention weights.
        3. Use the output embeddings to reconstruct the output tensor and generate the attention map using the decoder.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            (torch.Tensor): The reconstructed output tensor of shape (B, C, H, W).
            (torch.Tensor): The attention map of shape (B, 1, H, W).
    """
    def __init__(self,
                 input_channels: int,
                 input_height: int,
                 input_width: int,
                 patch_height: int,
                 patch_width: int,
                 num_layers: int,
                 num_heads: int,
                 dropout_prob: Optional[float] = None,
                 activation: Optional[str] = 'ReLU'):
        super(ViTBlock, self).__init__()
        self.embed_dim = input_channels * patch_height * patch_width

        self.embeder = ViTEmbedding(input_channels, input_height, input_width, patch_height, patch_width, self.embed_dim)

        self.encoder = ViTEncoder(num_layers, num_heads, self.embed_dim, self.embed_dim * 4, dropout_prob, activation)

        self.decoder = ViTDecoder(input_channels, input_height, input_width, patch_height, patch_width)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # (batch_size, channels, height, width) --> (batch_size, num_patches + 1, embed_dim)
        embeddings = self.embeder(x)
        # (batch_size, num_patches + 1, embed_dim) --> (batch_size, num_patches + 1, embed_dim), (batch_size, num_patches + 1, num_patches + 1)
        vit_output, attention_weights = self.encoder(embeddings)
        # (batch_size, num_patches + 1, embed_dim), (batch_size, num_patches + 1, num_patches + 1) -->
        # (batch_size, channels, height, width), (batch_size, 1, height, width) 
        output_tensor, attention_map = self.decoder(vit_output, attention_weights)

        return output_tensor, attention_map
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------