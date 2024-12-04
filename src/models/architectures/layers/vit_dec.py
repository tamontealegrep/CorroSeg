
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class PatchReconstructor(nn.Module):
    """
    A module to reconstruct a tensor from its patch embeddings.

    This module takes embeddings derived from patches of an input tensor
    and rearranges them to reconstruct a tensor with the same shape of the original tensor. Each embedding
    corresponds to a patch of the input tensor, and the module combines
    these patches back into the full tensor.

    Parameters:
        input_channels (int): The number of channels in the input tensor.
        input_height (int): The height of the input tensor.
        input_width (int): The width of the input tensor.
        patch_height (int): The height of each patch.
        patch_width (int): The width of each patch.

    Attributes:
        input_channels (int): The number of channels in the input tensor.
        input_height (int): The height of the input tensor.
        input_width (int): The width of the input tensor.
        patch_height (int): The height of each patch.
        patch_width (int): The width of each patch.
        num_patches_h (int): The number of patches along the height of the tensor.
        num_patches_w (int): The number of patches along the width of the tensor.

    Returns:
        (torch.Tensor): A reconstructed tensor of shape (batch_size, input_channels, input_height, input_width),
                      where each patch is populated according to its corresponding embedding.
    """
    def __init__(self, input_channels: int, input_height: int, input_width: int, patch_height: int, patch_width: int):
        super(PatchReconstructor, self).__init__()
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.num_patches_h = input_height // patch_height
        self.num_patches_w = input_width // patch_width

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, embed_dim = embeddings.shape
        expected_num_patches = self.num_patches_h * self.num_patches_w
        assert num_patches == expected_num_patches, f"Expected {expected_num_patches} patches, but got {num_patches}."
        
        # Reshape embeddings to (batch_size, num_patches_h, num_patches_w, channels, patch_h * patch_w)
        patches_flat = embeddings.view(batch_size, self.num_patches_h, self.num_patches_w, self.input_channels, self.patch_height * self.patch_width)

        # Initialize a tensor to hold the reconstructed tensor
        reconstructed_tensor = torch.zeros((batch_size, self.input_channels, self.input_height, self.input_width), device=embeddings.device)

        # Combine the patches back into the original tensor shape
        for i in range(self.num_patches_h):
            for j in range(self.num_patches_w):
                # Get the corresponding embedding for the patch
                patch_embedding = patches_flat[:, i, j, :, :]

                # Reshape it back to the original patch shape
                patch = patch_embedding.view(batch_size, self.input_channels, self.patch_height, self.patch_width)

                # Place the patch in the correct location in the reconstructed tensor
                reconstructed_tensor[:, :, i*self.patch_height:(i+1)*self.patch_height, j*self.patch_width:(j+1)*self.patch_width] = patch

        return reconstructed_tensor
    
class AttentionMapReconstructor(nn.Module):
    """
    A module to reconstruct an image from the attention weights provided by the CLS token.

    This module takes the attention weights derived from a Vision Transformer of shape (batch_size, seq_len, seq_len),
    extract the CLS token attention with the other tokens, and reconstructs an attention map
    by assigning each patch a weight corresponding to the attention the CLS token pays to that patch.

    Parameters:
        input_height (int): Height of the input tensor.
        input_width (int): Width of the input tensor.
        patch_height (int): Height of each patch.
        patch_width (int): Width of each patch.

    Attributes:
        input_height (int): Height of the input tensor.
        input_width (int): Width of the input tensor.
        patch_height (int): Height of each patch.
        patch_width (int): Width of each patch.
        num_patches_h (int): Number of patches along the height.
        num_patches_w (int): Number of patches along the width.

    Returns:
        (torch.Tensor): A reconstructed image of shape (batch_size, 1, input_height, input_width).
            Where each pixel in the reconstructed map represents the attention weight assigned to the corresponding pixel.

    """
    def __init__(self, input_height: int, input_width: int, patch_height: int, patch_width: int):
        super(AttentionMapReconstructor, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.num_patches_h = input_height // patch_height
        self.num_patches_w = input_width // patch_width
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_attention = x[:,0,1:] # Omit self-attention for the CLS token
        #cls_attention = F.softmax(cls_attention, dim=1)
        
        batch_size, seq_len = cls_attention.shape

        # Ensure cls_attention length matches number of patches
        if seq_len != self.num_patches_h * self.num_patches_w:
            raise ValueError(f"Expected seq_len {self.num_patches_h * self.num_patches_w}, but got {seq_len}.")

        # Reshape cls_attention_map to match patch grid
        attention_grid = cls_attention.view(batch_size, self.num_patches_h, self.num_patches_w)
        
        # Create the reconstructed image tensor
        attention_map = torch.zeros((batch_size, 1, self.input_height, self.input_width), device=cls_attention.device)

        # Fill the reconstructed image with attention weights for each patch
        for i in range(self.num_patches_h):
            for j in range(self.num_patches_w):
                # Calculate the position of the patch in the reconstructed image
                start_h = i * self.patch_height
                start_w = j * self.patch_width
                attention_weight = attention_grid[:, i, j].view(batch_size, 1, 1, 1)  # Shape: (batch_size, 1, 1, 1)
                
                # Assign the attention weight to the corresponding patch area
                attention_map[:, :, start_h:start_h + self.patch_height, start_w:start_w + self.patch_width] = attention_weight

        return attention_map

class ViTDecoder(nn.Module):
    """
    A decoder module that reconstructs an image and an attention map
    from the output of a transformer block.

    Parameters:
        input_channels (int): The number of channels in the input image.
        input_height (int): The height of the input image.
        input_width (int): The width of the input image.
        patch_height (int): The height of each patch.
        patch_width (int): The width of each patch.
        embed_dim (int): Dimension of the output embeddings for each patch.
        use_cls_token (bool, optional): Whether to include a classification token in the embeddings. Default is False.

    Attributes:
        num_patches (int): Number of patches calculated from the input dimensions.
        use_cls_token (bool): Flag indicating if a classification token was included in the embeddings.
        patch_reconstructor (PatchReconstructor): Module to reconstruct the image from patch embeddings.
        attention_map_reconstructor (Optional[AttentionMapReconstructor]): Module to reconstruct the attention map from attention weights, initialized only if `use_cls_token` is True.
        embed_scale (bool): Flag indicating if scaling is required based on the embedding dimension and feature dimension.
        scaler (Optional[nn.Linear]): Linear layer to scale embeddings if required.

    Returns:
        (tuple): A tuple containing.
            (torch.Tensor): Reconstructed image of shape (batch_size, input_channels, input_height, input_width).
            (torch.Tensor): Attention map of shape (batch_size, 1, input_height, input_width) if `use_cls_token` is True, else None.
    """
    def __init__(self,
                 input_channels: int,
                 input_height: int,
                 input_width: int, 
                 patch_height: int,
                 patch_width: int,
                 embed_dim: int,
                 use_cls_token: Optional[bool] = False):
        super(ViTDecoder, self).__init__()
        feature_dim = input_channels * patch_height * patch_width
        self.embed_scale = True if embed_dim != feature_dim  else False
        self.use_cls_token = use_cls_token
        self.scaler = nn.Linear(embed_dim, feature_dim) if self.embed_scale else None
        self.patch_reconstructor = PatchReconstructor(input_channels, input_height, input_width, patch_height, patch_width)
        self.attention_map_reconstructor = AttentionMapReconstructor(input_height, input_width, patch_height, patch_width) if self.use_cls_token else None
        self.num_patches = (input_height // patch_height) * (input_width // patch_width)

    def forward(self, embeddings: torch.Tensor, attention_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
            embeddings (torch.Tensor): Output tensor from the transformer block of shape (batch_size, num_patches, embed_dim).
            attention_weights (torch.Tensor): Attention weights from the transformer block of shape (batch_size, num_patches, num_patches).

        Returns:
            (tuple): A tuple containing.
                output_tensor (torch.Tensor): The reconstructed output tensor of shape (batch_size, input_channels, input_height, input_width).
                attention_map (torch.Tensor or None): The attention map of shape (batch_size, 1, input_height, input_width) if `use_cls_token` is True, else None.
        """
        # Reconstruct the output tensor from embeddings
        if self.embed_scale:
            embeddings = self.scaler(embeddings)
        if self.use_cls_token:
            embeddings = embeddings[:, 1:, :] # (batch_size, num_patches + 1, embed_dim) --> (batch_size, num_patches, embed_dim)
        output_tensor = self.patch_reconstructor(embeddings) 

        # Reconstruct the attention map from attention weights
        if self.use_cls_token:
            attention_map = self.attention_map_reconstructor(attention_weights)
            attention_map = attention_map * self.num_patches
        else:
            attention_map = None
        
        return output_tensor, attention_map
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------