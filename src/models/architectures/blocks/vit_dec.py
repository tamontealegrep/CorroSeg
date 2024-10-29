
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class PatchReconstructor(nn.Module):
    """
    A module to reconstruct an image from its patch embeddings.

    This module takes embeddings derived from patches of an input image
    and rearranges them to reconstruct the original image. Each embedding
    corresponds to a patch of the input image, and the module combines
    these patches back into the full image.

    Args:
        input_channels (int): The number of channels in the input image (e.g., 3 for RGB images).
        input_height (int): The height of the input image.
        input_width (int): The width of the input image.
        patch_height (int): The height of each patch.
        patch_width (int): The width of each patch.

    Attributes:
        input_channels (int): The number of channels in the input image.
        input_height (int): The height of the input image.
        input_width (int): The width of the input image.
        patch_height (int): The height of each patch.
        patch_width (int): The width of each patch.
        num_patches_h (int): The number of patches along the height of the image.
        num_patches_w (int): The number of patches along the width of the image.

    Returns:
        torch.Tensor: A reconstructed image of shape (batch_size, input_channels, input_height, input_width),
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

        # Reshape embeddings to (B, num_patches_h, num_patches_w, C * patch_h * patch_w)
        patches_flat = embeddings.view(batch_size, self.num_patches_h, self.num_patches_w, embed_dim)

        # Initialize a tensor to hold the reconstructed image
        reconstructed_image = torch.zeros((batch_size, self.input_channels, self.input_height, self.input_width), device=embeddings.device)

        # Combine the patches back into the original image shape
        for i in range(self.num_patches_h):
            for j in range(self.num_patches_w):
                # Get the corresponding embedding for the patch
                patch_embedding = patches_flat[:, i, j, :]

                # Reshape it back to the original patch shape
                patch = patch_embedding.view(batch_size, self.input_channels, self.patch_height, self.patch_width)

                # Place the patch in the correct location in the reconstructed image
                reconstructed_image[:, :, i*self.patch_height:(i+1)*self.patch_height, j*self.patch_width:(j+1)*self.patch_width] = patch

        return reconstructed_image
    
class AttentionMapReconstructor(nn.Module):
    """
    A module to reconstruct an image from the attention weights provided by the CLS token.

    This module takes the attention weights derived from a Vision Transformer of shape (batch_size, seq_len, seq_len),
    extract the CLS token attention with the other tokens, and reconstructs an attention map
    by assigning each patch a weight corresponding to the attention the CLS token pays to that patch.

    Args:
        input_height (int): Height of the input image.
        input_width (int): Width of the input image.
        patch_height (int): Height of each patch.
        patch_width (int): Width of each patch.

    Attributes:
        input_height (int): Height of the input image.
        input_width (int): Width of the input image.
        patch_height (int): Height of each patch.
        patch_width (int): Width of each patch.
        num_patches_h (int): Number of patches along the height.
        num_patches_w (int): Number of patches along the width.

    Returns:
        torch.Tensor: A reconstructed image of shape (batch_size, 1, input_height, input_width).
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
        cls_attention = F.softmax(cls_attention, dim=1)
        
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

    Args:
        input_channels (int): The number of channels in the input image (e.g., 3 for RGB images).
        input_height (int): The height of the input image.
        input_width (int): The width of the input image.
        patch_height (int): The height of each patch.
        patch_width (int): The width of each patch.

    Attributes:
        patch_reconstructor (PatchReconstructor): Module to reconstruct the image from patch embeddings.
        attention_map_reconstructor (AttentionMapReconstructor): Module to reconstruct the attention map from attention weights.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            torch.Tensor: Reconstructed image of shape (B, C, H, W).
            torch.Tensor: Attention map of shape (B, 1, H, W).
    """
    def __init__(self,
                 input_channels: int,
                 input_height: int,
                 input_width: int, 
                 patch_height: int,
                 patch_width: int):
        super(ViTDecoder, self).__init__()
        self.patch_reconstructor = PatchReconstructor(input_channels, input_height, input_width, 
                                                      patch_height, patch_width)
        self.attention_map_reconstructor = AttentionMapReconstructor(input_height, input_width, 
                                                                    patch_height, patch_width)

    def forward(self, embeddings: torch.Tensor, attention_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings (torch.Tensor): Output tensor from the transformer block of shape (B, N, embed_dim).
            attention_weights (torch.Tensor): Attention weights from the transformer block of shape (B, N, N).

        Returns:
            output_tensor (torch.Tensor): The reconstructed output tensor of shape (B, C, H, W).
            attention_map (torch.Tensor): The attention map of shape (B, 1, H, W).
        """
        # Reconstruct the output tensor from embeddings
        output_tensor = self.patch_reconstructor(embeddings)

        # Reconstruct the attention map from attention weights
        attention_map = self.attention_map_reconstructor(attention_weights)

        return output_tensor, attention_map
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------