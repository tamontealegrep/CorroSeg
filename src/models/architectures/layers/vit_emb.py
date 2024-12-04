
import torch
import torch.nn as nn
from typing import Optional

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class PatchExtractor(nn.Module):
    """
    A module to extract patches from input tensors.

    This class takes a batch of tensors with dimensions (batch_size, channels, height, width)
    and extracts patches of size (height, width) from each tensor,
    returning a tensor with dimensions (batch_size, num_patches, channels, patch_height, patch_width).

    Parameters:
        patch_height (int): Height of each patch, how tall each extracted patch will be.
        patch_width (int): Width of each patch, how wide each extracted patch will be.

    Attributes:
        patch_height (int): Height of each patch.
        patch_width (int): Width of each patch.
    
    Returns:
        (torch.Tensor): A tensor of shape (batch_size, num_patches, channels, patch_height, patch_width).
    """

    def __init__(self, patch_height:int, patch_width:int):
        super(PatchExtractor, self).__init__()
        self.patch_height = patch_height
        self.patch_width = patch_width

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        assert height % self.patch_height == 0 and width % self.patch_width == 0, "Tensor dimensions must be divisible by patch size."

        num_patches_h = height // self.patch_height
        num_patches_w = width // self.patch_width

        # Create patches
        patches = x.unfold(2, self.patch_height, self.patch_height).unfold(3, self.patch_width, self.patch_width)

        # Rearrange (batch_size, channels, num_patches_h, num_patches_w, patch_height, patch_width)
        patches = patches.contiguous().view(batch_size, channels, num_patches_h, num_patches_w, self.patch_height, self.patch_width)

        # Rearrange (batch_size, num_patches, channels, patch_height, patch_width)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, num_patches_h * num_patches_w, channels, self.patch_height, self.patch_width)

        return patches

class PatchEmbedding(nn.Module):
    """
    A module to create patch embeddings from input tensors.

    This class uses the PatchExtractor to extract patches from input tensors of shape (batch_size, channels, height, width),
    and then applies a linear transformation to create embeddings for each patch of shape (batch_size, num_patches, embed_dim).

    Parameters:
        input_channels (int): Number of channels of the input tensors.
        input_height (int): Height of the input tensors.
        input_width (int): Width of the input tensors.
        patch_height (int): Height of each patch.
        patch_width (int): Width of each patch.
        embed_dim (int): Dimension of the output embeddings for each patch.

    Attributes:
        patch_extractor (PatchExtractor): Instance of PatchExtractor to extract patches.
        projection (nn.Linear): Linear layer to project patches to embedding dimension.
        num_patches (int): Total number of patches extracted from the input tensor.

    Returns:
        (torch.Tensor): A tensor of shape (batch_size, num_patches, embed_dim) containing the embeddings for each patch.
    """

    def __init__(self, input_channels:int, input_height: int, input_width: int, patch_height: int, patch_width: int, embed_dim: int):
        super(PatchEmbedding, self).__init__()
        
        self.patch_extractor = PatchExtractor(patch_height, patch_width)
        self.num_patches = (input_height // patch_height) * (input_width // patch_width)
        self.projection = nn.Linear(input_channels * patch_height * patch_width, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract patches (batch_size, num_patches, channels, patch_height, patch_width)
        patches = self.patch_extractor(x)

        # Reshape patches to (batch_size, num_patches, channels * patch_height * patch_width)
        b, n, c, h, w = patches.shape
        patches = patches.reshape(b, n, c * h * w) 
        # Apply the linear projection to get embeddings (batch_size, num_patches, embed_dim)
        embeddings = self.projection(patches)

        return embeddings
    
class PositionalEmbedding(nn.Module):
    """
    A module to create positional embeddings for the input patches.

    This class generates a learnable positional embedding for each patch in the input
    and adds it to the provided input tensor.

    Parameters:
        num_patches (int): Number of patches.
        embed_dim (int): Dimension of the output embeddings.

    Attributes:
        position_embeddings (nn.Parameter): Learnable positional embeddings of shape (1, num_patches, embed_dim).
    
    Returns:
        (torch.Tensor): A tensor of shape (batch_size, num_patches, embed_dim) containing the summed 
            positional embeddings with the input tensor.
    """

    def __init__(self, num_patches: int, embed_dim: int):
        super(PositionalEmbedding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.rand(1, num_patches, embed_dim), requires_grad=True)
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.position_embeddings

class ViTEmbedding(nn.Module):
    """
    A block to combine patch embeddings and positional embeddings for input tensors for a Visual Transformer.

    This class processes input tensors of shape (batch_size, channels, height, width) to extract patches, 
    apply positional embeddings, and optionally incorporate a classification token (cls_token).

    Parameters:
        input_channels (int): Number of channels of the input tensors.
        input_height (int): Height of the input tensors.
        input_width (int): Width of the input tensors.
        patch_height (int): Height of each patch.
        patch_width (int): Width of each patch.
        embed_dim (int): Dimension of the output embeddings for each patch.
        use_cls_token (bool, optional): Whether to include a classification token. Default False.

    Attributes:
        patch_embedding (PatchEmbedding): Instance of PatchEmbedding to create embeddings from patches.
        positional_embedding (PositionalEmbedding): Instance of PositionalEmbedding to add positional information.
        num_patches (int): The total number of patches extracted from the input tensors.
        use_cls_token (bool): Flag indicating whether to include the classification token in the embeddings.
        cls_token (nn.Parameter): A learnable classification token of shape (1, 1, embed_dim) that is prepended to the embeddings.    
    
    Returns:
        (torch.Tensor): A tensor of shape (batch_size, num_patches + 1, embed_dim) if cls_token is used, 
            else (batch_size, num_patches, embed_dim).
    """

    def __init__(self,
                 input_channels: int,
                 input_height: int,
                 input_width: int,
                 patch_height: int,
                 patch_width: int,
                 embed_dim: int,
                 use_cls_token: Optional[bool] = False):
        super(ViTEmbedding, self).__init__()
        self.patch_embedding = PatchEmbedding(input_channels, input_height, input_width, patch_height, patch_width, embed_dim)
        self.num_patches = self.patch_embedding.num_patches
        self.use_cls_token = use_cls_token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim),requires_grad=True) if self.use_cls_token else None
        self.positional_embedding = PositionalEmbedding(self.num_patches + 1 if self.use_cls_token else self.num_patches, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        patch_embeddings = self.patch_embedding(x)

        if self.use_cls_token:
            cls_token = self.cls_token.expand(batch_size, -1, -1)  # Expand to (batch_size, 1, embed_dim)
            embeddings = self.positional_embedding(torch.cat((cls_token, patch_embeddings), dim=1))
        else:
            embeddings = self.positional_embedding(patch_embeddings)

        return embeddings
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------