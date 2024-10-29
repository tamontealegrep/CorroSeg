
import torch
import torch.nn as nn

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class PatchExtractor(nn.Module):
    """
    A module to extract patches from input tensors.

    This class takes a batch of tensors with dimensions (B, C, H, W) 
    and extracts patches of size (height, width) from each tensor,
    returning a tensor with dimensions (B, num_patches, C, height, width).

    Args:
        height (int): Height of each patch, how tall each extracted patch will be.
        width (int): Width of each patch, how wide each extracted patch will be.

    Attributes:
        height (int): Height of each patch.
        width (int): Width of each patch.
    
    Returns:
        torch.Tensor: A tensor of shape (B, num_patches, C, height, width).
    """

    def __init__(self, height:int, width:int):
        super(PatchExtractor, self).__init__()
        self.height = height
        self.width = width

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        assert h % self.height == 0 and w % self.width == 0, "Tensor dimensions must be divisible by patch size."

        num_patches_h = h // self.height
        num_patches_w = w // self.width

        # Create patches
        patches = x.unfold(2, self.height, self.height).unfold(3, self.width, self.width)

        # Rearrange the dimensions
        patches = patches.contiguous().view(b, c, num_patches_h, num_patches_w, self.height, self.width)

        # Rearrange to get (B, num_patches, C, height, width)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(b, num_patches_h * num_patches_w, c, self.height, self.width)

        return patches

class PatchEmbedding(nn.Module):
    """
    A module to create patch embeddings from input tensors.

    This class uses the PatchExtractor to extract patches from input tensors of shape (B, C, H, W),
    B is the batch size, C is the number of channels, H is the height, and W is the width;
    and then applies a linear transformation to create embeddings for each patch.

    Args:
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
        torch.Tensor: A tensor of shape (B, num_patches, embed_dim) containing the embeddings for each patch.
    """

    def __init__(self, input_channels:int, input_height: int, input_width: int, patch_height: int, patch_width: int, embed_dim: int):
        super(PatchEmbedding, self).__init__()
        
        self.patch_extractor = PatchExtractor(height=patch_height, width=patch_width)
        self.num_patches = (input_height // patch_height) * (input_width // patch_width)
        self.projection = nn.Linear(input_channels * patch_height * patch_width, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract patches using the PatchExtractor (B, N, C, H, W)
        patches = self.patch_extractor(x)

        # Reshape patches to (B, N, C * H * W)
        b, n, c, h, w = patches.shape
        patches = patches.reshape(b, n, c * h * w) 
        # Apply the linear projection to get embeddings (B, N, embed_dim)
        embeddings = self.projection(patches)

        return embeddings
    
class PositionalEmbedding(nn.Module):
    """
    A module to create positional embeddings for the input patches.

    This class generates a learnable positional embedding for each patch in the input
    and adds it to the provided input tensor.

    Args:
        num_patches (int): Number of patches.
        embed_dim (int): Dimension of the output embeddings.

    Attributes:
        position_embeddings (nn.Parameter): Learnable positional embeddings of shape (1, num_patches, embed_dim).
    
    Returns:
        torch.Tensor: A tensor of shape (B, num_patches, embed_dim) containing the summed 
                      positional embeddings with the input tensor, where B is the batch size.
    """

    def __init__(self, num_patches: int, embed_dim: int):
        super(PositionalEmbedding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.empty(1, num_patches, embed_dim), requires_grad=True)
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.position_embeddings

class ViTEmbedding(nn.Module):
    """
    A block to combine patch embeddings and positional embeddings for input tensors for a Visual Transformer.

    This class processes input tensors of shape (B, C, H, W) to extract patches, 
    apply positional embeddings, and incorporate a classification token (cls_token).

    Args:
        input_channels (int): Number of channels of the input tensors.
        input_height (int): Height of the input tensors.
        input_width (int): Width of the input tensors.
        patch_height (int): Height of each patch.
        patch_width (int): Width of each patch.
        embed_dim (int): Dimension of the output embeddings for each patch.

    Attributes:
        patch_embedding (PatchEmbedding): Instance of PatchEmbedding to create embeddings from patches.
        positional_embedding (PositionalEmbedding): Instance of PositionalEmbedding to add positional information.
        cls_token (nn.Parameter): A learnable classification token of shape (1, 1, embed_dim) that is prepended to the embeddings.
        num_patches (int): The total number of patches extracted from the input tensors.
    
    Returns:
        torch.Tensor: A tensor of shape (B, num_patches + 1, embed_dim) containing the combined embeddings, including the classification token.
    """

    def __init__(self,
                 input_channels: int,
                 input_height: int,
                 input_width: int,
                 patch_height: int,
                 patch_width: int,
                 embed_dim: int):
        super(ViTEmbedding, self).__init__()
        self.patch_embedding = PatchEmbedding(input_channels, input_height, input_width, patch_height, patch_width, embed_dim)
        self.num_patches = self.patch_embedding.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim),requires_grad=True) # (1, 1, embed_dim)
        self.positional_embedding = PositionalEmbedding(self.num_patches + 1, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        patch_embeddings = self.patch_embedding(x)

        cls_token = self.cls_token.expand(batch_size, -1, -1)  # Expand to (B, 1, embed_dim)

        embeddings = self.positional_embedding(torch.cat((cls_token, patch_embeddings), dim=1))

        return embeddings
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------