
import torch
import torch.nn as nn

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class ConvPatchExtractor(nn.Module):
    """
    A module to extract patches from input tensors using convolution.

    This class takes a batch of tensors with dimensions (B, C, H, W) 
    and extracts patches of size (height, width) from each tensor,
    returning a tensor with dimensions (B, num_patches, C, height, width).

    Parameters:
        height (int): Height of each patch, how tall each extracted patch will be.
        width (int): Width of each patch, how wide each extracted patch will be.

    Attributes:
        height (int): Height of each patch.
        width (int): Width of each patch.
        conv (nn.Conv2d): Convolutional layer used to extract patches.
    
    Returns:
        (torch.Tensor): A tensor of shape (B, num_patches, C, height, width).
    """

    def __init__(self, height: int, width: int):
        super(ConvPatchExtractor, self).__init__()
        self.height = height
        self.width = width

        # Define a convolutional layer to extract patches
        self.conv = nn.Conv2d(
            in_channels=1,   # We'll treat each patch extraction as a single channel
            out_channels=1,  # Output channel is also 1
            kernel_size=(height, width),
            stride=(height, width),
            padding=0,
            bias=False  # No bias is needed for patch extraction
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        assert h % self.height == 0 and w % self.width == 0, "Tensor dimensions must be divisible by patch size."

        print(x.shape)
        # Reshape the input to treat each channel separately
        x = x.view(b * c, 1, h, w)  # Shape becomes (B*C, 1, H, W)
        print(x.shape)
        # Use convolution to extract patches
        patches = self.conv(x)  # Output shape: (B*C, 1, num_patches_h, num_patches_w)
        print(patches.shape)
        # Calculate number of patches
        num_patches_h = h // self.height
        num_patches_w = w // self.width

        # Reshape back to (B, num_patches, C, height, width)
        patches = patches.view(b, c, num_patches_h, num_patches_w, 1, self.height, self.width)

        # Rearrange dimensions to get (B, num_patches, C, height, width)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(b, num_patches_h * num_patches_w, c, self.height, self.width)

        return patches
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------