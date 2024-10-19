
import torch
import torch.nn as nn
from src.models.architectures.blocks.double_conv_block import DoubleConvBlock

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """
    A decoder block that performs upsampling followed by a DoubleConvBlock.
    This block is used to increase the spatial dimensions of the feature maps 
    while refining the extracted features.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (str): The activation function to use in the DoubleConvBlock. Default is "ReLU". Options: "ReLU" and "LeakyReLU".
        dropout_prob (float, optional): Probability of dropout to apply after the DoubleConvBlock. Default is None (no Dropout).

    Attributes:
        upconv (torch.nn.ConvTranspose2d): Transposed convolution layer for upsampling.
        double_conv (DoubleConvBlock): The DoubleConvBlock for feature extraction after concatenation.

    Forward pass:
        The input tensors "x" and "skip" are passed through through the following sequence:
        1. Conv Transpose 2D -> Concatenate (x,skip) -> DoubleConvBlock

    Returns:
        torch.Tensor: The output tensor after applying upsampling, concatenation and the DoubleConvBlock.
    """
    
    def __init__(self, in_channels, out_channels, activation='ReLU', dropout_prob=None):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConvBlock(in_channels, out_channels, activation, dropout_prob)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat((x, skip), dim=1)
        x = self.double_conv(x)
        return x
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------