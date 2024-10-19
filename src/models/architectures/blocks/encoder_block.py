
import torch.nn as nn
from src.models.architectures.blocks.double_conv_block import DoubleConvBlock

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """
    An encoder block that applies a DoubleConvBlock followed by a max pooling operation.
    This block is used to downsample the input feature maps while extracting important features.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (str): The activation function to use in the DoubleConvBlock. Default is "ReLU". Options: "ReLU" and "LeakyReLU".
        dropout_prob (float, optional): Probability of dropout to apply after the DoubleConvBlock. Default is None (no Dropout).

    Attributes:
        double_conv (DoubleConvBlock): The DoubleConvBlock for feature extraction.
        pool (torch.nn.MaxPool2d): Max pooling layer for downsampling.

    Forward pass:
        The input tensor "x" is passed through the following sequence:
        1. DoubleConvBlock -> Max Pooling 2D

    Returns:
        tuple: A tuple containing:
        - x (torch.Tensor): The output tensor after the DoubleConvBlock and 
        - x_pooled (torch.Tensor): The output tensor after max pooling.
    """
    
    def __init__(self, in_channels, out_channels, activation='ReLU', dropout_prob=None):
        super(EncoderBlock, self).__init__()
        self.double_conv = DoubleConvBlock(in_channels, out_channels, activation, dropout_prob)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.double_conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------