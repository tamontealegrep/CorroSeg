
import torch.nn as nn
from typing import Optional

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """
    A block that applies two consecutive convolutional layers with Batch Normalization
    and ReLU activation. This is commonly used in U-Net architectures to enhance feature 
    extraction capabilities.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (str, optional): The activation function to use. Default is "ReLU". Options: "ReLU" and "LeakyReLU".
        dropout_prob (float, optional): Probability of dropout. If set, a Dropout layer will be applied after the second convolution. Default is None (no Dropout).

    Attributes:
        conv1 (torch.nn.Conv2d): First convolutional layer.
        norm1 (torch.nn.BatchNorm2d): Batch normalization layer after the first convolution.
        conv2 (torch.nn.Conv2d): Second convolutional layer.
        norm2 (torch.nn.BatchNorm2d): Batch normalization layer after the second convolution.
        activation (torch.nn.Module): The activation function.
        dropout (torch.nn.Dropout, optional): Dropout layer applied after the second convolution.

    Forward pass:
        The input tensor "x" is passed through the following sequence:
        1. First convolution -> Batch normalization -> Activation
        2. Second convolution -> Batch normalization -> Activation
        3. (Optional) Dropout

    Returns:
        torch.Tensor: The output tensor after two convolutional operations with normalization and activation applied, and optionally dropout.
    """
    def __init__(self, in_channels:int, out_channels:int, activation:Optional[str]='ReLU', dropout_prob:Optional[float]=None):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)

        # Set the activation function
        if activation == 'ReLU':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError("Unsupported activation type. Choose 'ReLU' or 'LeakyReLU'.")

        # Optional dropout layer
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob is not None else None

    def forward(self, x):
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))

        # Apply dropout if specified
        if self.dropout is not None:
            x = self.dropout(x)

        return x

#-----------------------------------------------------------------------------------------------------------------------------------------------------
