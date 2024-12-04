
import torch
import torch.nn as nn
from typing import Optional
from src.models.architectures.layers.cbam import CBAM
from src.models.architectures.layers.depthwise_separable_conv2d import DepthwiseSeparableConv2d

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class ResConvBlock(nn.Module):
    """
    A residual convolutional block that applies two consecutive convolutional layers with 
    Batch Normalization and an activation function. This block is designed to enhance feature 
    extraction capabilities and improve gradient flow through residual connections.

    Parameters:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.
        activation (str, optional): The activation function to use. Default is "ReLU". Options: "ReLU" and "LeakyReLU".
        dropout_prob (float, optional): Probability of dropout. If set, a Dropout layer will be applied after the second convolution. 
            Default is None (no Dropout).
        cbam (bool, optional): Whether to include the Convolutional Block Attention Module (CBAM). Default is False.
        cbam_reduction (int, optional): Reduction factor for the channel dimension in CBAM. Default is 16.
        cbam_kernel_size (int, optional): Kernel size for the spatial attention in CBAM. Default is 7.
        cbam_activation (str, optional): Activation function to use in channel attention in CBAM. 
            Default is "ReLU". Options: "ReLU" and "LeakyReLU".
        use_depthwise (bool, optional): Whether to use depthwise separable convolutions instead of standard convolutions. Default is False.

    Attributes:
        conv_adjust (torch.nn.Conv2d): 1x1 convolutional layer used to adjust the number of input channels to match output channels.
        conv1 (torch.nn.Module): First convolutional layer (can be standard or depthwise separable).
        conv2 (torch.nn.Module): Second convolutional layer (can be standard or depthwise separable).
        norm1 (torch.nn.BatchNorm2d): Batch normalization layer after the first convolution.
        norm2 (torch.nn.BatchNorm2d): Batch normalization layer after the second convolution.
        activation (torch.nn.Module): The activation function.
        dropout (torch.nn.Dropout, optional): Dropout layer applied after the second convolution.

    Forward pass:
        The input tensor "x" undergoes the following transformations:
        1. Adjusts the input tensor using a 1x1 convolution.
        2. Applies two sets of convolutions, each followed by activation and normalization.
        3. Adds the input tensor to the convolution output.
        4. Applies the CBAM module for attention, if enabled.
        5. Applies dropout if specified.

    Returns:
        (torch.Tensor): The output tensor after applying two convolutional operations with 
            normalization and activation applied, residual connection and optionally cbam and dropout.
    """
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 activation: Optional[str] = "ReLU",
                 dropout_prob: Optional[float] = None,
                 cbam: Optional[float] = False,
                 cbam_reduction: Optional[int] = 16,
                 cbam_kernel_size: Optional[int] = 7,
                 cbam_activation: Optional[str] = "ReLU",
                 use_depthwise: Optional[bool] = False):
        super(ResConvBlock, self).__init__()
        self.conv_adjust = nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
        if use_depthwise:
            self.conv1 = DepthwiseSeparableConv2d(output_channels, output_channels, kernel_size=3, padding=1)
            self.conv2 = DepthwiseSeparableConv2d(output_channels, output_channels, kernel_size=3, padding=1)
        else:
            self.conv1 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(output_channels)
        self.norm2 = nn.BatchNorm2d(output_channels)

        # Set the activation function
        if activation == 'ReLU':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError("Unsupported activation type. Choose 'ReLU' or 'LeakyReLU'.")

        # Optional CBAM module
        self.cbam = CBAM(output_channels, reduction=cbam_reduction, kernel_size=cbam_kernel_size, activation=cbam_activation) if cbam else None

        # Optional dropout layer
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob is not None else None

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        identity = self.conv_adjust(x)

        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))

        x = x + identity

        # Apply cbam if specified
        if self.cbam is not None:
            x = self.cbam(x)

        # Apply dropout if specified
        if self.dropout is not None:
            x = self.dropout(x)

        return x 

#-----------------------------------------------------------------------------------------------------------------------------------------------------
