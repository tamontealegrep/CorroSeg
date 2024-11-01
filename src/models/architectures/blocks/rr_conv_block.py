
import torch
import torch.nn as nn
from typing import Optional
from src.models.architectures.layers.cbam import CBAM

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class RRConvBlock(nn.Module):
    """
    A recurrent residual convolutional block that enhances feature extraction by applying two consecutive 
    convolutional layers with an activation function, Batch Normalization and optionally CBAM attention and dropout.
    This block is commonly utilized in architectures like U-Net for improved representational power.

    Args:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.
        activation (str, optional): The activation function to use. Default is "ReLU". Options: "ReLU" and "LeakyReLU".
        dropout_prob (float, optional): Probability of dropout. If specified, a Dropout layer will be applied after the second convolution. Default is None (no Dropout).
        num_recurrences (int, optional): Number of times to apply the convolutions in a recurrent manner. Default is 3.
        cbam (bool, optional): Whether to include the Convolutional Block Attention Module (CBAM). Default is False.
        cbam_reduction (int, optional): Reduction factor for the channel dimension in CBAM. Default is 16.
        cbam_kernel_size (int, optional): Kernel size for the spatial attention in CBAM. Default is 7.
        cbam_activation (str, optional): Activation function to use in channel attention in CBAM. Default is "ReLU". Options: "ReLU" and "LeakyReLU".

    Attributes:
        conv_adjust (torch.nn.Conv2d): 1x1 convolutional layer to adjust the number of input channels.
        conv1 (torch.nn.Conv2d): First convolutional layer.
        norm1 (torch.nn.BatchNorm2d): Batch normalization layer after the first convolution.
        conv2 (torch.nn.Conv2d): Second convolutional layer.
        norm2 (torch.nn.BatchNorm2d): Batch normalization layer after the second convolution.
        activation (torch.nn.Module): The chosen activation function.
        cbam (CBAM): CBAM module for channel and spatial attention.
        dropout (torch.nn.Dropout, optional): Dropout layer applied after the second convolution.

    Forward pass:
        The input tensor "x" undergoes the following transformations:
        1. Adjusts the input tensor using a 1x1 convolution.
        2. Applies two sets of convolutions, each followed by activation and normalization, repeated num_recurrences times.
        3. Adds the input tensor to the convolution output.
        4. Applies the CBAM module for attention, if enabled.
        5. Applies dropout if specified.

    Returns:
        torch.Tensor: The output tensor after the convolutional operations, normalization, 
        activation, residual connection and optionally cbam and dropout.
    """
    def __init__(self,
                 input_channels:int,
                 output_channels:int,
                 activation:Optional[str]="ReLU",
                 dropout_prob:Optional[float]=None,
                 num_recurrences:int=3,
                 cbam:float=False,
                 cbam_reduction:int=16,
                 cbam_kernel_size:int=7,
                 cbam_activation:Optional[str]="ReLU"):
        super(RRConvBlock, self).__init__()
        self.conv_adjust = nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
        self.conv1 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(output_channels)
        self.num_recurrences = num_recurrences

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
        x = self.conv_adjust(x)
        identity = x.clone()

        for _ in range(self.num_recurrences):
            x = self.activation(self.conv1(x))
        x = self.norm1(x)

        for _ in range(self.num_recurrences):
            x = self.activation(self.conv2(x))
        x = self.norm2(x)

        x = x + identity

        # Apply cbam if specified
        if self.cbam is not None:
            x = self.cbam(x)

        # Apply dropout if specified
        if self.dropout is not None:
            x = self.dropout(x)

        return x

#-----------------------------------------------------------------------------------------------------------------------------------------------------
