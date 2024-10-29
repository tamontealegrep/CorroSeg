
import torch
import torch.nn as nn
from typing import Optional
from src.models.architectures.layers.cbam import CBAM

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class ConvCBAMBlock(nn.Module):
    """
    A block that applies two consecutive convolutional layers with Batch Normalization,
    ReLU activation, and CBAM attention. This is commonly used in U-Net architectures 
    to enhance feature extraction capabilities.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (str, optional): The activation function to use. Default is "ReLU". Options: "ReLU" and "LeakyReLU".
        dropout_prob (float, optional): Probability of dropout. If set, a Dropout layer will be applied after the second convolution. Default is None (no Dropout).
        cbam_reduction (int, optional): Reduction factor for the channel dimension in CBAM. Default is 16.
        cbam_kernel_size (int, optional): Kernel size for the spatial attention in CBAM. Default is 7.
        cbam_activation (str, optional): Activation function to use in channel attention in CBAM. Default is "ReLU". Options: "ReLU" and "LeakyReLU".

    Attributes:
        conv1 (torch.nn.Conv2d): First convolutional layer.
        norm1 (torch.nn.BatchNorm2d): Batch normalization layer after the first convolution.
        conv2 (torch.nn.Conv2d): Second convolutional layer.
        norm2 (torch.nn.BatchNorm2d): Batch normalization layer after the second convolution.
        activation (torch.nn.Module): The activation function.
        cbam (CBAM): CBAM module for channel and spatial attention.
        dropout (torch.nn.Dropout, optional): Dropout layer applied after the second convolution.

    Forward pass:
        The input tensor "x" is passed through the following sequence:
        1. First convolution -> Batch normalization -> Activation
        2. Second convolution -> Batch normalization -> Activation
        3. CBAM for attention
        4. (Optional) Dropout

    Returns:
        torch.Tensor: The output tensor after two convolutional operations with normalization 
        and activation applied, attention and optionally dropout.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: Optional[str] = 'ReLU',
                 dropout_prob: Optional[float] = None,
                 cbam_reduction: int = 16,
                 cbam_kernel_size: int = 7,
                 cbam_activation: Optional[str] = 'ReLU'):
        super(ConvCBAMBlock, self).__init__()
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

        # CBAM module
        self.cbam = CBAM(out_channels, reduction=cbam_reduction, kernel_size=cbam_kernel_size, activation=cbam_activation)

        # Optional dropout layer
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))

        # Apply CBAM
        x = self.cbam(x)

        # Apply dropout if specified
        if self.dropout is not None:
            x = self.dropout(x)

        return x
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------