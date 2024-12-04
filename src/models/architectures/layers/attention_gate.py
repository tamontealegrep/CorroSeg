
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class AttentionGate(nn.Module):
    """
    Attention Gate module for selectively emphasizing features from skip connections 
    in a neural network, as commonly used in U-Net architectures.

    This module computes attention weights by combining a gating signal with skip 
    connection features through convolutional layers. The output is the input 
    feature map reweighted by the attention mechanism, enhancing the relevant features.

    Parameters:
        input_channels_x (int): Number of input channels from the skip connection.
        input_channels_g (int): Number of input channels from the gating signal.
        output_channels (int): Number of output channels for the attention gate.
        activation (str, optional): Activation function to use. Default is "ReLU". Options: "ReLU" and "LeakyReLU".

    Attributes:
        W_g (nn.Conv2d): Convolutional layer applied to the gating signal.
        W_x (nn.Conv2d): Convolutional layer applied to the skip connection.
        psi (nn.Conv2d): Final convolutional layer for combining features.
        activation (nn.Module): Activation function used in the attention mechanism.

    Forward pass:
        The input tensors "x" (skip connection) and "g" (gating signal) are passed through 
        the following sequence:
        1. The gating signal "g" is convolved with W_g to extract features.
        2. The skip connection "x" is convolved with W_x to extract corresponding features.
        3. The results from the previous two steps are summed and passed through the 
           specified activation function.
        4. The combined features are processed through the psi convolutional layer to 
           produce a single-channel output.
        5. A Sigmoid activation function is applied to generate attention weights.
        6. The input tensor "x" is multiplied by the attention weights, emphasizing 
           relevant features.

    Returns:
        (torch.Tensor): The output tensor of shape (batch_size, channels, height, width), where each channel is 
            reweighted based on the attention mechanism.
    """
    def __init__(self,
                 input_channels_x: int,
                 input_channels_g: int,
                 output_channels: int,
                 activation: Optional[str] = "ReLU"):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv2d(input_channels_g, output_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(input_channels_x, output_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(output_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)

        # Set the activation function
        if activation == 'ReLU':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError("Unsupported activation type. Choose 'ReLU' or 'LeakyReLU'.")

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        conv_g = self.W_g(g)
        conv_x = self.W_x(x)
        psi = self.activation(conv_g + conv_x)
        psi = self.psi(psi)
        attention = F.sigmoid(psi)
        return x * attention


#-----------------------------------------------------------------------------------------------------------------------------------------------------