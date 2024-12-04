
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """
    Channel Attention Module as described in the CBAM paper.

    This module computes channel attention by applying global average pooling 
    and global max pooling on the input feature map, followed by a shared 
    multilayer perceptron (MLP) to generate channel-wise attention weights.
    
    Parameters:
        input_channels (int): Number of input channels.
        reduction (int, optional): Reduction factor for the channel dimension. Default is 16.
        activation (str, optional): Activation function to use. Default is "ReLU". Options: "ReLU" and "LeakyReLU".

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        activation (nn.Module): Activation function used in the MLP.

    Forward pass:
        The input tensor "x" is passed through the following sequence:
        1. Global Average Pooling and Global Max Pooling on spatial dimensions (H, W).
        2. A fully connected layer (fc1) reduces the dimensionality by a factor of "reduction".
        3. An activation function is applied (ReLU or LeakyReLU).
        4. Another fully connected layer (fc2) projects the output back to the original channel dimension.
        5. The results from the average and max pooling paths are summed and passed through a Sigmoid activation.
    
    Returns:
        (torch.Tensor): The output tensor of shape (batch_size, channels, 1, 1), where each channel is reweighted
            based on the attention mechanism.
    """
    def __init__(self,
                 input_channels: int,
                 reduction: Optional[int] = 16,
                 activation: Optional[str]='ReLU'):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(input_channels, input_channels // reduction, bias=False)
        self.fc2 = nn.Linear(input_channels // reduction, input_channels, bias=False)

        # Set the activation function
        if activation == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "LeakyReLU":
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError("Unsupported activation type. Choose 'ReLU' or 'LeakyReLU'.")

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)

        avg_out = self.fc2(self.activation(self.fc1(avg_pool)))
        max_out = self.fc2(self.activation(self.fc1(max_pool)))

        return torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module as described in the CBAM paper.

    This module computes spatial attention by applying both average and max 
    pooling operations across the channel dimension. The results are concatenated 
    and passed through a convolutional layer to generate spatial attention weights.

    Parameters:
        kernel_size (int, optional): Size of the convolution kernel. Default is 7.

    Attributes:
        conv (nn.Conv2d): Convolutional layer used to generate spatial attention weights.

    Forward pass:
        The input tensor "x" is passed through the following sequence:
        1. Compute the average pooling and max pooling across the channel dimension.
        2. Concatenate the pooled results along the channel dimension.
        3. Apply a convolutional layer to the concatenated tensor to produce a single-channel output.
        4. Apply a Sigmoid activation to generate the final attention map.

    Returns:
        (torch.Tensor): The output tensor of shape (batch_size, 1, height, width), where each pixel is reweighted 
            based on the spatial attention mechanism.
    """
    def __init__(self, kernel_size:int=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        return torch.sigmoid(self.conv(concat))

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).

    CBAM integrates both channel and spatial attention mechanisms to refine 
    feature representations. The module applies channel attention followed by 
    spatial attention to the input feature map, enhancing important features 
    while suppressing less relevant ones.

    Parameters:
        input_channels (int): Number of input channels.
        reduction (int, optional): Reduction factor for the channel dimension in channel attention. Default is 16.
        kernel_size (int, optional): Size of the convolution kernel for spatial attention. Default is 7.
        activation (str, optional): Activation function to use in channel attention. Default is "ReLU". Options: "ReLU" and "LeakyReLU".

    Attributes:
        channel_attention (ChannelAttention): Instance of the channel attention module.
        spatial_attention (SpatialAttention): Instance of the spatial attention module.

    Forward pass:
        The input tensor "x" is processed as follows:
        1. Channel attention is computed and applied to the input tensor.
        2. Spatial attention is computed and applied to the resulting tensor.
        3. The final output is the input tensor weighted by both channel and spatial attention.

    Returns:
        (torch.Tensor): The output tensor after applying both attention mechanisms, with the same shape as the input.
    """
    def __init__(self,
                 input_channels: int,
                 reduction: Optional[int] = 16,
                 kernel_size: Optional[int] = 7,
                 activation: Optional[str] = "ReLU"):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(input_channels, reduction, activation)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        channel_attention = self.channel_attention(x)
        x = x * channel_attention

        spatial_attention = self.spatial_attention(x)
        x = x * spatial_attention

        return x

#-----------------------------------------------------------------------------------------------------------------------------------------------------