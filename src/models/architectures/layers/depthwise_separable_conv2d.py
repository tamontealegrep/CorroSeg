import torch
import torch.nn as nn

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution.

    This class implements a depthwise separable convolution, which consists of 
    two steps: a depthwise convolution that applies a filter to each channel 
    independently, followed by a pointwise convolution that combines the results.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the kernel for the depthwise convolution.
        stride (int, optional): Stride for the convolutions. Default is 1.
        padding (int, optional): Padding for the convolutions. Default is 0.
        bias (bool, optional): Whether to include a bias term in the convolutions. Default is True.

    Returns:
        torch.Tensor: The output tensor.
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 bias: bool = True):
        super(DepthwiseSeparableConv2d, self).__init__()
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------