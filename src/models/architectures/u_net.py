import torch
import torch.nn as nn
from src.models import DEVICE 
from src.models.architectures.blocks.double_conv_block import DoubleConvBlock
from src.models.architectures.blocks.encoder_block import EncoderBlock
from src.models.architectures.blocks.decoder_block import DecoderBlock
from src.models.train.train import train_model as tm
from src.models.train.train import train_validation_model as tvm

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class UNet(nn.Module):
    """
    U-Net model for semantic segmentation. This architecture consists of 
    an encoder-decoder structure with skip connections between the encoder 
    and decoder blocks.

    Args:
        in_channels (int): Number of input channels (e.g., 1 for grayscale images).
        out_channels (int): Number of output channels (number of classes for segmentation).
        base_channels (int): Number of base channels for the first encoder block.
        num_layers (int): Number of encoder/decoder layers in the U-Net.
        activation (str, optional): The activation function to use in the DoubleConvBlock. Default is "ReLU". Options: "ReLU" and "LeakyReLU".
        dropout_prob (float, optional): Probability of dropout to apply after the DoubleConvBlock. Default is None (no Dropout).
        device (str, optional): The device to perform calculations on. Default is DEVICE. Options: "cpu" and "cuda".

    Attributes:
        encoder_blocks (list): List of encoder blocks.
        bottleneck (DoubleConvBlock): Bottleneck layer to bridge encoder and decoder.
        decoder_blocks (list): List of decoder blocks.
        final_conv (torch.nn.Conv2d): Final convolutional layer to produce output segmentation map.

    Forward pass:
        The input tensor "x" is passed through the following sequence:
        1. Encoder blocks -> Bottleneck -> Decoder blocks -> Conv 2D

    Returns:
        torch.Tensor: The output segmentation map after processing through the U-Net.
    """
    
    def __init__(self, in_channels, out_channels, base_channels=64, num_layers=4, activation='ReLU', dropout_prob=None, device=DEVICE):
        super(UNet, self).__init__()
        self.device = device

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # Encoder blocks
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_ch
            out_ch = base_channels * (2 ** i)
            self.encoder_blocks.append(EncoderBlock(in_ch, out_ch, activation, dropout_prob))

        # Bottleneck
        self.bottleneck = DoubleConvBlock(base_channels * (2 ** (num_layers - 1)), base_channels * (2 ** num_layers), activation, dropout_prob)

        # Decoder blocks
        for i in range(num_layers):
            in_ch = base_channels * (2 ** num_layers) if i == 0 else out_ch
            out_ch = base_channels * (2 ** (num_layers - (i + 1)))
            self.decoder_blocks.append(DecoderBlock(in_ch, out_ch, activation, dropout_prob))

        # Final Conv
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder path
        for encoder in self.encoder_blocks:
            x, pooled = encoder(x)
            skip_connections.append(x)
            x = pooled

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for decoder in self.decoder_blocks:
            skip_conn = skip_connections.pop()
            x = decoder(x, skip_conn)

        # Final conv
        x = self.final_conv(x)
        return x #torch.sigmoid(x)
    
    def train_model(self,train_loader, criterion, optimizer, num_epochs=10):
        tm(self,train_loader, criterion, optimizer, num_epochs)

    def train_validate_model(self,train_loader, val_loader, criterion, optimizer, num_epochs=10):
        tvm(self,train_loader, val_loader, criterion, optimizer, num_epochs)

#-----------------------------------------------------------------------------------------------------------------------------------------------------