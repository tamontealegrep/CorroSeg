
import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.models.architectures.blocks.multihead_attention import MultiHeadAttention

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class FeedForward(nn.Module):
    """
    A feed-forward neural network block for Transformer models.

    This block consists of two linear transformations with a ReLU activation in between.

    Args:
        embed_dim (int): The dimension of the input embeddings.
        ff_dim (int): The dimension of the hidden layer in the feed-forward network.
        activation (str, optional): The activation function to use. Default is "ReLU". Options: "ReLU", "LeakyReLU", and "GeLU".
    
    Attributes:
        linear1 (nn.Linear): The first linear layer.
        activation (nn.Module): The activation function.
        linear2 (nn.Linear): The second linear layer.

    Forward pass:
        The input tensor "x" is passed through the following sequence:
        1. Linear transformation (linear1) -> Activation -> Linear transformation (linear2)

    Returns:
        torch.Tensor: A tensor of shape (B, N, embed_dim) containing the output of the feed-forward network.
    """

    def __init__(self, embed_dim: int, ff_dim: int, activation: Optional[str] = 'ReLU'):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)

        # Set the activation function
        if activation == 'ReLU':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation == 'GeLU':
            self.activation = nn.GELU()
        else:
            raise ValueError("Unsupported activation type. Choose 'ReLU', 'LeakyReLU', or 'GeLU'.")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(x)))
    
class ViTLayer(nn.Module):
    """
    A single layer of a Vision Transformer model.

    This layer consists of multi-head self-attention followed by a feed-forward network,
    with residual connections and layer normalization.

    Args:
        embed_dim (int): The dimension of the input embeddings.
        num_heads (int): The number of attention heads.
        ff_dim (int): The dimension of the feed-forward network.
        dropout_prob (float, optional): Probability of dropout. If set, a Dropout layer will be applied after attention layer and feed-forward. Default is None (no Dropout).
        activation (str, optional): The activation function to use in the feed-forward. Default is "ReLU". Options: "ReLU", "LeakyReLU", and "GeLU".

    Attributes:
        attention (nn.MultiheadAttention): The multi-head attention layer.
        feed_forward (nn.Sequential): The feed-forward network.
        layer_norm1 (nn.LayerNorm): The first layer normalization.
        layer_norm2 (nn.LayerNorm): The second layer normalization.
        dropout (nn.Dropout): Dropout layer after the attention and feed-forward network.

    Forward pass:
        The input tensor "x" is passed through the following sequence:
        1. Layer normalization (layer_norm1)
        2. Multi-head self-attention layer
        3. (Optional) Dropout
        4. Residual connection
        5. Layer normalization (layer_norm2)
        6. Feed-forward network
        7. (Optional) Dropout
        8. Residual connection

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            torch.Tensor: Output tensor of shape (B, N, embed_dim).
            torch.Tensor: Attention weights, averaged across heads.
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ff_dim: int,
                 dropout_prob: Optional[float] = None,
                 activation: Optional[str] = 'ReLU'):
        super(ViTLayer, self).__init__()
        
        self.attention = MultiHeadAttention(embed_dim, num_heads, True) #nn.MultiheadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, ff_dim, activation)
        
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob is not None else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Layer normalization before attention
        x_norm = self.layer_norm1(x)

        # Multi-head self-attention
        attn_output, attn_weights = self.attention(x_norm, x_norm, x_norm, mask)

        if self.dropout is not None:
            attn_output = self.dropout(attn_output)

        # Residual connection
        x = x + attn_output  

        # Layer normalization after attention
        x_norm = self.layer_norm2(x)

        # Feed-forward network
        ff_output = self.feed_forward(x_norm)

        if self.dropout is not None:
            ff_output = self.dropout(ff_output)

        # Residual connection
        x = x + ff_output

        return x, attn_weights

class ViTEncoder(nn.Module):
    """
    A block of Transformer layers.

    This block consists of multiple ViTLayer stacked together.

    Args:
        num_layers (int): The number of ViTLayer to stack.
        num_heads (int): The number of attention heads.
        embed_dim (int): The dimension of the input embeddings.
        ff_dim (int): The dimension of the feed-forward network.
        dropout_prob (float, optional): Probability of dropout. Default is None (no Dropout).
        activation (str, optional): The activation function to use in the feed-forward. Default is "ReLU". Options: "ReLU", "LeakyReLU", and "GeLU".

    Attributes:
        layers (List[ViTLayer]): The list of ViTLayer instances.

    Forward pass:
        The input tensor "x" is passed through each ViTLayer in sequence.

    Returns:
        torch.Tensor: Output tensor of shape (B, N, embed_dim).
        torch.Tensor: Attention weights from the last layer.
    """

    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 embed_dim: int,
                 ff_dim: int,
                 dropout_prob: Optional[float] = None,
                 activation: Optional[str] = 'ReLU'):
        super(ViTEncoder, self).__init__()
        self.layers = nn.ModuleList([
            ViTLayer(embed_dim, num_heads, ff_dim, dropout_prob, activation)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_weights = None
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
        return x, attn_weights
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------
