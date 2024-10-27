
import torch
import torch.nn as nn
from typing import Optional

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class FeedForward(nn.Module):
    """
    A feed-forward neural network block for Transformer models.

    This block consists of two linear transformations with a ReLU activation in between.

    Args:
        embed_dim (int): The dimension of the input embeddings.
        ff_dim (int): The dimension of the hidden layer in the feed-forward network.
        activation (str, optional): The activation function to use. Default is "ReLU". Options: "ReLU" and "LeakyReLU".

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
        else:
            raise ValueError("Unsupported activation type. Choose 'ReLU' or 'LeakyReLU'.")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(x)))
    
class TransformerBlock(nn.Module):
    """
    A single block of a Transformer model.

    This block consists of multi-head self-attention followed by a feed-forward network,
    with residual connections and layer normalization.

    Args:
        embed_dim (int): The dimension of the input embeddings.
        num_heads (int): The number of attention heads.
        ff_dim (int): The dimension of the feed-forward network.
        dropout_prob (float, optional): Probability of dropout. If set, a Dropout layer will be applied after attention layer and feed-forward. Default is None (no Dropout).
        activation (str, optional): The activation function to use in the feed-forward. Default is "ReLU". Options: "ReLU" and "LeakyReLU".

    Attributes:
        attention (nn.MultiheadAttention): The multi-head attention layer.
        feed_forward (nn.Sequential): The feed-forward network.
        layer_norm1 (nn.LayerNorm): The first layer normalization.
        layer_norm2 (nn.LayerNorm): The second layer normalization.
        dropout (nn.Dropout): Dropout layer after the attention and feed-forward network.

    Forward pass:
        The input tensor "x" is passed through the following sequence:
        1. Multi-head self-attention layer
        2. (Optional) Dropout
        3. Residual connection and layer normalization
        4. Feed-forward network
        5. (Optional) Dropout
        6. Residual connection and layer normalization

    Returns:
        torch.Tensor: Output tensor of shape (B, N, embed_dim). 
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ff_dim: int,
                 dropout_prob: Optional[float] = None,
                 activation: Optional[str] = 'ReLU'):
        super(TransformerBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, ff_dim, activation)
        
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob is not None else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Multi-head self-attention
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)

        if self.dropout is not None:
            attn_output = self.dropout(attn_output)

        x = self.layer_norm1(x + attn_output)  

        # Feed-forward network
        ff_output = self.feed_forward(x)

        if self.dropout is not None:
            ff_output = self.dropout(ff_output)

        x = self.layer_norm2(x + ff_output)

        return x
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------
