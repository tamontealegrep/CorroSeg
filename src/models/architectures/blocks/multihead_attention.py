
import torch
import torch.nn as nn
from typing import Optional, Tuple

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    A module that implements multi-head attention mechanism, commonly used in transformer architectures to enhance the representation of input sequences.

    Args:
        embed_dim (int): The size of the input embedding vector.
        num_heads (int): The number of attention heads. Must divide embed_dim evenly.
        average_attention (bool, optional): If True, returns the average attention weights across all heads. Default is False.
        
    Attributes:
        embed_dim (int): The size of the embedding vector.
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each head's key, query, and value.
        average_attention (bool): If True, attention weights will be averaged across all heads when returned. If False, the attention weights for each head will be returned individually.
        W_q (torch.nn.Linear): Linear layer for query transformation.
        W_k (torch.nn.Linear): Linear layer for key transformation.
        W_v (torch.nn.Linear): Linear layer for value transformation.
        W_o (torch.nn.Linear): Linear layer for output transformation.

    Methods:
        linear_projection(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            Applies linear transformations to queries, keys, and values.
        
        split_heads(x: torch.Tensor) -> torch.Tensor:
            Reshapes the input to separate heads for multi-head attention.
        
        scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
            Computes scaled dot-product attention scores and outputs.
        
        merge_heads(x: torch.Tensor) -> torch.Tensor:
            Combines the outputs from multiple heads back into the original embedding dimension.

    Forward pass:
        The input tensors "q", "k", and "v" are passed through the following sequence:
        1. Linear projections to obtain Q, K, V.
        2. Split Q, K, V into multiple heads.
        3. Compute scaled dot-product attention.
        4. Merge the attention output back to the original embedding dimension.
        5. Apply the final output linear transformation.

    Input dimensions:
        q (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim) for queries.
        k (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim) for keys.
        v (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim) for values.
        mask (torch.Tensor, optional): A tensor of shape (batch_size, 1, seq_len, seq_len) indicating which elements to attend to (1 for valid, 0 for masked).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            torch.Tensor: The output tensor after applying multi-head attention and the final linear transformation.
            torch.Tensor: Attention weights, either averaged across heads or for each head, depending on the average_heads flag.
    """
    def __init__(self, embed_dim: int, num_heads: int, average_attention:Optional[bool]=False):
        super(MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim # Embedding vector size
        self.num_heads = num_heads # Number of heads
        self.head_dim = embed_dim // num_heads # Dimension of each head's key, query, and value
        self.average_attention = average_attention  # Whether to average attention weights

        # Define linear projections for Q, K, and V
        self.W_q = nn.Linear(embed_dim, embed_dim) #Wq - Query transformation
        self.W_k = nn.Linear(embed_dim, embed_dim) #Wk - Key transformation
        self.W_v = nn.Linear(embed_dim, embed_dim) #Wv - Value transformation

        # Define output linear projection
        self.W_o = nn.Linear(embed_dim, embed_dim) #Wo - Output transformation

    def linear_projection(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, embed_dim) @ (batch_size, embed_dim, embed_dim) --> (batch_size, seq_len, embed_dim)
        Q = self.W_q(q) 
        K = self.W_k(k)
        V = self.W_v(v)
        return Q, K, V

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, embed_dim) --> (batch_size, seq_len, num_heads, head_dim) --> (batch_size, num_heads, seq_len, head_dim)
        batch_size, seq_len, embed_dim = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len) --> (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(self.head_dim)

        # Apply mask if specified
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Multiply by values
        attention_output = torch.matmul(attention_weights, V)

        return attention_output, attention_weights 

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, num_heads, seq_len, head_dim) --> (batch_size, seq_len, num_heads, head_dim) --> (batch_size, seq_len, embed_dim)
        batch_size, num_heads, seq_len, head_dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of the multi-head attention mechanism.

        Input dimensions:
            q (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim) for queries.
            k (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim) for keys.
            v (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim) for values.
            mask (torch.Tensor, optional): A tensor of shape (batch_size, 1, seq_len, seq_len) indicating which elements to attend to.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                torch.Tensor: The output tensor of shape (batch_size, seq_len, embed_dim).
                torch.Tensor: Attention weights of shape (batch_size, seq_len, seq_len) if average_attention is True, otherwise (batch_size, num_heads, seq_len, seq_len).
        """
        # Apply linear projections
        Q, K, V = self.linear_projection(q, k, v)

        # Reshape the input to have num_heads for multi-head attention
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Perform scaled dot-product attention
        attention_output, attention_weights  = self.scaled_dot_product_attention(Q, K, V, mask)

        # Reshape the input back to original shape of embed_dim
        attention_output = self.merge_heads(attention_output)

        # Apply output linear projection
        attention_output = self.W_o(attention_output)

        # Average the attention weights across heads
        if self.average_attention:
            # (batch_size, num_heads, seq_len, seq_len) --> (batch_size, seq_len, seq_len)
            attention_weights = attention_weights.mean(dim=1)  

        return attention_output, attention_weights 

#-----------------------------------------------------------------------------------------------------------------------------------------------------