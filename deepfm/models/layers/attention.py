import math

import torch
import torch.nn as nn


class FieldAttention(nn.Module):
    """Multi-head self-attention over field embeddings.

    Each field embedding attends to all other field embeddings,
    learning which feature interactions are most important.

    Args:
        embedding_dim: Dimension of input field embeddings.
        num_heads: Number of attention heads.
        attention_dim: Total dimension of Q/K/V (split across heads).
        dropout: Attention dropout rate.
        use_residual: If True, add residual connection + LayerNorm.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 4,
        attention_dim: int = 64,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()
        assert attention_dim % num_heads == 0, (
            f"attention_dim ({attention_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.d_k = attention_dim // num_heads
        self.num_heads = num_heads

        self.W_Q = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.W_K = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.W_V = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.W_O = nn.Linear(attention_dim, embedding_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.use_residual = use_residual

    def forward(self, field_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            field_embeddings: (B, num_fields, embedding_dim)

        Returns:
            (B, num_fields, embedding_dim)
        """
        B, F, D = field_embeddings.shape

        Q = self.W_Q(field_embeddings).view(B, F, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(field_embeddings).view(B, F, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(field_embeddings).view(B, F, self.num_heads, self.d_k).transpose(1, 2)
        # Q, K, V: (B, H, F, d_k)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = self.dropout(torch.softmax(attn_scores, dim=-1))
        context = torch.matmul(attn_weights, V)  # (B, H, F, d_k)

        context = context.transpose(1, 2).contiguous().view(B, F, -1)  # (B, F, attention_dim)
        output = self.W_O(context)  # (B, F, D)

        if self.use_residual:
            output = self.layer_norm(output + field_embeddings)

        return output
