"""Multi-head self-attention layer for AttentionDeepFM."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """Stacked multi-head self-attention over field embeddings.

    Applies N layers of multi-head attention with optional residual connections
    and LayerNorm. Learns pairwise field importance weights so the model can
    focus on the most relevant feature interactions.

    Args:
        embed_dim: Input/output embedding dimension per field (D).
        num_heads: Number of attention heads.
        attention_dim: Total dimension across all heads for Q/K/V projections.
        num_layers: Number of stacked attention layers.
        use_residual: Whether to use residual connections + LayerNorm.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        attention_dim: int = 64,
        num_layers: int = 1,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.head_dim = attention_dim // num_heads
        self.use_residual = use_residual

        if attention_dim % num_heads != 0:
            raise ValueError(
                f"attention_dim ({attention_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                _AttentionBlock(embed_dim, num_heads, attention_dim, use_residual)
            )

    def forward(self, field_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            field_embeddings: (B, F, D) — field embeddings.

        Returns:
            (B, F, D) — attention-refined field embeddings.
        """
        x = field_embeddings
        for layer in self.layers:
            x = layer(x)
        return x


class _AttentionBlock(nn.Module):
    """Single multi-head self-attention block with optional residual + LayerNorm."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_dim: int,
        use_residual: bool,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.use_residual = use_residual

        self.W_q = nn.Linear(embed_dim, attention_dim)
        self.W_k = nn.Linear(embed_dim, attention_dim)
        self.W_v = nn.Linear(embed_dim, attention_dim)
        self.W_out = nn.Linear(attention_dim, embed_dim)

        if use_residual:
            self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, F, D = x.shape

        # Project to Q, K, V: (B, F, attention_dim)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape to multi-head: (B, num_heads, F, head_dim)
        Q = Q.view(B, F, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, F, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, F, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: (B, num_heads, F, F)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values: (B, num_heads, F, head_dim)
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads: (B, F, attention_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, F, -1)

        # Output projection: (B, F, D)
        output = self.W_out(attn_output)

        if self.use_residual:
            output = self.layer_norm(output + x)

        return output
