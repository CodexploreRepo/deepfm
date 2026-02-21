"""Compressed Interaction Network (CIN) layer for xDeepFM."""

from __future__ import annotations

import torch
import torch.nn as nn


class CIN(nn.Module):
    """CIN: explicit vector-wise higher-order feature interactions.

    Each CIN layer computes an outer product between the previous hidden state
    and the original input, then compresses with a Conv1d (kernel_size=1).
    When ``split_half=True``, half of each layer's output feeds into the next
    layer while the other half goes directly to the output — reducing parameters
    and enabling multi-granularity interactions.

    Args:
        num_fields: Number of input fields (F).
        embed_dim: Embedding dimension per field (D).
        layer_sizes: Number of feature maps per CIN layer, e.g. [128, 128].
        split_half: If True, split each layer's output in half — one half
            feeds forward, the other half goes to the output pool.
    """

    def __init__(
        self,
        num_fields: int,
        embed_dim: int,
        layer_sizes: list[int] | None = None,
        split_half: bool = True,
    ) -> None:
        super().__init__()
        layer_sizes = layer_sizes or [128, 128]
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.split_half = split_half

        self.conv_layers = nn.ModuleList()
        prev_num_maps = num_fields
        self.direct_sizes: list[int] = []  # sizes that go to output pool
        self.next_sizes: list[int] = []  # sizes that feed into next layer

        for i, layer_size in enumerate(layer_sizes):
            # Conv1d compresses the outer product result
            # Input channels = prev_num_maps * num_fields (from outer product)
            self.conv_layers.append(
                nn.Conv1d(prev_num_maps * num_fields, layer_size, kernel_size=1)
            )

            if split_half and i < len(layer_sizes) - 1:
                # Split: half to output, half to next layer
                direct = layer_size // 2
                next_size = layer_size - direct
                self.direct_sizes.append(direct)
                self.next_sizes.append(next_size)
                prev_num_maps = next_size
            else:
                # Last layer or no split: all go to output
                self.direct_sizes.append(layer_size)
                self.next_sizes.append(layer_size)
                prev_num_maps = layer_size

        self.output_dim = sum(self.direct_sizes)

    def forward(self, field_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            field_embeddings: (B, F, D) — field embeddings from FeatureEmbedding.

        Returns:
            (B, output_dim) — sum-pooled and concatenated across CIN layers.
        """
        batch_size = field_embeddings.size(0)
        x0 = field_embeddings  # (B, F, D)
        hidden = x0  # (B, H_k, D), starts as (B, F, D)

        output_parts: list[torch.Tensor] = []

        for i, conv in enumerate(self.conv_layers):
            # Outer product: (B, H_k, D) x (B, F, D) → (B, H_k * F, D)
            # Using einsum: z_{h,f,d} = hidden_{h,d} * x0_{f,d}
            outer = torch.einsum("bhd,bfd->bhfd", hidden, x0)  # (B, H_k, F, D)
            outer = outer.reshape(
                batch_size, -1, self.embed_dim
            )  # (B, H_k*F, D)

            # Compress with Conv1d: (B, H_k*F, D) → (B, layer_size, D)
            compressed = conv(outer)  # (B, layer_size, D)
            compressed = torch.relu(compressed)

            if self.split_half and i < len(self.conv_layers) - 1:
                direct, hidden = compressed.split(
                    [self.direct_sizes[i], self.next_sizes[i]], dim=1
                )
            else:
                direct = compressed
                hidden = compressed

            # Sum-pool over D: (B, num_maps, D) → (B, num_maps)
            output_parts.append(direct.sum(dim=2))

        # Concatenate across layers: (B, output_dim)
        return torch.cat(output_parts, dim=1)
