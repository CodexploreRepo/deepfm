from typing import List

import torch
import torch.nn as nn


class CIN(nn.Module):
    """Compressed Interaction Network for xDeepFM.

    Captures arbitrary-order explicit feature interactions at the vector-wise level.
    Uses Conv1d with kernel_size=1 for efficient compression of outer products.

    Args:
        num_fields: Number of input feature fields (m).
        embedding_dim: Dimension of each field embedding (D).
        layer_sizes: Number of feature maps per CIN layer.
        activation: Activation function name.
        split_half: If True, first half of each layer feeds the next layer,
                    second half goes to output. Last layer always goes to output.
    """

    def __init__(
        self,
        num_fields: int,
        embedding_dim: int,
        layer_sizes: List[int],
        activation: str = "relu",
        split_half: bool = True,
    ):
        super().__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.layer_sizes = layer_sizes
        self.split_half = split_half

        self.conv_layers = nn.ModuleList()
        prev_size = num_fields
        for i, size in enumerate(layer_sizes):
            self.conv_layers.append(
                nn.Conv1d(prev_size * num_fields, size, kernel_size=1)
            )
            if split_half and i < len(layer_sizes) - 1:
                prev_size = size // 2
            else:
                prev_size = size

        self.activation = nn.ReLU() if activation == "relu" else nn.Identity()

        # Compute output dimension
        if split_half:
            self.output_dim = sum(s // 2 for s in layer_sizes[:-1]) + layer_sizes[-1]
        else:
            self.output_dim = sum(layer_sizes)

    def forward(self, field_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            field_embeddings: (B, num_fields, embedding_dim)

        Returns:
            (B, output_dim)
        """
        B = field_embeddings.size(0)
        x0 = field_embeddings  # (B, m, D) — kept fixed
        x_prev = field_embeddings  # (B, H_{k-1}, D) — evolves

        output_layers = []
        for i, conv in enumerate(self.conv_layers):
            # Outer product: (B, H_{k-1}, 1, D) * (B, 1, m, D) -> (B, H_{k-1}, m, D)
            outer = torch.einsum("bhd,bmd->bhmd", x_prev, x0)
            # Flatten: (B, H_{k-1}*m, D)
            outer = outer.reshape(B, -1, self.embedding_dim)
            # Compress via Conv1d: (B, H_k, D)
            x_k = self.activation(conv(outer))

            if self.split_half and i < len(self.conv_layers) - 1:
                split_point = x_k.size(1) // 2
                x_prev = x_k[:, :split_point, :]
                output_layers.append(x_k[:, split_point:, :])
            else:
                x_prev = x_k
                output_layers.append(x_k)

        # Sum pooling over embedding dim D, then concatenate
        pooled = [layer.sum(dim=-1) for layer in output_layers]
        return torch.cat(pooled, dim=-1)  # (B, output_dim)
