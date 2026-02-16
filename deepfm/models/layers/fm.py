import torch
import torch.nn as nn


class FMInteraction(nn.Module):
    """Factorization Machine second-order interaction layer.

    Efficient O(n*d) computation using the identity:
        sum_{i<j} <v_i, v_j> = 0.5 * (||sum(v_i)||^2 - sum(||v_i||^2))

    Args:
        reduce_sum: If True, sum over embedding dim to produce (B, 1).
                    If False, return (B, embedding_dim).
    """

    def __init__(self, reduce_sum: bool = True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, field_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            field_embeddings: (B, num_fields, embedding_dim)

        Returns:
            (B, 1) if reduce_sum else (B, embedding_dim)
        """
        square_of_sum = field_embeddings.sum(dim=1).pow(2)  # (B, D)
        sum_of_squares = field_embeddings.pow(2).sum(dim=1)  # (B, D)
        interaction = 0.5 * (square_of_sum - sum_of_squares)  # (B, D)

        if self.reduce_sum:
            return interaction.sum(dim=-1, keepdim=True)  # (B, 1)
        return interaction
