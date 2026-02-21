"""Factorization Machine second-order interaction layer."""

from __future__ import annotations

import torch
import torch.nn as nn


class FMInteraction(nn.Module):
    """Efficient FM second-order interaction: O(n*d) via sum-of-squares trick.

    Input:  field_embeddings (B, F, D)
    Output: interaction       (B, 1)

    Formula: 0.5 * sum_d( (sum_f e_f,d)^2 - sum_f (e_f,d)^2 )
    """

    def forward(self, field_embeddings: torch.Tensor) -> torch.Tensor:
        # field_embeddings: (B, F, D)
        square_of_sum = field_embeddings.sum(dim=1).pow(2)  # (B, D)
        sum_of_squares = field_embeddings.pow(2).sum(dim=1)  # (B, D)
        interaction = 0.5 * (square_of_sum - sum_of_squares).sum(dim=1, keepdim=True)  # (B, 1)
        return interaction
