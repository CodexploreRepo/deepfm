"""Deep Neural Network (MLP) component for CTR models."""

from __future__ import annotations

import torch
import torch.nn as nn


class DNN(nn.Module):
    """Configurable MLP: Linear → (BatchNorm) → Activation → Dropout, stacked.

    Args:
        input_dim: Input feature dimension.
        hidden_units: List of hidden layer sizes, e.g. [256, 128, 64].
        activation: Activation function name ("relu", "leaky_relu", "gelu").
        dropout: Dropout probability.
        use_batch_norm: Whether to apply BatchNorm before activation.
    """

    ACTIVATIONS = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }

    def __init__(
        self,
        input_dim: int,
        hidden_units: list[int],
        activation: str = "relu",
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if not hidden_units:
            raise ValueError("hidden_units must be non-empty")

        act_cls = self.ACTIVATIONS.get(activation.lower())
        if act_cls is None:
            raise ValueError(
                f"Unknown activation: {activation}. Choose from {list(self.ACTIVATIONS)}"
            )

        layers: list[nn.Module] = []
        in_dim = input_dim
        for out_dim in hidden_units:
            layers.append(nn.Linear(in_dim, out_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(act_cls())
            layers.append(nn.Dropout(p=dropout))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_units[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
