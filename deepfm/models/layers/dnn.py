from typing import List

import torch
import torch.nn as nn


class DNN(nn.Module):
    """Multi-layer perceptron with configurable depth, width, activation,
    batch normalization, and dropout.

    Architecture per layer: Linear -> (BatchNorm) -> Activation -> Dropout
    """

    ACTIVATIONS = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "prelu": nn.PReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
    }

    def __init__(
        self,
        input_dim: int,
        hidden_units: List[int],
        activation: str = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for i, units in enumerate(hidden_units):
            layers.append(nn.Linear(prev_dim, units))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(units))
            is_last = i == len(hidden_units) - 1
            if not is_last:
                layers.append(self.ACTIVATIONS[activation]())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            prev_dim = units

        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_units[-1] if hidden_units else input_dim
        self._init_weights(activation)

    def _init_weights(self, activation: str):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                if activation in ("relu", "leaky_relu", "elu", "gelu"):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
