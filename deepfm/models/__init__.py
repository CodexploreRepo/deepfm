"""CTR prediction models with factory registry."""

from __future__ import annotations

from deepfm.config import ExperimentConfig
from deepfm.data.schema import DatasetSchema
from deepfm.models.attention_deepfm import AttentionDeepFM
from deepfm.models.base import BaseCTRModel
from deepfm.models.deepfm import DeepFM
from deepfm.models.xdeepfm import xDeepFM

MODEL_REGISTRY: dict[str, type[BaseCTRModel]] = {
    "deepfm": DeepFM,
    "xdeepfm": xDeepFM,
    "attention_deepfm": AttentionDeepFM,
}


def create_model(
    name: str, schema: DatasetSchema, config: ExperimentConfig
) -> BaseCTRModel:
    """Create a model by name from the registry.

    Args:
        name: Model name (deepfm, xdeepfm, attention_deepfm).
        schema: Dataset schema describing input features.
        config: Full experiment configuration.

    Returns:
        Instantiated model ready for training.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](schema, config)


__all__ = [
    "AttentionDeepFM",
    "BaseCTRModel",
    "DeepFM",
    "MODEL_REGISTRY",
    "create_model",
    "xDeepFM",
]
