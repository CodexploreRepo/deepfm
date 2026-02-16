from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepfm.config import ExperimentConfig
    from deepfm.data.schema import DatasetSchema
    from deepfm.models.base import BaseCTRModel


MODEL_REGISTRY = {}


def _register_models():
    """Lazy import to avoid circular dependencies."""
    global MODEL_REGISTRY
    if MODEL_REGISTRY:
        return

    from deepfm.models.attention_deepfm import AttentionDeepFM
    from deepfm.models.deepfm import DeepFM
    from deepfm.models.xdeepfm import xDeepFM

    MODEL_REGISTRY.update(
        {
            "deepfm": DeepFM,
            "xdeepfm": xDeepFM,
            "attention_deepfm": AttentionDeepFM,
        }
    )


def build_model(model_name: str, schema: DatasetSchema, config: ExperimentConfig) -> BaseCTRModel:
    """Factory function for model construction."""
    _register_models()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](schema=schema, config=config)
