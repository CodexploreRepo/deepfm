"""Dataclass-based configuration with YAML loading via dacite."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# This function simplifies the creation of instances of Python dataclasses from dictionaries, which is especially useful when loading configurations from YAML files. It automatically maps the dictionary keys to the corresponding dataclass fields, handling nested structures as well. This allows for a clean and structured way to manage configuration settings in a Python application.ååååååå
from dacite import from_dict


@dataclass
class DataConfig:
    dataset_name: str = "movielens"
    data_dir: str = "/Users/codexplore/Developer/repos/deepfm/data/ml-100k"
    split_strategy: str = "leave_one_out"
    min_interactions: int = 3
    label_threshold: float = 4.0
    num_neg_train: int = 4
    num_neg_eval: int = 999


@dataclass
class FeatureConfig:
    fm_embed_dim: int = 16
    embedding_l2_reg: float = 1e-5


@dataclass
class FMConfig:
    use_first_order: bool = True
    use_second_order: bool = True


@dataclass
class DNNConfig:
    hidden_units: list[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = "relu"
    dropout: float = 0.1
    use_batch_norm: bool = True


@dataclass
class CINConfig:
    layer_sizes: list[int] = field(default_factory=lambda: [128, 128])
    split_half: bool = True


@dataclass
class AttentionConfig:
    num_heads: int = 4
    attention_dim: int = 64
    num_layers: int = 1
    use_residual: bool = True


@dataclass
class TrainingConfig:
    num_epochs: int = 50
    batch_size: int = 4096
    lr: float = 1e-3
    optimizer: str = "adam"
    scheduler: str = "reduce_on_plateau"
    early_stopping_patience: int = 5
    metric: str = "auc"
    gradient_clip_norm: float = 1.0


@dataclass
class ExperimentConfig:
    model_name: str = "deepfm"
    seed: int = 42
    device: str = "auto"
    output_dir: str = "outputs"
    data: DataConfig = field(default_factory=DataConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    fm: FMConfig = field(default_factory=FMConfig)
    dnn: DNNConfig = field(default_factory=DNNConfig)
    cin: CINConfig = field(default_factory=CINConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(
    yaml_path: str | Path, overrides: list[str] | None = None
) -> ExperimentConfig:
    """Load config from YAML file with optional dot-notation overrides.

    Args:
        yaml_path: Path to YAML config file.
        overrides: List of "key.subkey=value" strings, e.g. ["training.batch_size=2048"].
    """
    with open(yaml_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    if overrides:
        for override in overrides:
            key, value = override.split("=", 1)
            parts = key.strip().split(".")
            target = raw
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = _parse_value(value.strip())

    return from_dict(data_class=ExperimentConfig, data=raw)


def _parse_value(value: str) -> Any:
    """Parse a string value into the appropriate Python type."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        import ast
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    return value
