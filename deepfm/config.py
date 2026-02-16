from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    dataset_name: str = "movielens-100k"
    data_dir: str = "./data/ml-100k"
    split_strategy: str = "leave_one_out"
    min_interactions: int = 3
    label_threshold: float = 4.0
    num_neg_train: int = 4
    num_neg_eval: int = 999
    auto_download: bool = True
    seed: int = 42


@dataclass
class FeatureConfig:
    embedding_l2_reg: float = 1e-5
    fm_embedding_dim: int = 16


@dataclass
class FMConfig:
    use_first_order: bool = True
    use_second_order: bool = True


@dataclass
class DNNConfig:
    hidden_units: List[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = "relu"
    dropout: float = 0.1
    use_batch_norm: bool = True
    l2_reg: float = 1e-5


@dataclass
class CINConfig:
    layer_sizes: List[int] = field(default_factory=lambda: [128, 128])
    activation: str = "relu"
    split_half: bool = True


@dataclass
class AttentionConfig:
    num_heads: int = 4
    attention_dim: int = 64
    dropout: float = 0.1
    num_layers: int = 1
    use_residual: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 50
    optimizer: str = "adam"
    scheduler: str = "reduce_on_plateau"
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 5
    early_stopping_metric: str = "auc"
    early_stopping_mode: str = "max"
    gradient_clip_norm: Optional[float] = 1.0
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ExperimentConfig:
    model_name: str = "deepfm"
    data: DataConfig = field(default_factory=DataConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    fm: FMConfig = field(default_factory=FMConfig)
    dnn: DNNConfig = field(default_factory=DNNConfig)
    cin: CINConfig = field(default_factory=CINConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "./outputs"
    seed: int = 42
    device: str = "auto"


def load_config(yaml_path: str) -> ExperimentConfig:
    """Load YAML and deserialize into ExperimentConfig using dacite."""
    import dacite
    import yaml

    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    return dacite.from_dict(data_class=ExperimentConfig, data=raw)
