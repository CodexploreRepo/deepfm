"""Command-line interface for DeepFM training framework.

Usage:
    python -m deepfm train --config configs/deepfm_movielens.yaml
    python -m deepfm evaluate --config configs/deepfm_movielens.yaml --checkpoint outputs/best_model.pt
    python -m deepfm train --config configs/deepfm_movielens.yaml --override training.batch_size=2048
"""

from __future__ import annotations

import argparse
import logging

import torch

from deepfm.config import ExperimentConfig, load_config
from deepfm.data.movielens import MovieLensAdapter
from deepfm.models import build_model
from deepfm.training.trainer import Trainer
from deepfm.utils.logging import setup_logging
from deepfm.utils.seeding import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="DeepFM Training Framework")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=str, required=True)
    train_parser.add_argument("--override", nargs="*", default=[])

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--config", type=str, required=True)
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument("--override", nargs="*", default=[])

    return parser.parse_args()


def apply_overrides(config: ExperimentConfig, overrides: list):
    """Apply dot-notation overrides like 'training.batch_size=2048'."""
    for override in overrides:
        key, value = override.split("=", 1)
        parts = key.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)

        attr = parts[-1]
        current = getattr(obj, attr)

        # Cast to the type of the current value
        if isinstance(current, bool):
            value = value.lower() in ("true", "1", "yes")
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        elif isinstance(current, list):
            import json

            value = json.loads(value)

        setattr(obj, attr, value)


def resolve_device(device_str: str) -> str:
    """Resolve 'auto' to the best available device (MPS > CPU)."""
    if device_str == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_str


def _build_data(config: ExperimentConfig):
    """Build datasets from config."""
    adapter = MovieLensAdapter(
        data_dir=config.data.data_dir,
        label_threshold=config.data.label_threshold,
    )
    train_ds, val_ds, test_ds = adapter.build_datasets(
        min_interactions=config.data.min_interactions,
        num_neg_train=config.data.num_neg_train,
        num_neg_eval=config.data.num_neg_eval,
        auto_download=config.data.auto_download,
    )
    return adapter.schema, train_ds, val_ds, test_ds


def main():
    args = parse_args()

    if args.command is None:
        print("Usage: python -m deepfm {train,evaluate} --config CONFIG")
        return

    config = load_config(args.config)
    if args.override:
        apply_overrides(config, args.override)

    setup_logging(config.output_dir)
    seed_everything(config.seed)

    device = resolve_device(config.device)
    logging.info(f"Using device: {device}")

    # Build data
    schema, train_ds, val_ds, test_ds = _build_data(config)

    # Build model
    model = build_model(config.model_name, schema, config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model: {config.model_name}, Parameters: {num_params:,}")

    if args.command == "train":
        trainer = Trainer(
            model=model,
            config=config,
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            device=device,
        )
        metrics = trainer.fit()
        logging.info(f"Final metrics: {metrics}")

    elif args.command == "evaluate":
        from deepfm.utils.io import load_checkpoint

        load_checkpoint(model, args.checkpoint)
        model.to(device)
        trainer = Trainer(
            model=model,
            config=config,
            train_dataset=train_ds,
            val_dataset=val_ds,
            device=device,
        )
        metrics = trainer.evaluate(test_ds)
        logging.info(f"Test metrics: {metrics}")


if __name__ == "__main__":
    main()
