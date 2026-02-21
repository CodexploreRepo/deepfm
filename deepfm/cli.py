"""CLI entry point for DeepFM training and evaluation."""

from __future__ import annotations

import argparse

import torch

from deepfm.config import ExperimentConfig, load_config
from deepfm.data.movielens import MovieLensAdapter
from deepfm.models import create_model
from deepfm.training.trainer import Trainer
from deepfm.utils import get_logger, seed_everything


def resolve_device(config_device: str) -> str:
    """Resolve device string: 'auto' → MPS if available, else CPU.

    No CUDA — this project targets Apple M2.
    """
    if config_device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return config_device


def _build_adapter(config: ExperimentConfig) -> MovieLensAdapter:
    """Build the appropriate dataset adapter based on config."""
    if config.data.dataset_name == "movielens":
        return MovieLensAdapter(config.data)
    raise ValueError(f"Unknown dataset: {config.data.dataset_name}")


def train_command(config: ExperimentConfig) -> None:
    """Execute the training pipeline."""
    logger = get_logger("deepfm", log_file=f"{config.output_dir}/train.log")

    seed_everything(config.seed)
    device = resolve_device(config.device)
    logger.info(f"Device: {device}")

    # Build data
    logger.info("Loading and preparing data...")
    adapter = _build_adapter(config)
    schema, train_ds, val_ds, test_ds = adapter.build()
    logger.info(
        f"Data ready: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )
    logger.info(f"Schema: {list(schema.fields.keys())}")

    # Build model
    model = create_model(config.model_name, schema, config)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {config.model_name} ({total_params:,} parameters)")

    # Train
    trainer = Trainer(
        model=model,
        schema=schema,
        config=config,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        adapter=adapter,
        device=device,
    )
    trainer.train()


def evaluate_command(config: ExperimentConfig) -> None:
    """Evaluate a saved model checkpoint."""
    logger = get_logger("deepfm")

    seed_everything(config.seed)
    device = resolve_device(config.device)

    # Build data
    adapter = _build_adapter(config)
    schema, _, val_ds, test_ds = adapter.build()

    # Build model and load checkpoint
    model = create_model(config.model_name, schema, config)
    checkpoint_path = f"{config.output_dir}/best_model.pt"
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(
        f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})"
    )

    # Evaluate
    trainer = Trainer(
        model=model,
        schema=schema,
        config=config,
        train_ds=val_ds,  # dummy — not used for eval
        val_ds=val_ds,
        test_ds=test_ds,
        device=device,
    )

    logger.info("--- Validation ---")
    val_metrics = trainer.evaluate(val_ds, "val")
    for k, v in val_metrics.items():
        logger.info(f"  val_{k} = {v:.4f}")

    logger.info("--- Test ---")
    test_metrics = trainer.evaluate(test_ds, "test")
    for k, v in test_metrics.items():
        logger.info(f"  test_{k} = {v:.4f}")


def _print_comparison_table(runs: list[dict]) -> None:
    """Print an aligned side-by-side metric table for a list of run dicts."""
    # Column widths
    W_RUN = 28
    W_MODEL = 20
    W_HPARAM = 20
    W_METRIC = 10

    header = (
        "Run".ljust(W_RUN)
        + "Model".ljust(W_MODEL)
        + "LR·BS·Emb".ljust(W_HPARAM)
        + "Val AUC".rjust(W_METRIC)
        + "Val LogL".rjust(W_METRIC)
        + "Tst AUC".rjust(W_METRIC)
        + "Tst LogL".rjust(W_METRIC)
        + "HR@10".rjust(W_METRIC)
        + "NDCG@10".rjust(W_METRIC)
        + "BstEp".rjust(W_METRIC)
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for run in runs:
        cfg = run.get("config", {})
        training_cfg = cfg.get("training", {})
        feature_cfg = cfg.get("feature", {})

        run_id = run.get("run_id", "?")
        model = cfg.get("model_name", "?")
        lr = training_cfg.get("lr", "?")
        bs = training_cfg.get("batch_size", "?")
        emb = feature_cfg.get("fm_embed_dim", "?")
        hparam = f"{lr}·{bs}·{emb}"

        vm = run.get("val_metrics", {})
        tm = run.get("test_metrics", {})
        ti = run.get("training_info", {})

        def _fmt(d: dict, key: str) -> str:
            v = d.get(key)
            return f"{v:.4f}" if isinstance(v, float) else "-"

        row = (
            str(run_id)[:W_RUN].ljust(W_RUN)
            + str(model)[:W_MODEL].ljust(W_MODEL)
            + str(hparam)[:W_HPARAM].ljust(W_HPARAM)
            + _fmt(vm, "auc").rjust(W_METRIC)
            + _fmt(vm, "logloss").rjust(W_METRIC)
            + _fmt(tm, "auc").rjust(W_METRIC)
            + _fmt(tm, "logloss").rjust(W_METRIC)
            + _fmt(tm, "HR@10").rjust(W_METRIC)
            + _fmt(tm, "NDCG@10").rjust(W_METRIC)
            + str(ti.get("best_epoch", "-")).rjust(W_METRIC)
        )
        print(row)

    print(sep)


def compare_command(args) -> None:
    """Print a comparison table of all results.json files under --dir."""
    import json
    from pathlib import Path

    base = Path(args.dir)
    files = sorted(base.rglob("results.json"))
    if not files:
        print(f"No results.json files found under {base}")
        return

    runs = [json.loads(f.read_text()) for f in files]
    _print_comparison_table(runs)


def main() -> None:
    """Parse arguments and dispatch to train/evaluate/compare."""
    parser = argparse.ArgumentParser(
        prog="deepfm",
        description="DeepFM: CTR prediction with FM, xDeepFM, and AttentionDeepFM",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--config", required=True, help="Path to YAML config"
    )
    train_parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config values, e.g. training.num_epochs=10",
    )

    # Evaluate subcommand
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a saved model"
    )
    eval_parser.add_argument(
        "--config", required=True, help="Path to YAML config"
    )
    eval_parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config values",
    )

    # Compare subcommand
    cmp_parser = subparsers.add_parser(
        "compare", help="Compare experiment results"
    )
    cmp_parser.add_argument(
        "--dir",
        default="outputs",
        help="Directory to scan for results.json files",
    )

    args = parser.parse_args()

    if args.command == "compare":
        compare_command(args)
        return

    config = load_config(args.config, args.override or None)

    if args.command == "train":
        train_command(config)
    elif args.command == "evaluate":
        evaluate_command(config)
