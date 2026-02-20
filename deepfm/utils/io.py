"""Checkpoint save/load utilities."""

from pathlib import Path

import torch


def save_checkpoint(state: dict, path: str | Path) -> None:
    """Save a checkpoint dict to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict:
    """Load a checkpoint dict from disk."""
    return torch.load(path, map_location=device, weights_only=False)
