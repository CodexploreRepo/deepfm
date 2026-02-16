import logging
import os

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(model, optimizer, epoch, metrics, output_dir):
    """Save model and optimizer state to output_dir/best_model.pt."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "best_model.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    logger.info(f"Saved checkpoint to {path} (epoch {epoch})")


def load_checkpoint(model, path_or_dir, optimizer=None):
    """Load model state from a checkpoint file or directory."""
    if os.path.isdir(path_or_dir):
        path = os.path.join(path_or_dir, "best_model.pt")
    else:
        path = path_or_dir
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info(f"Loaded checkpoint from {path} (epoch {ckpt['epoch']})")
    return ckpt
