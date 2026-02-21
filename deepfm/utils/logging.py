"""Logging configuration."""

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Return a configured logger with stdout and optional file handler.

    Child loggers (dotted names, e.g. "deepfm.trainer") automatically
    propagate to their parent. If the parent is already configured, no
    StreamHandler is added to the child to avoid duplicate stdout output.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Skip StreamHandler on child loggers whose parent is already configured —
    # propagation will route their output through the parent's handlers.
    parent_name = name.rsplit(".", 1)[0] if "." in name else None
    parent_configured = bool(parent_name and logging.getLogger(parent_name).handlers)

    if not parent_configured:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
