import logging
import os
import sys


def setup_logging(output_dir: str, level: int = logging.INFO):
    """Configure logging to stdout and a file in output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(output_dir, "train.log")),
    ]

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
