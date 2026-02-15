"""Logging configuration for training and other entrypoint scripts."""

import logging
from pathlib import Path


def setup_logging(exp_dir: Path) -> None:
    """Configure logging to console and to a file in the experiment logs directory."""
    log_file = exp_dir / "logs" / "training.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    format_str = "%(asctime)s | %(levelname)s | %(message)s"
    formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Avoid adding duplicate handlers if main() is called more than once (e.g. in tests)
    if not root.handlers:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
