"""
Config and path validation for training and related entrypoints.

Use this module to run pre-flight checks before loading configs or data,
e.g. required files exist, and semantic rules (e.g. grayscale + single-channel
normalization) are satisfied.
"""

from pathlib import Path
from typing import Union


def validate_training_paths(
    config_path: Union[str, Path],
    metadata_path: Union[str, Path],
) -> None:
    """
    Validate that required training paths exist.

    Raises:
        FileNotFoundError: If config_path or metadata_path does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
