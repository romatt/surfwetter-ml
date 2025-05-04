__version__ = "0.1.0"

import os
from pathlib import Path

from pydantic_yaml import parse_yaml_file_as

from surfwetter_ml.config.setting import LibrarySettings

from .core import compute

__all__ = [
    "compute",
]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = parse_yaml_file_as(LibrarySettings, Path(ROOT_DIR, "config", "config.yaml"))
