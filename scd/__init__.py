"""
Swarm Contrastive Decomposition (SCD)
"""

__version__ = "0.1.0"

from scd.models.scd import SwarmContrastiveDecomposition
from scd.config.structures import Config, set_random_seed
from scd.processing.postprocess import save_results

from scd.train import (
    load_config,
    load_data,
    preprocess_data,
    train_model,
    train,
)

__all__ = [
    "__version__",
    "SwarmContrastiveDecomposition",
    "Config",
    "set_random_seed",
    "save_results",
    "load_config",
    "load_data",
    "preprocess_data",
    "train_model",
    "train",
]