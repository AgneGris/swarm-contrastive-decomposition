"""
Swarm Contrastive Decomposition (SCD)
"""

__version__ = "0.2.4"

from scd.config.structures import Config, set_random_seed
from scd.models.scd import SwarmContrastiveDecomposition
from scd.processing.postprocess import save_results, signal_to_array
from scd.processing.preprocess import (
    estimate_baseline_noise,
    recommended_extension_factor,
    replace_bad_channels_with_noise,
)
from scd.train import (
    load_config,
    load_data,
    preprocess_data,
    train,
    train_model,
)

__all__ = [
    "Config",
    "SwarmContrastiveDecomposition",
    "__version__",
    "estimate_baseline_noise",
    "load_config",
    "load_data",
    "preprocess_data",
    "recommended_extension_factor",
    "replace_bad_channels_with_noise",
    "save_results",
    "set_random_seed",
    "signal_to_array",
    "train",
    "train_model",
]
