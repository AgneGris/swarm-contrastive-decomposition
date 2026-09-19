"""
Swarm Contrastive Decomposition (SCD)
"""

__version__ = "0.2.3"

from scd.models.scd import SwarmContrastiveDecomposition
from scd.config.structures import Config, set_random_seed
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
    train_model,
    train,
)

__all__ = [
    "__version__",
    "SwarmContrastiveDecomposition",
    "Config",
    "set_random_seed",
    "save_results",
    "signal_to_array",
    "estimate_baseline_noise",
    "recommended_extension_factor",
    "replace_bad_channels_with_noise",
    "load_config",
    "load_data",
    "preprocess_data",
    "train_model",
    "train",
]
