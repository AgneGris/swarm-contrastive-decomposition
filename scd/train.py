import json
import logging
from pathlib import Path
from importlib import resources

import numpy as np
import scipy.io as sio
import torch

from scd.config.structures import Config, set_random_seed
from scd.models.scd import SwarmContrastiveDecomposition

set_random_seed(seed=42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_name: str = "default", config_file: Path = None) -> Config:
    """
    Load configuration from JSON file.
    
    Parameters
    ----------
    config_name : str
        Name of the configuration to load (e.g., "default")
    config_file : Path, optional
        Path to custom config file. If None, uses built-in configs.json
    
    Returns
    -------
    Config
        Configuration object
    """
    if config_file is None:
        # Load from package's built-in configs.json
        with resources.files("scd").joinpath("configs.json").open("r") as f:
            config_data = json.load(f)
    else:
        with open(config_file, "r") as f:
            config_data = json.load(f)
    
    selected_config = config_data.get(config_name, config_data["default"])
    logger.info(f"Loaded config: {config_name}")
    return Config(**selected_config)

def load_data(path: Path, key: str = "emg", device: str = "cuda") -> torch.Tensor:
    """
    Load neural data from .mat or .npy file.
    
    Parameters
    ----------
    path : Path
        Path to data file
    key : str
        Key/variable name in .mat file (default: "emg")
    device : str
        Device to load tensor to ("cuda" or "cpu")
    
    Returns
    -------
    torch.Tensor
        Neural data with shape (time, channels)
    """
    path = Path(path)
    
    # Load data based on file format
    if path.suffix == ".mat":
        mat = sio.loadmat(path)
        data = mat[key]
    elif path.suffix == ".npy":
        data = np.load(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}. Use .mat or .npy")
    
    # Convert to tensor
    neural_data = torch.from_numpy(data).to(device=device, dtype=torch.float32)
    
    # Ensure shape is (time, channels) - time should be the longer dimension
    if neural_data.shape[1] > neural_data.shape[0]:
        neural_data = neural_data.T
        logger.info(f"Transposed data to (time, channels)")
    
    logger.info(f"Loaded data from {path.name}, shape: {neural_data.shape}")
    return neural_data

def preprocess_data(neural_data: torch.Tensor, config: Config) -> torch.Tensor:
    """
    Preprocess neural data (slicing, bad channel removal).
    
    Parameters
    ----------
    neural_data : torch.Tensor
        Raw neural data (time, channels)
    config : Config
        Configuration object
    
    Returns
    -------
    torch.Tensor
        Preprocessed neural data
    """
    start_idx = int(config.start_time * config.sampling_frequency)
    
    if config.end_time == -1 or config.end_time <= 0:
        end_idx = None
    else:
        end_idx = int(config.end_time * config.sampling_frequency)
    
    neural_data = neural_data[start_idx:end_idx, :]
    
    # Zero out bad channels if specified
    if hasattr(config, 'bad_channels') and config.bad_channels:
        neural_data[:, config.bad_channels] = 0
        logger.info(f"Zeroed bad channels: {config.bad_channels}")
    
    logger.info(f"Preprocessed data shape: {neural_data.shape}")
    return neural_data

def train_model(neural_data: torch.Tensor, config: Config) -> tuple:
    """
    Run the SwarmContrastiveDecomposition model.
    
    Parameters
    ----------
    neural_data : torch.Tensor
        Preprocessed neural data (time, channels)
    config : Config
        Configuration object
    
    Returns
    -------
    tuple
        (dictionary, predicted_timestamps)
    """
    model = SwarmContrastiveDecomposition()
    predicted_timestamps, dictionary = model.run(neural_data, config)
    return dictionary, predicted_timestamps

def train(
    path: Path,
    config_name: str = "default",
    config_file: Path = None,
    key: str = "emg",
    **config_overrides
) -> tuple:
    """
    Full training pipeline: load data, preprocess, and train.
    
    Parameters
    ----------
    path : Path
        Path to data file (.mat or .npy)
    config_name : str
        Name of configuration to load from configs.json
    config_file : Path, optional
        Path to custom config file
    key : str
        Key/variable name in .mat file
    **config_overrides
        Override specific config values (e.g., max_iterations=100)
    
    Returns
    -------
    tuple
        (dictionary, predicted_timestamps)
    
    Example
    -------
    >>> dictionary, timestamps = train("data/emg.mat", config_name="default")
    >>> dictionary, timestamps = train("data/emg.mat", max_iterations=100)
    """
    # Load config
    config = load_config(config_name, config_file)
    
    # Apply any overrides
    for key_name, value in config_overrides.items():
        if hasattr(config, key_name):
            setattr(config, key_name, value)
            logger.info(f"Config override: {key_name} = {value}")
    
    # Load and preprocess data
    neural_data = load_data(path, key=key, device=config.device)
    neural_data = preprocess_data(neural_data, config)
    
    # Train
    dictionary, timestamps = train_model(neural_data, config)
    
    return dictionary, timestamps