"""Functions to postprocess the results of the model."""

import pickle as pkl
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch


def signal_to_array(
    neural_data: torch.Tensor | npt.ArrayLike,
) -> npt.NDArray[np.float32]:
    """
    Convert a loaded signal to the layout stored under ``"data"``.

    Accepts a torch tensor (any device) or array-like of shape
    (time, channels) or (channels, time) and returns a CPU float32 numpy
    array of shape (channels, samples) - the layout scd-edition uses for
    the raw EMG in its own files.
    """
    if isinstance(neural_data, torch.Tensor):
        neural_data = neural_data.detach().cpu().numpy()
    data = np.asarray(neural_data)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D signal, got shape {data.shape}")
    # Time is the longer axis, as in load_data
    if data.shape[0] > data.shape[1]:
        data = data.T
    # Always copy: preprocess_data and the model modify the loaded tensor in
    # place, and a transposed view of a Fortran-ordered array (as scipy
    # returns MATLAB matrices) is already C-contiguous, so ascontiguousarray
    # would alias it.
    return np.array(data, dtype=np.float32, order="C", copy=True)


def save_results(
    output_datafile: str | Path,
    dictionary_result: MutableMapping[str, Any],
    neural_data: torch.Tensor | npt.ArrayLike | None = None,
) -> None:
    """
    Save the dictionary_result to the output_datafile.

    Args:
        output_datafile (Path | str): The path to the output data file.
        dictionary_result (dict): The dictionary to be saved.
        neural_data (Tensor | ndarray | None): The signal as returned by
            load_data (before preprocess_data). When given it is stored under
            ``"data"`` as a (channels, samples) float32 array so the output can
            be edited in scd-edition (MUAPs, filter recalculation). This makes
            the file larger by channels x samples x 4 bytes. ``train`` already
            attaches the signal when ``config.save_data`` is true, so pass it
            here only in the step-by-step workflow.
    """

    if neural_data is not None:
        dictionary_result["data"] = signal_to_array(neural_data)

    output_datafile = Path(output_datafile)
    output_datafile.parent.mkdir(parents=True, exist_ok=True)
    with open(output_datafile, "wb") as f:
        pkl.dump(dictionary_result, f)
