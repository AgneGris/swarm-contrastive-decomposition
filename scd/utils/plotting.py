"""Matplotlib functions for plotting decomposition outputs"""

from typing import List, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt

from scd.config.structures import set_random_seed

plt.rcParams["pdf.fonttype"] = 42
set_random_seed(seed=42)


def plot_sources(
    sources: torch.Tensor,
    timestamps: Optional[List[torch.Tensor]] = None,
    exponents: Optional[torch.Tensor] = None,
    fitness: Optional[torch.Tensor] = None,
):
    """Sources of shape [time, sources]"""

    # Check sources dimensions
    if len(sources.shape) == 1:
        sources = sources.unsqueeze(1)
    elif len(sources.shape) > 2:
        raise ValueError(
            "sources must be shape [time, source] or [time] for a single source"
        )
    assert (
        sources.shape[0] > sources.shape[1]
    ), "time must be longer than source in sources input, is it transposed?"

    # Convert inputs to something plt can understand
    sources = sources.detach().cpu().numpy()
    if timestamps is not None:
        timestamps = [t.detach().cpu().numpy() for t in timestamps]
    if exponents is not None:
        exponents = exponents.detach().cpu().numpy()
    if fitness is not None:
        fitness = fitness.detach().cpu().numpy()

    # Build out the plt axis objects
    num_rows = (sources.shape[1] // 2) + (1 if sources.shape[1] % 2 != 0 else 0)
    _, ax_mat = plt.subplots(
        num_rows, 1 if (sources.shape[1] == 1) else 2, figsize=(40, 20)
    )
    if sources.shape[1] != 1:
        ax_mat = ax_mat.flatten()
    else:
        ax_mat = [ax_mat]

    # Add data to plots source by source
    for source_idx in range(sources.shape[1]):
        # Plot the source
        ax_mat[source_idx].plot(sources[:, source_idx])

        # If silhouettes and timestamps entered then highlight "best" timestamps
        if fitness is not None:
            colour = "or" if source_idx == np.argmax(fitness) else "ok"
        else:
            colour = "ok"

        # If timestamps entered then can mark these on the sources
        if timestamps is not None:
            ax_mat[source_idx].plot(
                timestamps[source_idx],
                sources[timestamps[source_idx], source_idx],
                colour,
            )

        # Finally add additional text if exponents and/or fitness entered
        exp, sil = "", ""
        if exponents is not None:
            exp = "Exponent: " + str(np.round(exponents[source_idx], 2))
        if fitness is not None:
            sil = "Fitness: " + str(np.round(fitness[source_idx], 2))
        ax_mat[source_idx].set_title(exp + " " + sil, fontsize=22)

    plt.show()


def plot_accepted_source(source, best_timestamps):
    plt.figure(figsize=(40, 20))
    source = source.cpu().detach().numpy()
    best_timestamps = best_timestamps.cpu().detach().numpy()
    plt.plot(source, linewidth=2)
    plt.plot(best_timestamps, source[best_timestamps], "ro", markersize=20)
    plt.show()
