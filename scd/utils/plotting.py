"""Matplotlib functions for plotting decomposition outputs"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from scd.config.structures import set_random_seed

plt.rcParams["pdf.fonttype"] = 42
set_random_seed(seed=42)


def plot_sources(
    sources: torch.Tensor,
    timestamps: list[torch.Tensor] | None = None,
    exponents: torch.Tensor | None = None,
    fitness: torch.Tensor | None = None,
) -> None:
    """Sources of shape [time, sources]"""

    # Check sources dimensions
    if len(sources.shape) == 1:
        sources = sources.unsqueeze(1)
    elif len(sources.shape) > 2:
        raise ValueError(
            "sources must be shape [time, source] or [time] for a single source"
        )
    assert sources.shape[0] > sources.shape[1], (
        "time must be longer than source in sources input, is it transposed?"
    )

    # Convert inputs to something plt can understand
    source_values = sources.detach().cpu().numpy()
    timestamp_values = (
        [timestamp.detach().cpu().numpy() for timestamp in timestamps]
        if timestamps is not None
        else None
    )
    exponent_values = (
        exponents.detach().cpu().numpy() if exponents is not None else None
    )
    fitness_values = fitness.detach().cpu().numpy() if fitness is not None else None

    # Build out the plt axis objects
    num_rows = (source_values.shape[1] // 2) + (
        1 if source_values.shape[1] % 2 != 0 else 0
    )
    _, ax_mat = plt.subplots(
        num_rows, 1 if (source_values.shape[1] == 1) else 2, figsize=(40, 20)
    )
    ax_mat = ax_mat.flatten() if source_values.shape[1] != 1 else [ax_mat]

    # Add data to plots source by source
    for source_idx in range(source_values.shape[1]):
        # Plot the source
        ax_mat[source_idx].plot(source_values[:, source_idx])

        # If silhouettes and timestamps entered then highlight "best" timestamps
        if fitness_values is not None:
            colour = "or" if source_idx == np.argmax(fitness_values) else "ok"
        else:
            colour = "ok"

        # If timestamps entered then can mark these on the sources
        if timestamp_values is not None:
            ax_mat[source_idx].plot(
                timestamp_values[source_idx],
                source_values[timestamp_values[source_idx], source_idx],
                colour,
            )

        # Finally add additional text if exponents and/or fitness entered
        exp, sil = "", ""
        if exponent_values is not None:
            exp = "Exponent: " + str(np.round(exponent_values[source_idx], 2))
        if fitness_values is not None:
            sil = "Fitness: " + str(np.round(fitness_values[source_idx], 2))
        ax_mat[source_idx].set_title(exp + " " + sil, fontsize=22)

    plt.show()


def plot_accepted_source(source: torch.Tensor, best_timestamps: torch.Tensor) -> None:
    plt.figure(figsize=(40, 20))
    source_values = source.cpu().detach().numpy()
    timestamp_values = best_timestamps.cpu().detach().numpy()
    plt.plot(source_values, linewidth=2)
    plt.plot(timestamp_values, source_values[timestamp_values], "ro", markersize=20)
    plt.show()
