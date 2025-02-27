"""Functions needed to convert time series into spike timestamps and
assess the quality of these timestamps"""

from typing import Tuple, List

import torch
from scipy.signal import find_peaks
from config.structures import set_random_seed

set_random_seed(seed=42)


def scatter_mean(input: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Calculates the mean of subsets within a 1D tensor of floats, with
    subsets specified by a 1D tensor of ints
    """

    total = torch.zeros(int(index.max()) + 1, dtype=input.dtype).scatter_add_(
        0, index, input
    )
    count = torch.zeros(int(index.max()) + 1, dtype=input.dtype).scatter_add_(
        0, index, torch.ones_like(input)
    )

    return total / count.clamp(min=1)


def scatter_median(input: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Calculates the median of subsets within a 1D tensor of floats, with
    subsets specified by a 1D tensor of ints
    """

    # Ensure input and index are 1D tensors
    if input.dim() != 1 or index.dim() != 1:
        return torch.tensor(float("nan"), dtype=input.dtype)

    # Ensure the lengths of input and index tensors match
    if input.size(0) != index.size(0):
        return torch.tensor(float("nan"), dtype=input.dtype)

    unique_indices = torch.unique(index)
    output = torch.empty_like(unique_indices, dtype=input.dtype)

    for i, unique_index in enumerate(unique_indices):
        mask = index == unique_index
        subset = input[mask]

        # Check if the subset is empty and return NaN if so
        if subset.numel() == 0:
            median = torch.tensor(float("nan"), dtype=input.dtype)
        else:
            median = torch.median(subset)
        output[i] = median

    return output


def pairwise_silhouette(heights: torch.Tensor, centroids: torch.Tensor):
    """Calculates a two-class true silhouette based on the distance of each
    sample to other samples in its class and samples in the nearest other class
    heights and centroids must be 1D tensors
    """

    # Get distances to centroids
    distances = torch.abs(heights.unsqueeze(1).tile([1, 2]) - centroids.unsqueeze(0))
    assignments = distances.sort(1)[1].type_as(distances)

    # First find pairwise distance and cluster assignments (same or different)
    pair_distances = torch.abs(heights - heights.t())
    pair_assignments = torch.matmul(assignments, assignments.t())

    # Next calculate the average distances between samples in the same cluster
    in_class_sum = (pair_distances * pair_assignments).sum(1)
    in_class_total = (assignments * assignments.sum(0)).max(1)[0]
    in_class_mean = in_class_sum / (in_class_total - 1)

    # Next calculate the average distances between samples in different clusters
    out_class_sum = (pair_distances * (1 - pair_assignments)).sum(1)
    out_class_total = ((1 - assignments) * assignments.sum(0)).max(1)[0]
    out_class_mean = out_class_sum / out_class_total

    # Finally use these means to calculate the silhouette score
    maximal_value = torch.maximum(out_class_mean, in_class_mean)
    return ((out_class_mean - in_class_mean) / maximal_value).mean()


def centroid_silhouette(heights: torch.Tensor, centroids: torch.Tensor):
    """Calculates a two-class pseudo-silhouette based on the distance of each
    sample to its assigned and nearest non-assigned cetroids
    heights and centroids must be 1D tensors
    """

    # Get distances to centroids
    distances = torch.abs(heights.unsqueeze(1).tile([1, 2]) - centroids.unsqueeze(0))
    assignments = distances.sort(1)[1].type_as(distances)

    # Get distances to in class and out class centroids
    in_class = (distances * (1 - assignments)).sum(1)
    out_class = (distances * assignments).sum(1)

    # Finally use these means to calculate the silhouette score
    maximal_value = torch.maximum(out_class, in_class)
    return ((out_class - in_class) / maximal_value).mean()


def source_to_timestamps(
    source: torch.Tensor,
    min_peak_separation: int = 30,
    use_pairwise_silhouette: bool = False,
    use_mean: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Two class k means/median peak selection that returns the cluster with
    the highest peaks and the silhouette score

    Source must be a 1D torch tensor

    source_centroid_weighting controls whether to increase the source centroid
    value, reducing false positives at the cost of more false negatives

    Silhouette will be calculated relative to centroids if pairwise_silhouette
    is True, or relative to all other samples if set to False
    """

    # Part 1: Peak finding
    # Find peaks in source
    assert source.isfinite().all(), "Source contains NaNs/Infs"
    locations, properties = find_peaks(
        source.detach().cpu(), height=0, distance=min_peak_separation
    )
    locations, heights = [
        torch.from_numpy(arr).to(device=source.device)
        for arr in [locations, properties["peak_heights"]]
    ]

    # Part 2: K mean/median clustering
    # Set whether to use mean or median for average
    scatter_average = scatter_mean if use_mean else scatter_median

    # Use values of peaks to optimise centroid locations
    heights = heights.unsqueeze(1)
    if heights.numel() > 0:
        centroids = torch.tensor(
            [heights.min(), heights.max()], device=heights.device
        ).type_as(heights)

        centroid_distance = centroids.unsqueeze(0)
        if centroid_distance.device != heights.device:
            centroid_distance = centroid_distance.to(heights.device)

        for _ in range(100):
            distances = torch.abs(heights.tile([1, 2]) - centroid_distance)
            centroids = scatter_average(heights.squeeze(), distances.argmin(1))

        # Check if centroids are empty
        if centroids.numel() == 0 or torch.isnan(centroids).any():
            return torch.tensor([]).type_as(heights), torch.tensor(0.0).type_as(heights)

        # Part 3: Calculate silhouette score relative to centroids or pairwise
        silhouette = (
            pairwise_silhouette(heights.squeeze(), centroids)
            if use_pairwise_silhouette
            else centroid_silhouette(heights.squeeze(), centroids)
        )

        locations = locations[distances.argmin(1) == centroids.argmax()]

        return locations, source[locations], silhouette

    else:
        return torch.tensor([]).type_as(heights), torch.tensor(0.0).type_as(heights), torch.tensor(0.0).type_as(heights)



def spike_triggered_average(
    emg: torch.Tensor, timestamps: torch.Tensor, window: int = 1
) -> torch.Tensor:
    """Given timestamps, calculate a spike triggered average of
    shape [window, channels] on emg of shape [time, channels]"""

    assert len(emg.shape) == 2, "EMG must be of shape [time, channels]"
    if timestamps.ndim == 0:
        return torch.zeros((window, emg.shape[1])).to(emg.device)

    hw = int(window / 2)
    odd = int(window % 2)

    emg_windows = []

    for s in timestamps:
        if s - hw < 0 or s + hw + odd > emg.shape[0]:
            continue
        emg_windows.append(emg[(s - hw) : (s + hw + odd)])

    if not emg_windows:
        return torch.zeros((window, emg.shape[1])).to(emg.device)

    sta = torch.stack(emg_windows).mean(0)

    return sta


def peel_off_source(
    emg: torch.Tensor, timestamps: torch.Tensor, window: int
) -> torch.Tensor:
    """
    Function to estimate and remove source contributions.

    Parameters:
    - emg: The input EMG data of shape [samples, channels]
    - timestamps: A list of spike timestamps
    - window: STA window size
    """

    sta = spike_triggered_average(emg, timestamps, window)

    hw = int(window / 2)
    odd = int(window % 2)
    for s in timestamps:
        if s - hw < 0 or s + hw + odd > emg.shape[0]:
            continue

        emg[(s - hw) : (s + hw + odd)] -= sta

    return emg


def rate_of_agreement(
    timestamps_1: torch.Tensor,
    timestamps_2: torch.Tensor,
    tolerance: int = 5,
    max_shift: int = 100,
) -> torch.Tensor:
    """Matrix operation based implementation of rate of agreement finding
    between two timestamp sets"""

    assert tolerance >= 0, "tolerance must be non-negative"
    assert max_shift >= 0, "max_shift must be non-negative"

    # Make sure timestamps on same device
    timestamps_2 = timestamps_2.type_as(timestamps_1)

    # Set tolerance aranges
    tol_range = torch.arange(-tolerance, tolerance).type_as(timestamps_1)

    # Create shifted versions of timestamp_1 with tolerance bands
    expand_1 = timestamps_1.unsqueeze(0)
    expand_1 = expand_1 - tol_range.unsqueeze(1)
    expand_1 = expand_1.unsqueeze(-1).tile([1, 1, timestamps_2.shape[0]])

    # Tile up timestamp_2
    expand_2 = (
        timestamps_2.unsqueeze(0)
        .unsqueeze(0)
        .tile([expand_1.shape[0], expand_1.shape[1], 1])
    )

    # Iterate through shift amounts
    matches = []
    for shift in range(-max_shift, max_shift):
        matches.append((expand_1 + shift == expand_2).any(2).any(0).sum())
    matches = torch.stack(matches).max()

    # Return RoA as percentage
    return 100 * matches.divide(timestamps_1.shape[0] + timestamps_2.shape[0] - matches)


def bootstrapped_coeff_var(timestamps: torch.Tensor, n_iterations: int = 1000):
    """Runs a bootstrapped version of the interspike coefficient of variation
    and returns the 75th quantile"""

    isi = timestamps.diff().float()

    # Remove big gaps in ISI (likely to be real derecruitment-recruitments)
    isi = isi[isi < 5 * isi.median()]
    n_isi = isi.shape[0]
    assert n_isi > 2, "Not enough timestamps for ISI"

    # Generate a matrix of random indices for bootstrapping
    indices = torch.randint(0, n_isi, (n_iterations, n_isi)).to(timestamps.device)

    # Using the indices to get bootstrapped ISIs
    resampled_isi = torch.index_select(isi, 0, indices.view(-1)).view(
        n_iterations, n_isi
    )

    resampled_coeff_vars = resampled_isi.std(1) / resampled_isi.mean(1)

    return resampled_coeff_vars.quantile(0.75)


def calculate_firing_rates(
    unit, window_size_in_seconds: int = 1, fsamp2: float = 10240
):
    spike_times_in_seconds = unit.float() / fsamp2

    # Create a tensor of window start times
    window_start_times = torch.arange(
        spike_times_in_seconds[0],
        spike_times_in_seconds[-1],
        window_size_in_seconds,
    ).to(spike_times_in_seconds.device)

    # Create a tensor of window end times
    window_end_times = (window_start_times + window_size_in_seconds).to(
        spike_times_in_seconds.device
    )

    # Use broadcasting to identify spikes in each window (creating a 2D tensor of booleans)
    spikes_in_windows = (spike_times_in_seconds[:, None] >= window_start_times) & (
        spike_times_in_seconds[:, None] < window_end_times
    )

    # Sum along the time axis to get the number of spikes in each window
    spikes_count_per_window = spikes_in_windows.sum(dim=0)

    # Calculate the firing rate in Hz
    firing_rates_in_hz = (
        (spikes_count_per_window.float() / window_size_in_seconds).mean().item()
    )

    return firing_rates_in_hz


def find_quality_metric(
    timestamps_1: torch.Tensor,
    timestamps_2: torch.Tensor,
    metric: str = "roa",
    tolerance: int = 5,
    max_shift: int = 100,
) -> torch.Tensor:
    """Matrix operation based implementation of quality metric finding
    between two timestamp sets. If using precision or recall,
    timestamps_1 is the prediction and timestamps_2 the ground truth"""

    assert tolerance >= 0, "tolerance must be non-negative"
    assert max_shift >= 0, "max_shift must be non-negative"

    # Make sure timestamps on same device
    timestamps_2 = timestamps_2.type_as(timestamps_1)

    # Set tolerance aranges
    tol_range = torch.arange(-tolerance, tolerance).type_as(timestamps_1)

    # Create shifted versions of timestamp_1 with tolerance bands
    expand_1 = timestamps_1.unsqueeze(0)
    expand_1 = expand_1 - tol_range.unsqueeze(1)
    expand_1 = expand_1.unsqueeze(-1).tile([1, 1, timestamps_2.shape[0]])

    # Tile up timestamp_2
    expand_2 = (
        timestamps_2.unsqueeze(0)
        .unsqueeze(0)
        .tile([expand_1.shape[0], expand_1.shape[1], 1])
    )

    # Iterate through shift amounts
    matches = []
    for shift in range(-max_shift, max_shift):
        matches.append((expand_1 + shift == expand_2).any(2).any(0).sum())
    matches = torch.stack(matches).max()

    if metric == "roa":
        return 100 * matches.divide(
            timestamps_1.shape[0] + timestamps_2.shape[0] - matches
        )
    elif metric == "precision":
        return 100 * matches.divide(timestamps_1.shape[0])
    elif metric == "recall":
        return 100 * matches.divide(timestamps_2.shape[0])
    else:
        raise ValueError("Specified quality metric not available.")
