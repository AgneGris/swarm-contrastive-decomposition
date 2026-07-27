"""Functions to preprocess the signal ready for blind source separation"""

from typing import Optional, Sequence

import torch
from scipy.signal import butter, filtfilt

from scd.config.structures import set_random_seed

set_random_seed(seed=42)


def estimate_baseline_noise(x: torch.Tensor) -> float:
    """
    Estimate the baseline noise amplitude of an EMG segment, robustly.

    Uses the median absolute deviation rescaled to a Gaussian standard
    deviation, sigma = MAD / 0.6745, computed per channel and then reduced
    by the median across channels. MUAPs are sparse and large, so they
    dominate a standard deviation but barely shift a median — this tracks
    the noise floor between discharges rather than the activity on top of it.

    Robustness has limits worth knowing. At high activation the surface EMG
    interference pattern is dense and approximately Gaussian, with no quiet
    inter-spike baseline to find; there the MAD estimate approaches the
    standard deviation because the signal genuinely has no separable
    baseline. The estimate is also computed on unfiltered data, so strong
    mains interference inflates it.

    Parameters
    ----------
    x : torch.Tensor
        EMG of shape (samples, channels).

    Returns
    -------
    float
        Estimated noise standard deviation, or 1e-6 if `x` has no channels.
    """
    if x.shape[1] == 0:
        return 1e-6

    med = x.median(dim=0, keepdim=True).values
    mad = (x - med).abs().median(dim=0).values      # per channel
    sigma = mad / 0.6745
    return float(sigma.median().item())


def replace_bad_channels_with_noise(
    x: torch.Tensor,
    bad_channels: Sequence[int],
    seed: int = 42,
    noise_std: Optional[float] = None,
) -> torch.Tensor:
    """
    Replace bad channels with baseline noise matched to the good channels.

    Zeroing a bad channel leaves an exactly-zero row in the extended
    observation matrix, so the covariance is rank-deficient by one row per
    extension step and whitening amplifies those directions through its
    fixed eigenvalue floor. Filling the channel with noise at the amplitude
    of the good channels keeps the matrix full rank and stops the dead
    channel contributing a spurious common component.

    The noise amplitude is the baseline estimate from
    `estimate_baseline_noise` over the good channels, which tracks the noise
    floor between discharges rather than the overall signal amplitude.

    Noise is drawn from a dedicated CPU generator seeded with `seed`, so it
    is reproducible, unaffected by other random draws, and identical
    regardless of the device `x` lives on.

    NOTE: scd-edition (`_replace_bad_channels` in
    scd_app.core.filter_recalculation) uses the pooled standard deviation of
    the good channels instead, which includes MUAP activity and so gives a
    larger amplitude. To reproduce its output exactly, pass that value via
    `noise_std`.

    Parameters
    ----------
    x : torch.Tensor
        EMG of shape (samples, channels). Modified in place.
    bad_channels : sequence of int
        Channel indices to replace.
    seed : int
        Seed for the noise generator (default 42, matching scd-edition).
    noise_std : float, optional
        Override the estimated amplitude with an explicit value.

    Returns
    -------
    torch.Tensor
        The same tensor, with the bad channels replaced.
    """
    bad = list(bad_channels)
    if not bad:
        return x

    if noise_std is None:
        good = [c for c in range(x.shape[1]) if c not in bad]
        noise_std = estimate_baseline_noise(x[:, good]) if good else 1e-6

    gen = torch.Generator()          # CPU generator -> device-independent noise
    gen.manual_seed(seed)
    noise = torch.randn(x.shape[0], len(bad), generator=gen) * noise_std
    x[:, bad] = noise.to(device=x.device, dtype=x.dtype)
    return x


def recommended_extension_factor(
    num_channels: int,
    bad_channels: Optional[Sequence[int]] = None,
    target_extended_channels: int = 1000,
) -> int:
    """
    Suggest an extension factor from the standard K*M ~ target heuristic.

    Common practice in EMG decomposition is to extend until the observation
    matrix has on the order of 1000 rows, i.e. K ~ target / M. This depends
    only on the channel count, not on the sampling rate, and is the value
    SCD reports in its configuration advice.

    Parameters
    ----------
    num_channels : int
        Total channels in the raw EMG data.
    bad_channels : sequence of int, optional
        Rejected channel indices (zeroed before decomposition).
    target_extended_channels : int
        Target number of extended channels K*M (default 1000).

    Returns
    -------
    int
        Recommended K (at least 1).
    """
    M = num_channels - (len(bad_channels) if bad_channels else 0)
    if M <= 0:
        return 1
    return max(1, round(target_extended_channels / M))


def notch_filter(emg: torch.Tensor, f_samp: float, notch_params: tuple, cutoff_lowpass: int):
    """Filter emg channel by channel"""

    keep = torch.zeros(1).type_as(emg)
    emg = emg.cpu().numpy()

    f_notch, bw, filt_harms = notch_params

    # Select the frequencies to filter
    freqs_to_filter = [f_notch] # base frequency (e.g. 50 Hz for powerline noise in Europe)

    if filt_harms:
        freqs_to_filter.extend([f_notch * i for i in range(2, cutoff_lowpass // f_notch + 1)]) # harmonics up to the low pass cutoff

    # Filter
    for f in freqs_to_filter:
        b, a = butter(2, [2 * (f - bw) / f_samp, 2 * (f + bw) / f_samp], btype="bandstop")
        for channel in range(emg.shape[1]):
            emg[:, channel] = filtfilt(b, a, emg[:, channel])

    return torch.from_numpy(emg).type_as(keep)


def high_pass_filter(emg: torch.Tensor, f_samp: float, cut_off: int):
    """Filter emg channel by channel"""

    keep = torch.zeros(1).type_as(emg)
    emg = emg.cpu().numpy()

    b, a = butter(2, 2 * cut_off / f_samp, btype="highpass")
    for channel in range(emg.shape[1]):
        emg[:, channel] = filtfilt(b, a, emg[:, channel])

    return torch.from_numpy(emg).type_as(keep)


def low_pass_filter(emg: torch.Tensor, f_samp: float, cut_off: int):
    """Filter emg channel by channel"""

    keep = torch.zeros(1).type_as(emg)
    emg = emg.cpu().numpy()

    b, a = butter(2, 2 * cut_off / f_samp, btype="lowpass")
    for channel in range(emg.shape[1]):
        emg[:, channel] = filtfilt(b, a, emg[:, channel])

    return torch.from_numpy(emg).type_as(keep)

def time_differentiate(emg: torch.Tensor) -> torch.Tensor:
    """
    Perform time differentiation of emg channel by channel

    Apply when the number of active sources is high to suppress small
    ones and improve discrimination between active sources
    """
    # Differentiate the signal
    emg = emg[1:] - emg[:-1]
    
    # Duplicate the first sample of each channel back to maintain the original shape
    emg = torch.cat((emg[:1], emg), dim=0)

    return emg

def extend(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Extends each sample with factor past values"""

    assert x.ndim == 2, "Input must be two-dimensional"

    # Pad end with zeros to stop torch.roll moving end samples to start
    x = torch.concat([torch.zeros([factor, x.shape[1]]).type_as(x), x])

    # Perform extension and return with rolled values removed
    x = torch.concat([x.roll(shift, 0) for shift in range(factor)], 1)
    return x[factor:]


def whiten(x: torch.Tensor, method: str = "zca", return_matrix: bool = False) -> torch.Tensor:
    """
    Performs whitening on input of shape (samples, channels)

    Whitening transform can be calculated by selecting the below options
    for the method argument:

    "chol": Cholesky method
    "zca": ZCA method on covariance matrix
    "pca": PCA method on covariance matrix
    "zca_cor": ZCA method on correlation matrix
    "pca_cor": PCA method on correlation matrix

    """

    # Inconsistent behaviour on cuda - switch to cpu
    keep = torch.zeros(1).type_as(x)
    x = x.to(device="cpu", dtype=torch.float32)

    x = x.t()
    x -= x.mean(1, keepdim=True)
    cov = x.cov()

    if method in ["zca", "pca", "chol"]:
        u, s, _ = torch.linalg.svd(cov)
    elif method in ["zca_cor", "pca_cor"]:
        v_inv_sqrt = cov.diag().sqrt().reciprocal().diag()
        corr = v_inv_sqrt.matmul(cov).matmul(v_inv_sqrt)
        u, s, _ = torch.linalg.svd(corr)
    else:
        raise Exception("Specified method not in list.")

    if method == "chol":
        s_inv = (s + 1e-10).reciprocal().diag()
        cov_inv = u.matmul(s_inv).matmul(u.t())
        w = torch.linalg.cholesky(cov_inv).t()
    else:
        s_inv_sqrt = torch.sqrt(s + 1e-10).reciprocal().diag()

    if method == "zca":
        w = u.matmul(s_inv_sqrt).matmul(u.t())
    elif method == "pca":
        w = torch.matmul(s_inv_sqrt, u.t())
    elif method == "zca_cor":
        w = u.matmul(s_inv_sqrt).matmul(u.t()).matmul(v_inv_sqrt)
    elif method == "pca_cor":
        w = s_inv_sqrt.matmul(u.t()).matmul(v_inv_sqrt)

    whitened_x = torch.matmul(w, x).t().type_as(keep)
    
    if return_matrix:
        return whitened_x, w.type_as(keep)
    return whitened_x


def autocorrelation_whiten(x: torch.Tensor, extension_factor: int, method: str = "zca"):
    """Performs segmented autocorrelation whitening on each channel
    Using the overlap add method to reconstruct the signal"""

    all_channels = []
    for channel in x.t():
        windowed_channel = torch.nn.functional.unfold(
            channel.unsqueeze(0).unsqueeze(0), (1, extension_factor)
        ).t()

        channel = torch.nn.functional.fold(
            whiten(windowed_channel, method).t(),
            output_size=(1, channel.shape[0]),
            kernel_size=(1, extension_factor),
            stride=1,
        ).squeeze()

        overlap_count = torch.nn.functional.fold(
            torch.ones_like(windowed_channel.t()),
            (1, channel.shape[0]),
            (1, extension_factor),
        ).squeeze()

        all_channels.append(channel / overlap_count)

    return torch.stack(all_channels, 1)
