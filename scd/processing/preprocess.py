"""Functions to preprocess the signal ready for blind source separation"""

import torch
from scipy.signal import butter, filtfilt

from scd.config.structures import set_random_seed

set_random_seed(seed=42)

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


def whiten(x: torch.Tensor, method: str = "zca") -> torch.Tensor:
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

    return torch.matmul(w, x).t().type_as(keep)


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
