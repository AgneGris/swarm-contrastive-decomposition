# Swarm-Contrastive Decomposition 🧠

[![PyPI version](https://badge.fury.io/py/swarm-contrastive-decomposition.svg)](https://pypi.org/project/swarm-contrastive-decomposition/)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](https://www.python.org/downloads/)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm%20Noncommercial%201.0.0-blue.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0/)

A Python package for decomposition of neurophysiological time series signals using a Particle Swarm Optimised Independence Estimator for Blind Source Separation.

<div align="center">
    <img src="https://raw.githubusercontent.com/AgneGris/swarm-contrastive-decomposition/main/images/pipeline.png" alt="Pipeline" width="500"/>
</div>

## Table of Contents 📚

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Test Data](#test-data)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Installation 🛠️

### From PyPI (Recommended)

```bash
pip install swarm-contrastive-decomposition
```

### From GitHub (Latest Development Version)

```bash
pip install git+https://github.com/AgneGris/swarm-contrastive-decomposition.git
```

### From Source

```bash
git clone https://github.com/AgneGris/swarm-contrastive-decomposition
cd swarm-contrastive-decomposition
pip install -e .
```

### Verify Installation

```bash
python -c "import scd; print(f'SCD version: {scd.__version__}')"
```

## Quick Start 🚀

```python
import scd

# Train with default configuration
dictionary, timestamps = scd.train("path/to/your/data.mat")

# Save results
scd.save_results("data/output/emg.pkl", dictionary)
```

## Usage

### Basic Usage

```python
import scd

# Use a predefined configuration
dictionary, timestamps = scd.train(
    "path/to/your/data.mat",
    config_name="surface"  # or "default", "intramuscular"
)

scd.save_results("output.pkl", dictionary)
```

### With Configuration Overrides

```python
import scd

# Override specific parameters
dictionary, timestamps = scd.train(
    "path/to/your/data.mat",
    config_name="surface",
    max_iterations=100,  # override for quick testing
    output_final_source_plot=True
)
```

### Step-by-Step Control

```python
import scd

# Load configuration
config = scd.load_config("surface")

# Load data
neural_data = scd.load_data("path/to/your/data.mat", device=config.device)

# Preprocess
neural_data = scd.preprocess_data(neural_data, config)

# Train model
dictionary, timestamps = scd.train_model(neural_data, config)

# Save results
scd.save_results("output.pkl", dictionary)
```

### Supported Data Formats

- `.mat` — MATLAB files, including HDF5-based v7.3 files (specify the variable name with `key` parameter)
- `.npy` — NumPy arrays

```python
# For .mat files with custom variable name
dictionary, timestamps = scd.train("data.mat", key="emg_data")

# For .npy files
dictionary, timestamps = scd.train("data.npy")
```

Data should have shape `(time, channels)` or `(channels, time)` — the loader will automatically transpose if needed.

## Configuration ⚙️

Configurations are defined in `scd/configs.json`. Available presets:

| Config Name | Use Case | Sampling Rate | Description |
|-------------|----------|---------------|-------------|
| `default` | General purpose | 10240 Hz | Balanced settings for most EMG data |
| `surface` | Surface EMG | 10240 Hz | Optimized for surface recordings |
| `intramuscular` | Intramuscular EMG | 10240 Hz | Higher iterations for fine-wire recordings |

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `device` | `"cuda"` for GPU or `"cpu"`; automatically selected when omitted | CUDA when available, otherwise CPU |
| `acceptance_silhouette` | Quality threshold for source acceptance | `0.85` |
| `extension_factor` | Leave unset to derive `1000 / kept channels` automatically. Set it to override. Never capped — see [Choosing the Extension Factor](#choosing-the-extension-factor) | derived |
| `bad_channels` | Channel indices to reject. Replaced with noise at the estimated baseline amplitude of the good channels before decomposition, and excluded from the kept-channel count used to derive `extension_factor` | `null` |
| `low_pass_cutoff` | Low-pass filter cutoff frequency (Hz) | `4400` |
| `high_pass_cutoff` | High-pass filter cutoff frequency (Hz) | `10` |
| `sampling_frequency` | Sampling frequency of your signal (Hz) | `10240` |
| `start_time` | Start time for signal trimming (s). Use `0` for beginning | `0` |
| `end_time` | End time for signal trimming (s). Use `-1` for entire signal | `-1` |
| `max_iterations` | Maximum decomposition iterations | `200` |
| `peel_off_window_size_ms` | Window size for spike-triggered average (ms). `peel_off_window_size` in samples is derived automatically as `ms × fs / 1000` | `20` |
| `reset_peak_separation_ms` | Minimum distance between two detected peaks in the source signal (ms), converted to samples as `ms × fs / 1000` | `4.0` |
| `edge_mask_size_ms` | Masked region at each end of the signal during optimisation (ms), converted to samples as `ms × fs / 1000`. Set `edge_mask_size` instead to pin a raw sample count | `19.5` |
| `output_final_source_plot` | Generate plot of final sources | `false` |
| `use_coeff_var_fitness` | Use coefficient of variation fitness. `true` for EMG, `false` for intracortical | `true` |
| `remove_bad_fr` | Filter sources with firing rates < 2 Hz or > 100 Hz | `true` |
| `adapt_clamp` | Adaptively clamp each source using its personal-best spike statistics; falls back to hard ±30 σ when no spike history exists. Set to `false` to always use the fixed ±30 σ hard clamp | `true` |

### Custom Configuration

Add your own configuration to `scd/configs.json`:

```json
{
    "my_experiment": {
        "device": "cuda",
        "acceptance_silhouette": 0.80,
        "extension_factor": 30,
        "sampling_frequency": 2048,
        ...
    }
}
```

Then use it:

```python
dictionary, timestamps = scd.train("data.mat", config_name="my_experiment")
```

## Choosing the Extension Factor

`K` is chosen so that the extended observation matrix has on the order of 1000 rows:

```
K  ≈  1000 / M          M = kept channels = total channels − bad_channels
```

**This is the default.** If `extension_factor` is absent from your config, SCD derives it from the channel count of the data you pass in and logs the value it used:

```
INFO - extension_factor not set; using 16 (1000 / 63 kept channels)
```

Set `extension_factor` explicitly in your config to override it with any value you like — the derived default applies only when it is unset. All three built-in presets set it explicitly.

To compute the same number yourself:

```python
from scd import recommended_extension_factor

K = recommended_extension_factor(num_channels=64, bad_channels=[56])   # 16
```

Whatever you set is used as-is: SCD never rejects or adjusts an extension factor. Note that larger `K` costs time and memory, as the covariance and whitening stages scale roughly with `(K · M)²` and `(K · M)³` respectively, and that whitening runs on CPU.

### Changed in 0.2.0

This release contains breaking changes. Configs written for 0.1.x may need editing — `Config` rejects unknown keys, so a removed parameter left in a config file raises `TypeError` on load.

**No extension factor is rejected any more.** Versions 0.1.7 and 0.1.8 validated `extension_factor` against a temporal-separation bound `K ≤ T − L` and raised a `ValueError` when it was exceeded. That bound is uninformative: SCD's assumed MUAP duration (`peel_off_window_size_ms`, 20 ms) equals the minimum ISI at `max_firing_rate_hz = 50`, so it evaluates to zero at every sampling rate. In practice it blocked valid configurations, particularly at lower sampling rates such as 2048 Hz, where the standard `1000 / M` choice exceeds it.

| Removed | Reason |
|---|---|
| `compute_extension_factor_bounds()` | Both bounds it returned are unreachable in practice |
| `max_firing_rate_hz` | Only ever fed the removed checks |
| `final_peak_separation` | Referenced nowhere |

**Changed defaults:**

- `extension_factor` was a fixed `100`, which silently produced badly over-extended runs for any config that omitted the key. It is now unset by default and derived as `1000 / kept channels`.
- `edge_mask_size` is now specified in ms as `edge_mask_size_ms` and scaled by the sampling rate; it was a raw sample count tuned for 10 240 Hz. Set `edge_mask_size` to pin a sample count instead.
- Rejected channels are replaced with baseline noise rather than zeroed. Zeroing left exactly-zero rows in the extended matrix, making the covariance rank-deficient and letting whitening amplify those directions. Amplitude comes from `estimate_baseline_noise`, a MAD-based estimate of the noise floor between discharges.

Runs using `bad_channels` will give different results than 0.1.x. Note also that scd-edition scales its channel fill by the pooled standard deviation of the good channels, which includes MUAP activity; pass `noise_std` explicitly to `replace_bad_channels_with_noise` to reproduce its output.

## Test Data 🧪

The source repository includes test data to verify a development installation:

- **File:** `data/input/emg.mat`
- **Type:** Surface EMG
- **Sampling rate:** 10240 Hz
- **Configuration:** Use `"surface"` config

```python
import scd

# Run with test data
dictionary, timestamps = scd.train(
    "data/input/emg.mat",
    config_name="surface"
)

print(f"Found {len(dictionary)} motor units")
```

## Contributing 🤝

We welcome contributions! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/newfeature`)
3. Commit your changes (`git commit -m 'Add some newfeature'`)
4. Push to the branch (`git push origin feature/newfeature`)
5. Open a pull request

## License 📜

This project is licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/). Commercial use is not permitted without a separate license from the licensor.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{grison2024particle,
  author={Grison, Agnese and Clarke, Alexander Kenneth and Muceli, Silvia and Ibáñez, Jaime and Kundu, Aritra and Farina, Dario},
  journal={IEEE Transactions on Biomedical Engineering}, 
  title={A Particle Swarm Optimised Independence Estimator for Blind Source Separation of Neurophysiological Time Series}, 
  year={2025},
  volume={72},
  number={1},
  pages={227--237},
  doi={10.1109/TBME.2024.3446806},
  keywords={Recording; Time series analysis; Sorting; Vectors; Measurement; Electrodes; Probes; Independent component analysis; particle swarm optimisation; blind source separation; intramuscular electromyography; intracortical recording}
}

@article{grison2025unlocking,
  title={Unlocking the full potential of high-density surface EMG: novel non-invasive high-yield motor unit decomposition},
  author={Grison, Agnese and Mendez Guerra, Irene and Clarke, Alexander Kenneth and Muceli, Silvia and Ib{\'a}{\~n}ez, Jaime and Farina, Dario},
  journal={The Journal of Physiology},
  volume={603},
  number={8},
  pages={2281--2300},
  year={2025},
  publisher={Wiley Online Library}
}
```

## Contact

For questions or inquiries:

**Agnese Grison**  
📧 agnese.grison@outlook.it
