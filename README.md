# Swarm-Contrastive Decomposition 🧠

## Overview 📝

A Python package for decomposition of neurophysiological time series signals using a Particle Swarm Optimised Independence Estimator for Blind Source Separation.

<div align="center">
    <img src="images/pipeline.png" alt="Pipeline" width="500"/>
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

### Prerequisites

- Python 3.10+
- PyTorch with CUDA support (recommended)

### Option 1: Install as a Package (Recommended)

```bash
# Clone the repository
git clone https://github.com/AgneGris/swarm-contrastive-decomposition
cd swarm-contrastive-decomposition

# Install the package
pip install -e .
```

### Option 2: Using Conda Environment

```bash
# Clone the repository
git clone https://github.com/AgneGris/swarm-contrastive-decomposition
cd swarm-contrastive-decomposition

# Create conda environment
conda env create -f decomposition.yml
conda activate decomposition

# Install the package
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
dictionary, timestamps = scd.train("data/input/emg.npy")

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
    "data/input/emg.npy",
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
neural_data = scd.load_data("data/input/emg.npy", device=config.device)

# Preprocess
neural_data = scd.preprocess_data(neural_data, config)

# Train model
dictionary, timestamps = scd.train_model(neural_data, config)

# Save results
scd.save_results("output.pkl", dictionary)
```

### Supported Data Formats

- `.mat` — MATLAB files (specify the variable name with `key` parameter)
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
| `device` | `"cuda"` for GPU or `"cpu"` | `"cuda"` |
| `acceptance_silhouette` | Quality threshold for source acceptance | `0.85` |
| `extension_factor` | Typically `1000 / num_channels`. Higher values may improve results | `25` |
| `low_pass_cutoff` | Low-pass filter cutoff frequency (Hz) | `4400` |
| `high_pass_cutoff` | High-pass filter cutoff frequency (Hz) | `10` |
| `sampling_frequency` | Sampling frequency of your signal (Hz) | `10240` |
| `start_time` | Start time for signal trimming (s). Use `0` for beginning | `0` |
| `end_time` | End time for signal trimming (s). Use `-1` for entire signal | `-1` |
| `max_iterations` | Maximum decomposition iterations | `200` |
| `peel_off_window_size_ms` | Window size for spike-triggered average (ms) | `20` |
| `output_final_source_plot` | Generate plot of final sources | `false` |
| `use_coeff_var_fitness` | Use coefficient of variation fitness. `true` for EMG, `false` for intracortical | `true` |
| `remove_bad_fr` | Filter sources with firing rates < 2 Hz or > 100 Hz | `true` |
| `clamp_percentile` | Percentile for amplitude clamping | `0.999` |

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

## Test Data 🧪

The repository includes test data to verify your installation:

- **File:** `data/input/emg.npy`
- **Type:** Surface EMG
- **Sampling rate:** 10240 Hz
- **Configuration:** Use `"surface"` config

```python
import scd

# Run with test data
dictionary, timestamps = scd.train(
    "data/input/emg.npy",
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

This project is licensed under the CC BY-NC 4.0 License.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{grison2024particle,
  author={Grison, Agnese and Clarke, Alexander Kenneth and Muceli, Silvia and Ibáñez, Jaime and Kundu, Aritra and Farina, Dario},
  journal={IEEE Transactions on Biomedical Engineering}, 
  title={A Particle Swarm Optimised Independence Estimator for Blind Source Separation of Neurophysiological Time Series}, 
  year={2024},
  volume={},
  number={},
  pages={1-11},
  doi={10.1109/TBME.2024.3446806},
  keywords={Recording; Time series analysis; Sorting; Vectors; Measurement; Electrodes; Probes; Independent component analysis; particle swarm optimisation; blind source separation; intramuscular electromyography; intracortical recording}
}
```

## Contact

For questions or inquiries:

**Agnese Grison**  
📧 agnese.grison16@imperial.ac.uk