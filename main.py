import numpy as np
import torch
from pathlib import Path
import scipy.io as sio

from config.structures import set_random_seed, Config
from models.scd import SwarmContrastiveDecomposition
from processing.postprocess import save_results

set_random_seed(seed=42)


def train(path):

    device = "cuda"
    acceptance_silhouette = 0.85
    extension_factor = 20
    low_pass_cutoff = 4400
    high_pass_cutoff = 10
    start_time = 0
    end_time = -1
    max_iterations = 250
    sampling_frequency = 10240
    peel_off_window_size_ms = 20   # ms
    output_final_source_plot = True
    use_coeff_var_fitness = True
    remove_bad_fr = True

    config = Config(
        device=device,
        acceptance_silhouette=acceptance_silhouette,
        extension_factor=extension_factor,
        low_pass_cutoff=low_pass_cutoff,
        high_pass_cutoff=high_pass_cutoff,
        sampling_frequency=sampling_frequency,
        start_time=start_time,
        end_time=end_time,
        max_iterations=max_iterations,
        peel_off_window_size_ms=peel_off_window_size_ms,
        output_final_source_plot=output_final_source_plot,
        use_coeff_var_fitness=use_coeff_var_fitness,
        remove_bad_fr=remove_bad_fr,
    )

    # Load data
    if path.suffix == ".mat":
        mat = sio.loadmat(path)
        neural_data = (
            torch.from_numpy(mat["emg"]).t().to(device=device, dtype=torch.float32)
        )  # time, channels
    elif path.suffix == ".npy":
        npy_data = np.load(path)
        neural_data = torch.from_numpy(npy_data).to(device=device, dtype=torch.float32)
    else:
        raise ValueError(
            "Data format not supported. Please provide data in .mat or .npy format."
        )

    neural_data = neural_data[
        config.start_time * sampling_frequency : config.end_time * sampling_frequency, :
    ]

    # Initiate the model and run
    model = SwarmContrastiveDecomposition()
    predicted_timestamps, dictionary = model.run(neural_data, config)

    return dictionary, predicted_timestamps


if __name__ == "__main__":
    # Uncomment the next three lines to run in interactive window
    # import sys
    # sys.argv=['']
    # del sys

    HOME = Path.cwd().joinpath("data", "input")
    file_name = "emg"
    path = HOME.joinpath(file_name).with_suffix(".npy")
    output_path = (
        Path(str(HOME).replace("input", "output"))
        .joinpath(file_name)
        .with_suffix(".pkl")
    )

    dictionary, _ = train(path)

    save_results(output_path, dictionary)
    print(f"Saved results to {output_path}")
