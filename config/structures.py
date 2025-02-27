"""Dataclass structure for configuration and model data containers"""

from typing import Optional, Sequence, Tuple
from dataclasses import dataclass

import torch
import random
import numpy as np


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)


@dataclass
class Config:
    # Data parameters
    start_time: int = 0
    end_time: int = -1

    # EMG preprocessing parameters
    sampling_frequency: Optional[int] = None
    time_differentiate: Optional[bool] = None
    notch_params: Optional[Tuple[int, float, bool]] = None  # powerline frequency, bandwidth, harmonics
    low_pass_cutoff: Optional[int] = None
    high_pass_cutoff: Optional[int] = None
    extension_factor: int = 100
    whitening_method: str = "zca"
    autocorrelation_whiten: bool = False

    # Main run parameters
    max_iterations: int = 250
    iteration_patience: int = 20
    acceptance_silhouette: float = 0.85
    acceptance_max_roa: float = 30
    peel_off_window_size_ms: int = 20
    peel_off_repeats: bool = True
    remove_bad_fr: bool = True
    clamp_percentile: Optional[float] = 0.999

    # ICA parameters
    max_ica_steps: int = 1000
    ica_patience: int = 10
    ica_learning_rate: float = 0.001
    ica_momentum: float = 0.9
    edge_mask_size: int = 200

    # Swarm parameters
    max_swarm_steps: int = 100
    swarm_patience: int = 10
    starting_exponents: Sequence[float] = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    swarm_step_std_personal: float = 0.1
    swarm_step_std_global: float = 0.1
    personal_weighting: float = 0.3
    global_weighting: float = 0.15
    swarm_inertia_decay: float = 0.1
    minimum_swarm_inertia: float = 0.0
    use_coeff_var_fitness: bool = True

    # Timestamping parameters
    reset_peak_separation: int = 40
    final_peak_separation: int = 40
    source_centroid_weighting: float = 0.0
    use_pairwise_silhouette: bool = False
    use_mean_when_clustering: bool = False
    min_peaks_in_source: int = 15
    roa_tolerance_ms: float = 0.5   # ms
    roa_max_shift_ms: int = 30      # ms

    # Plotting and verbosity
    output_source_plot: bool = False
    output_final_source_plot: bool = False
    verbose_mode: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def peel_off_window_size(self) -> int:
        return int(self.peel_off_window_size_ms * self.sampling_frequency / 1000)

    @property
    def roa_tolerance(self) -> int:
        return int(self.roa_tolerance_ms * self.sampling_frequency / 1000)

    @property
    def roa_max_shift(self) -> int:
        return int(self.roa_max_shift_ms * self.sampling_frequency / 1000)
    
@dataclass
class Data:
    emg: torch.Tensor
    starting_exponents: Optional[Sequence[float]] = None
    ica_learning_rate: float = 0.1
    ica_momentum: float = 0.9
    edge_mask_size: int = 200

    def __post_init__(self):
        self.init_all()

    def init_all(self):
        """Initalise all variables except emg"""
        self.edge_mask = None
        self.ica_weights = None
        self.global_best = None
        self.personal_best = None
        self.swarm_inertia = None
        self.exponents = None
        self.swarm_velocities = None
        self.ica_optimiser = None

        self.init_all()

    def init_all(self):
        """Initialise all variables except the EMG data"""

        self.init_swarm()
        self.init_weights()
        self.init_optimiser()
        self.init_edge_mask()

    def init_swarm(self):
        """Initialise the variables associated with the swarm particles"""

        # The exponents are the current positions of the particle swarm
        # Meaning the number of exponents are the number of particles
        if self.starting_exponents is None:
            self.starting_exponents = [2, 3, 4, 5, 6, 7]
        self.exponents = torch.tensor(self.starting_exponents).type_as(self.emg)

        # We also need to track the particle velocities and inertias as
        # they move over the exponent values
        self.swarm_velocities = torch.zeros_like(self.exponents)
        self.swarm_inertia = torch.ones_like(self.exponents)[0]

        # Finally we need to track the personal best of each particle and
        # the global best over all particles
        self.personal_best = {
            "exponents": self.exponents,
            "silhouettes": torch.zeros_like(self.exponents),
            "fitness": torch.zeros_like(self.exponents),
            "spike_heights": torch.zeros_like(self.exponents),
            "spike_means": torch.zeros_like(self.exponents),
            "spike_stds": torch.zeros_like(self.exponents),
            "spike_outliers": torch.zeros_like(self.exponents),
        }
        self.global_best = {
            "exponents": self.exponents[0],
            "fitness": torch.zeros_like(self.exponents)[0],
            "source": None,
            "timestamps": None,
            "silhouette": None,
        }

    def init_weights(self):
        """Initialise the gradient descent ICA separation vector"""

        mean = torch.zeros([self.emg.shape[1], 1]).type_as(self.emg)
        sample = torch.normal(mean, torch.ones_like(mean))
        weights = torch.divide(sample, sample.abs().sum())
        self.ica_weights = torch.nn.Parameter(
            weights.tile([1, self.exponents.shape[0]])
        )

    def init_optimiser(self):
        """Initialise the gradient descent ICA optimiser"""

        self.ica_optimiser = torch.optim.SGD(
            params=[self.ica_weights],
            lr=self.ica_learning_rate,
            momentum=self.ica_momentum,
        )

    def init_edge_mask(self):
        """Initialise a mask to prevent edge effects during optimisation"""

        zeros = torch.zeros([self.edge_mask_size, self.exponents.shape[0]])
        ones = torch.ones([self.emg.shape[0] - 2 * zeros.shape[0], zeros.shape[1]])
        self.edge_mask = torch.concat([zeros, ones, zeros]).type_as(self.emg)
