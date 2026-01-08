"""The main function for running swarm contrastive decomposition"""

from typing import Optional, List, Tuple, Dict
import torch

from scd.config.structures import Config, Data
from scd.processing.preprocess import (
    whiten,
    autocorrelation_whiten,
    extend,
    time_differentiate,
    notch_filter,
    low_pass_filter,
    high_pass_filter,
)
from scd.models.timestamping import (
    source_to_timestamps,
    spike_triggered_average,
    peel_off_source,
    find_quality_metric,
    bootstrapped_coeff_var,
    calculate_firing_rates,
)

from scd.utils.plotting import plot_accepted_source, plot_sources
from scd.config.structures import set_random_seed

set_random_seed(seed=42)


class SwarmContrastiveDecomposition(torch.nn.Module):
    """
    Class implementing a swarm contrastive decomposition
    """

    def __init__(self):
        super().__init__()

    def preprocess_emg(self, emg: torch.Tensor) -> torch.Tensor:
        """Applies preprocessing steps to emg as specified by config"""

        # First apply a notch filter
        if self.config.notch_params is not None:
            assert (
                self.config.sampling_frequency is not None
            ), "Sampling frequency must be set in config if filtering."
            emg = notch_filter(
                emg,
                self.config.sampling_frequency,
                self.config.notch_params,
                self.config.low_pass_cutoff,
            )

        # Then a low pass
        if self.config.low_pass_cutoff is not None:
            assert (
                self.config.sampling_frequency is not None
            ), "Sampling frequency must be set in config if filtering."
            emg = low_pass_filter(
                emg,
                self.config.sampling_frequency,
                self.config.low_pass_cutoff,
            )

        # Finally a high pass
        if self.config.high_pass_cutoff is not None:
            assert (
                self.config.sampling_frequency is not None
            ), "Sampling frequency must be set in config if filtering."
            emg = high_pass_filter(
                emg,
                self.config.sampling_frequency,
                self.config.high_pass_cutoff,
            )

        # Apply time differentiation
        if self.config.time_differentiate:
            emg = time_differentiate(emg)

        # Extend the emg to approx an instantaneous source separation problem
        emg = extend(emg, self.config.extension_factor)

        # Finally decorrelate the extended emg
        emg = whiten(emg, self.config.whitening_method)

        if self.config.autocorrelation_whiten:
            emg = autocorrelation_whiten(
                emg, self.config.extension_factor, self.config.whitening_method
            )

        # Return the emg shape if in verbose mode
        if self.config.verbose_mode:
            print(f"EMG shape: {emg.shape}")

        return emg

    def calculate_sources(self) -> torch.Tensor:
        """Apply separation vectors to emg to get sources"""

        sources = torch.matmul(self.data.emg, self.data.ica_weights)
        sources = (sources - sources.mean(0)) / sources.std(0)

        # Clamp sources to avoid outlying spikes
        if self.config.clamp_percentile:

            if torch.all(self.data.personal_best['spike_heights'] == 0):
                sources = sources.clamp(max=30)
            else:
                for s in range(sources.shape[1]):
                    if self.data.personal_best['spike_outliers'][s]:
                        thr = self.data.personal_best['spike_heights'][s]
                        mu = self.data.personal_best['spike_means'][s]
                        std = self.data.personal_best['spike_stds'][s]
                        if torch.isnan(std):
                            std = 0.5
                        sources[sources[:,s] > thr, s] = mu + torch.randn_like(sources[sources[:,s] > thr, s]) * std
                    else:
                        sources[sources[:,s] > 30, s] = 30
            
        else:
            sources = sources.clamp(max=30)

        return sources

    def ica_step(self):
        """Calculate ICA loss with nonlinearity"""

        self.data.ica_optimiser.zero_grad()

        # Get source and edge mask
        sources = self.calculate_sources() * self.data.edge_mask

        # Finally we calculate an asymmetric polynomial loss with exponents
        loss = -torch.stack(
            [
                s.sign() * s.abs().pow(e)
                for s, e in zip(sources.t(), self.data.exponents)
            ],
            1,
        ).mean()

        # The sources are independent, so we can update grad on all at once
        loss.backward()
        self.data.ica_optimiser.step()

        return loss.detach()

    def run_ica(self):
        """Runs a single source independent component analysis"""

        patience = 0
        history = torch.empty(0).type_as(self.data.emg)
        for _ in range(self.config.max_ica_steps):
            loss = self.ica_step()

            # End ica run if loss is no longer improving
            if (loss < history).all():
                patience = 0
            else:
                patience += 1
                if patience == self.config.ica_patience:
                    break

            history = torch.concatenate([history, loss.unsqueeze(0)])

    def swarm_step(self, fitness: torch.Tensor) -> None:
        """Updates records on global and personal bests of the swarm particles
        using the input fitness values.
        """

        # First decay the inertia
        self.data.swarm_inertia = max(
            self.data.swarm_inertia * self.config.swarm_inertia_decay,
            self.config.minimum_swarm_inertia,
        )

        # Update the global best if fitness is better
        if fitness.max() > self.data.global_best["fitness"]:
            self.data.global_best["exponents"] = self.data.exponents[fitness.argmax()]
            self.data.global_best["fitness"] = fitness.max()

        # Update the personal bests if better
        for idx, [sil, exp] in enumerate(zip(fitness, self.data.exponents)):
            if sil > self.data.personal_best["fitness"][idx]:
                self.data.personal_best["exponents"][idx] = exp
                self.data.personal_best["fitness"][idx] = sil

        # Calculate the global and personal pulls on the particles
        pers_pull = self.data.personal_best["exponents"] - self.data.exponents
        glob_pull = self.data.global_best["exponents"] - self.data.exponents

        # Get momentum, personal and global terms of velocity update
        i_term = self.data.swarm_velocities * self.data.swarm_inertia
        p_term = self.config.personal_weighting * pers_pull
        g_term = self.config.global_weighting * glob_pull

        # Add exploration noise to the personal and global terms
        p_term += torch.normal(
            mean=p_term,
            std=self.config.swarm_step_std_personal,
        )
        g_term += (
            torch.normal(
                mean=g_term,
                std=self.config.swarm_step_std_global,
            )
            if self.data.global_best["fitness"] != 0
            else 0
        )

        # Finally update velocities and exponents
        self.data.swarm_velocities = i_term + p_term + g_term
        self.data.exponents = self.data.exponents + self.data.swarm_velocities
        self.exponents_list.append(self.data.exponents.detach().cpu().numpy().copy())
        self.best_exp_idx_list.append(fitness.argmax())

    def calculate_timestamps(
        self, sources: torch.Tensor, min_peak_separation: int
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Find the timestamps with a two class k means/median clustering.
        and aggregate. Calculates a fitness function for the swarm."""

        # Calculate timestamps from k means with associated silhouettes
        timestamps, spike_heights, silhouettes = zip(
            *[
                (
                    source_to_timestamps(
                        s,
                        min_peak_separation,
                    )
                    if s.isfinite().all()
                    else [torch.tensor(0).type_as(s)] * 2
                )
                for s in sources.t()
            ]
        )

        return timestamps, spike_heights, silhouettes

    def reset_swarm_and_ica(self) -> None:
        """Does a swarm update and then resets the ICA parameters with STA"""

        # First get the timestamps and silhouettes
        sources = self.calculate_sources()
        timestamps, spike_heights, silhouettes = self.calculate_timestamps(
            sources, self.config.reset_peak_separation
        )

        # Use bootstrapped coeff var for fitness if selected
        if self.config.use_coeff_var_fitness:
            fitness = []
            for t in timestamps:

                if t.numel() < 2:
                    fitness.append(torch.tensor(0).type_as(t))
                else:
                    isi = t.diff().float()
                    num_intervals = (isi < 5 * isi.median()).sum()

                    fitness.append(
                        1 - bootstrapped_coeff_var(t)
                        if num_intervals > 2
                        else torch.tensor(0).type_as(t)
                    )
        else:
            fitness = silhouettes

        # Make nans and low timestamps zeros in fitness and silhouettes
        fitness, silhouettes = zip(
            *[
                (
                    [f, s]
                    if t.nelement() > self.config.min_peaks_in_source
                    else [torch.zeros_like(f)] * 2
                )
                for t, f, s in zip(timestamps, fitness, silhouettes)
            ]
        )

        # Convert fitness to tensor
        fitness = torch.stack(fitness)

        # Save the source if new global best of fitness
        if fitness.max() > self.data.global_best["fitness"]:
            self.data.global_best["source"] = sources[:, [fitness.argmax()]]
            self.data.global_best["timestamps"] = timestamps[fitness.argmax()]
            self.data.global_best["silhouette"] = silhouettes[fitness.argmax()]

        # Use the fitness to update the swarm particles (the exponents)
        self.swarm_step(fitness)

        # Use the timestamps with the best fitness to update the weights by STA
        sample = spike_triggered_average(
            self.data.emg, timestamps[fitness.argmax()]
        ).t()
        weights = sample.divide(sample.abs().sum())
        self.data.ica_weights = torch.nn.Parameter(
            weights.tile([1, self.data.exponents.shape[0]])
        )

        # Reinitialise the optimiser with the updated weights
        self.data.init_optimiser()

        # # Finally output a source plot if in verbose mode
        if self.config.output_source_plot:
            plot_sources(sources, timestamps, self.data.exponents, fitness)

    def scd_step(self) -> None:
        """Runs a swarm contrastive decomposition to find a single source"""

        self.data.init_all()

        patience = 0
        history = torch.empty(0).type_as(self.data.emg)
        for _ in range(self.config.max_swarm_steps):
            self.run_ica()
            self.reset_swarm_and_ica()

            if (self.data.global_best["fitness"] < history).all():
                patience = 0
            else:
                patience += 1
                if patience == self.config.swarm_patience:
                    break

            history = torch.concatenate(
                [history, self.data.global_best["fitness"].unsqueeze(0)]
            )

    def initialise_dictionary(self):
        # Initialise empty dictionary to store results
        self.decomp = {
            "silhouettes": [],
            "timestamps": [],
            "source": [],
            "RoA": [],
            "filters": [],
            "fr": [],
            "cov": [],
            "best_exp": [],
            "source": [],
        }

    def run(
        self,
        emg: torch.Tensor,
        config: Optional[Config] = None,
    ) -> Tuple[List[torch.Tensor], Dict]:
        """Sets optimiser and runs"""

        self.initialise_dictionary()

        # Initialise dataclasses, preprocessing the emg prior to assignment
        self.config = config if config is not None else Config()
        self.data = Data(
            emg=self.preprocess_emg(emg),
            starting_exponents=self.config.starting_exponents,
            ica_learning_rate=self.config.ica_learning_rate,
            ica_momentum=self.config.ica_momentum,
            edge_mask_size=self.config.edge_mask_size,
        )

        # Finally run swarm contrastive decomposition with source checking
        patience = 0
        library = []
        for iteration in range(self.config.max_iterations):
            # First run a swarm contrastive decomposition for a single source
            self.exponents_list = []
            self.best_exp_idx_list = []

            self.scd_step()

            # Categorise the source as good, bad or repeat
            if ((self.data.global_best["silhouette"]) is not None) and (
                self.data.global_best["silhouette"] > self.config.acceptance_silhouette
            ):

                # Find the highest rates of agreement with found sources
                max_roa = (
                    max(
                        [
                            find_quality_metric(
                                self.data.global_best["timestamps"],
                                t,
                                "roa",
                                self.config.roa_tolerance,
                                self.config.roa_max_shift,
                            )
                            for t in library
                        ]
                    )
                    if len(library) > 0
                    else 0.0
                )

                source_type = (
                    "good" if max_roa < self.config.acceptance_max_roa else "repeat"
                )

                fr = calculate_firing_rates(
                    self.data.global_best["timestamps"],
                    window_size_in_seconds=1,
                    fsamp2=self.config.sampling_frequency,
                )
                if self.config.remove_bad_fr:
                    if fr < 2 or fr > 100:
                        source_type = "bad"
            else:
                source_type = "bad"

            if source_type == "good":
                patience = 0
                message = str(iteration) + ": accept new source."
                peel = True

                timestamp_list = [self.data.global_best["timestamps"]]

                for timestamps in timestamp_list:
                    library.append(timestamps)

                if self.config.output_final_source_plot:
                    plot_accepted_source(self.data.global_best["source"], timestamps)

                self.decomp["silhouettes"].append(self.data.global_best["silhouette"])
                self.decomp["timestamps"].append(timestamps)
                self.decomp["fr"].append(
                    calculate_firing_rates(
                        timestamps,
                        window_size_in_seconds=1,
                        fsamp2=self.config.sampling_frequency,
                    )
                )
                self.decomp["cov"].append(bootstrapped_coeff_var(timestamps))
                self.decomp["best_exp"].append(
                    self.exponents_list[-1][self.best_exp_idx_list[-1].item()]
                )
                self.decomp["filters"].append(
                    self.data.ica_weights.detach()
                    .cpu()
                    .numpy()
                    .copy()[:, [self.best_exp_idx_list[-1].item()]]
                )
                self.decomp["source"].append(
                    self.data.global_best["source"].detach().cpu().numpy().copy()
                )

            elif source_type == "repeat":
                patience += 1
                message = str(iteration) + ": reject repeat source."
                peel = True if self.config.peel_off_repeats else False
            elif source_type == "bad":
                patience += 1
                message = str(iteration) + ": reject low silhouette source."
                peel = False

            if peel:
                self.data.emg = peel_off_source(
                    self.data.emg,
                    self.data.global_best["timestamps"],
                    self.config.peel_off_window_size,
                )

            # Print the message if mode is verbose
            if self.config.verbose_mode:
                print(message)

            # Finish finding sources if patience is broken
            if patience == self.config.iteration_patience:
                break

        return library, self.decomp
