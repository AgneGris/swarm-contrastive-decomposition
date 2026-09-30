"""The main function for running swarm contrastive decomposition"""

from collections.abc import Callable
from typing import Any, Literal

import torch

from scd.config.structures import Config, Data, set_random_seed
from scd.models.timestamping import (
    bootstrapped_coeff_var,
    calculate_firing_rates,
    find_quality_metric,
    peel_off_source,
    source_to_timestamps,
    spike_triggered_average,
)
from scd.processing.preprocess import (
    autocorrelation_whiten,
    extend,
    high_pass_filter,
    low_pass_filter,
    notch_filter,
    recommended_extension_factor,
    time_differentiate,
    whiten,
)
from scd.utils.plotting import plot_accepted_source, plot_sources

set_random_seed(seed=42)


class SwarmContrastiveDecomposition(torch.nn.Module):
    """
    Class implementing a swarm contrastive decomposition
    """

    def __init__(self) -> None:
        super().__init__()

        self.source_callback: Callable[..., None] | None = None
        self.config: Config
        self.data: Data
        self.decomp: dict[str, Any]
        self.w_mat: torch.Tensor
        self.exponents_list: list[Any]
        self.best_exp_idx_list: list[torch.Tensor]

    def _capture_preprocessing_config(self) -> None:
        """Snapshot the preprocessing parameters into decomp for later replay."""
        self.decomp["preprocessing_config"] = {
            "notch_params": self.config.notch_params,
            "low_pass_cutoff": self.config.low_pass_cutoff,
            "high_pass_cutoff": self.config.high_pass_cutoff,
            "time_differentiate": self.config.time_differentiate,
            "extension_factor": self.config.extension_factor,
            "whitening_method": self.config.whitening_method,
            "autocorrelation_whiten": self.config.autocorrelation_whiten,
            "sampling_frequency": self.config.sampling_frequency,
            "peel_off_window_size": self.config.peel_off_window_size,
            "adapt_clamp": self.config.adapt_clamp,
            "edge_mask_size": self.config.edge_mask_samples,
            "square_sources_spike_det": self.config.square_sources_spike_det,
            # What preprocess_data did to the signal before it reached run():
            # lets an editor holding the signal as loaded (decomp["data"])
            # reproduce the slice and bad-channel fill exactly.
            "bad_channels": list(self.config.bad_channels or []),
            "start_time": self.config.start_time,
            "end_time": self.config.end_time,
        }

    def preprocess_emg(self, emg: torch.Tensor) -> torch.Tensor:
        """Applies preprocessing steps to emg as specified by config"""

        # First apply a notch filter
        if self.config.notch_params is not None:
            sampling_frequency = self.config.require_sampling_frequency()
            emg = notch_filter(
                emg,
                sampling_frequency,
                self.config.notch_params,
                self.config.low_pass_cutoff,
            )

        # Then a low pass
        if self.config.low_pass_cutoff is not None:
            sampling_frequency = self.config.require_sampling_frequency()
            emg = low_pass_filter(
                emg,
                sampling_frequency,
                self.config.low_pass_cutoff,
            )

        # Finally a high pass
        if self.config.high_pass_cutoff is not None:
            sampling_frequency = self.config.require_sampling_frequency()
            emg = high_pass_filter(
                emg,
                sampling_frequency,
                self.config.high_pass_cutoff,
            )

        # Apply time differentiation
        if self.config.time_differentiate:
            emg = time_differentiate(emg)

        # Extend the emg to approx an instantaneous source separation problem
        extension_factor = self.config.extension_factor
        if extension_factor is None:
            raise RuntimeError("extension_factor must be resolved before preprocessing")
        emg = extend(emg, extension_factor)

        # Finally decorrelate the extended emg
        emg, self.w_mat = whiten(emg, self.config.whitening_method, return_matrix=True)

        if self.config.autocorrelation_whiten:
            emg = autocorrelation_whiten(
                emg, extension_factor, self.config.whitening_method
            )

        # Return the emg shape if in verbose mode
        if self.config.verbose_mode:
            print(f"EMG shape: {emg.shape}")

        return emg

    def calculate_sources(self) -> torch.Tensor:
        """Apply separation vectors to emg to get sources"""

        sources = torch.matmul(self.data.emg, self.data.ica_weights)
        sources = (sources - sources.mean(0)) / sources.std(0)

        # Clamp sources to avoid outlying spikes
        if self.config.adapt_clamp:
            for s in range(sources.shape[1]):
                if self.data.personal_best["spike_outliers"][s]:
                    thr = self.data.personal_best["spike_heights"][s]
                    mu = self.data.personal_best["spike_means"][s]
                    std = self.data.personal_best["spike_stds"][s]
                    if torch.isnan(std):
                        std = 0.5
                    sources[sources[:, s] > thr, s] = (
                        mu + torch.randn_like(sources[sources[:, s] > thr, s]) * std
                    )
                else:
                    sources[sources[:, s] > 30, s] = 30
        else:
            sources = sources.clamp(max=30)

        return sources

    def ica_step(self) -> torch.Tensor:
        """Calculate ICA loss with nonlinearity"""

        self.data.ica_optimiser.zero_grad()

        # Get source and edge mask
        sources = self.calculate_sources() * self.data.edge_mask

        # Finally we calculate an asymmetric polynomial loss with exponents
        loss = -torch.stack(
            [
                s.sign() * s.abs().pow(e)
                for s, e in zip(sources.t(), self.data.exponents, strict=True)
            ],
            1,
        ).mean()

        # The sources are independent, so we can update grad on all at once
        loss.backward()
        self.data.ica_optimiser.step()

        return loss.detach()

    def run_ica(self) -> None:
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
        self.data.swarm_inertia = torch.clamp_min(
            self.data.swarm_inertia * self.config.swarm_inertia_decay,
            self.config.minimum_swarm_inertia,
        )

        # Update the global best if fitness is better
        if fitness.max() > self.data.global_best["fitness"]:
            self.data.global_best["exponents"] = self.data.exponents[fitness.argmax()]
            self.data.global_best["fitness"] = fitness.max()

        # Update the personal bests if better
        for idx, (sil, exp) in enumerate(
            zip(fitness, self.data.exponents, strict=True)
        ):
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
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Find the timestamps with a two class k means/median clustering.
        and aggregate. Calculates a fitness function for the swarm."""

        # Square sources for spike detection if specified in config
        if self.config.square_sources_spike_det:
            sources = sources**2

        # Calculate timestamps from k means with associated silhouettes
        timestamps, spike_heights, silhouettes = zip(
            *[
                (
                    source_to_timestamps(
                        s,
                        min_peak_separation,
                        use_pairwise_silhouette=self.config.use_pairwise_silhouette,
                        use_mean=self.config.use_mean_when_clustering,
                    )
                    if s.isfinite().all()
                    else [torch.tensor(0).type_as(s)] * 3
                )
                for s in sources.t()
            ],
            strict=True,
        )

        return list(timestamps), list(spike_heights), list(silhouettes)

    def reset_swarm_and_ica(self) -> None:
        """Does a swarm update and then resets the ICA parameters with STA"""

        # First get the timestamps and silhouettes
        sources = self.calculate_sources()
        timestamps, _spike_heights, silhouettes = self.calculate_timestamps(
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
                for t, f, s in zip(timestamps, fitness, silhouettes, strict=True)
            ],
            strict=True,
        )

        # Convert fitness to tensor
        fitness = torch.stack(fitness)

        # Save the source if new global best of fitness
        if fitness.max() > self.data.global_best["fitness"]:
            self.data.global_best["source"] = sources[:, [fitness.argmax()]]
            self.data.global_best["timestamps"] = timestamps[fitness.argmax()]
            self.data.global_best["silhouette"] = silhouettes[fitness.argmax()]

        if self.config.swarm:
            # Use the fitness to update the swarm particles (the exponents)
            self.swarm_step(fitness)
        else:
            # Use fixed exponent for all sources
            if self.config.fixed_exponent is None:
                raise ValueError("fixed_exponent must be set when swarm is disabled")
            self.data.exponents = torch.full_like(
                self.data.exponents, float(self.config.fixed_exponent)
            )
            # Store in exponents list for tracking
            self.exponents_list.append(
                self.data.exponents.detach().cpu().numpy().copy()
            )
            self.best_exp_idx_list.append(torch.tensor(0))

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

    def initialise_dictionary(self) -> None:
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
            # Ordered record of every peel-off performed during decomposition.
            # Each entry: {
            #   "timestamps":        np.ndarray  (sample indices used for the peel)
            #   "accepted_unit_idx": int | None  (index in decomp["timestamps"] if
            #                                     this was an accepted unit; None for
            #                                     rejected repeats that were still peeled)
            # }
            # Replaying these entries in sequence recreates the exact EMG state
            # that each unit's filter was originally computed on.
            "peel_off_sequence": [],
            "preprocessing_config": {},
        }

    def _peel_and_record(self, peel: bool, accepted_unit_idx: int | None) -> None:
        """Record peel provenance then apply the peel.

        Args:
            source_type:       "good" | "repeat" | "bad"
            peel:              whether a peel should actually be performed
            accepted_unit_idx: the index the current unit has in decomp["timestamps"]
                            if it was accepted (i.e. len(decomp["timestamps"]) - 1
                            after appending), or None for repeats/bad units
        """
        if peel and self.config.peel_off:
            ts = self.data.global_best["timestamps"]
            if ts is None:
                raise RuntimeError("Cannot peel a source without timestamps")

            # Record provenance before mutating self.data.emg
            self.decomp["peel_off_sequence"].append(
                {
                    "timestamps": ts.detach().cpu().numpy().copy(),
                    "accepted_unit_idx": accepted_unit_idx,
                }
            )

            self.data.emg = peel_off_source(
                self.data.emg,
                ts,
                self.config.peel_off_window_size,
            )

    def run(
        self,
        emg: torch.Tensor,
        config: Config | None = None,
        source_callback: Callable[..., None] | None = None,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        """Sets optimiser and runs swarm contrastive decomposition."""

        self.initialise_dictionary()
        self.source_callback = source_callback

        # Initialise dataclasses, preprocessing the emg prior to assignment
        self.config = config if config is not None else Config()

        # Derive the extension factor when it was left unset
        if self.config.extension_factor is None:
            self.config.extension_factor = recommended_extension_factor(
                num_channels=emg.shape[1],
                bad_channels=self.config.bad_channels,
            )
            if self.config.verbose_mode:
                print(f"Extension factor: {self.config.extension_factor}")

        if not self.config.swarm:
            if self.config.fixed_exponent is None:
                raise ValueError("fixed_exponent must be set when swarm is disabled")
            starting_exponents = [float(self.config.fixed_exponent)]
        else:
            starting_exponents = self.config.starting_exponents

        self.data = Data(
            emg=self.preprocess_emg(emg),
            starting_exponents=starting_exponents,
            ica_learning_rate=self.config.ica_learning_rate,
            ica_momentum=self.config.ica_momentum,
            edge_mask_size=self.config.edge_mask_samples,
            electrode=self.config.electrode,
        )

        self.decomp["w_mat"] = self.w_mat.cpu().numpy().copy()

        # Snapshot preprocessing config for filter recalculation in the editor
        self._capture_preprocessing_config()

        # Finally run swarm contrastive decomposition with source checking
        patience = 0
        library: list[torch.Tensor] = []
        sampling_frequency = self.config.require_sampling_frequency()
        for iteration in range(self.config.max_iterations):
            # First run a swarm contrastive decomposition for a single source
            self.exponents_list = []
            self.best_exp_idx_list = []

            self.scd_step()

            # Categorise the source as good, bad or repeat
            best_silhouette = self.data.global_best["silhouette"]
            best_timestamps = self.data.global_best["timestamps"]
            source_type: Literal["good", "repeat", "bad"]
            if (
                best_silhouette is not None
                and best_timestamps is not None
                and best_silhouette > self.config.acceptance_silhouette
            ):
                # Find the highest rates of agreement with found sources
                max_roa = (
                    max(
                        find_quality_metric(
                            best_timestamps,
                            t,
                            "roa",
                            self.config.roa_tolerance,
                            self.config.roa_max_shift,
                        )
                        for t in library
                    )
                    if len(library) > 0
                    else 0.0
                )

                source_type = (
                    "good" if max_roa < self.config.acceptance_max_roa else "repeat"
                )

                fr = calculate_firing_rates(
                    best_timestamps,
                    window_size_in_seconds=1,
                    fsamp2=sampling_frequency,
                )
                if self.config.remove_bad_fr and (fr < 2 or fr > 100):
                    source_type = "bad"
            else:
                source_type = "bad"

            # ── Handle accepted / repeat / bad
            if source_type == "good":
                patience = 0
                message = str(iteration) + ": accept new source."
                peel = True

                source = self.data.global_best["source"]
                timestamps = self.data.global_best["timestamps"]
                silhouette = self.data.global_best["silhouette"]
                if source is None or timestamps is None or silhouette is None:
                    raise RuntimeError(
                        "Accepted source is missing decomposition results"
                    )
                library.append(timestamps)

                if self.source_callback:
                    self.source_callback(
                        source=source.detach().cpu(),
                        timestamps=timestamps.detach().cpu(),
                        iteration=iteration,
                        silhouette=silhouette.item(),
                    )

                if self.config.output_final_source_plot:
                    plot_accepted_source(source, timestamps)

                self.decomp["silhouettes"].append(silhouette)
                self.decomp["timestamps"].append(timestamps)
                self.decomp["fr"].append(
                    calculate_firing_rates(
                        timestamps,
                        window_size_in_seconds=1,
                        fsamp2=sampling_frequency,
                    )
                )
                self.decomp["cov"].append(bootstrapped_coeff_var(timestamps))
                best_idx = int(self.best_exp_idx_list[-1].item())
                self.decomp["best_exp"].append(self.exponents_list[-1][best_idx])
                self.decomp["filters"].append(
                    self.data.ica_weights.detach().cpu().numpy().copy()[:, [best_idx]]
                )
                self.decomp["source"].append(source.detach().cpu().numpy().copy())

                # Index of this unit in decomp["timestamps"] (just appended above)
                accepted_unit_idx = len(self.decomp["timestamps"]) - 1

            elif source_type == "repeat":
                patience += 1
                message = str(iteration) + ": reject repeat source."
                peel = bool(self.config.peel_off_repeats)
                accepted_unit_idx = None  # repeat — no entry in results

            else:
                patience += 1
                message = str(iteration) + ": reject low silhouette source."
                peel = False
                accepted_unit_idx = None

            # ── Peel off (records provenance + mutates self.data.emg)
            self._peel_and_record(peel, accepted_unit_idx)

            if self.config.verbose_mode:
                print(message)

            if patience == self.config.iteration_patience:
                break

        return library, self.decomp
