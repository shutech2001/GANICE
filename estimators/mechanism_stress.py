from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray

from .utils import ensure_row_matrix, make_rng


TargetShiftSet = Literal["hard", "negative_control", "H", "H0"]


@dataclass(frozen=True, slots=True)
class MechanismStressConfig:
    """Configuration for the pre-specified finite-state stress benchmark.

    The paper sensitivity runner sets its nominal design explicitly.
    ``partition_bits`` affects only partition diagnostics; it never changes
    the generated observations or
    potential outcomes.
    """

    conditioning_dim: int = 4
    max_conditioning_dim: int = 8
    master_observed_size: int = 8192
    overlap_kappa: float = 1.0
    target_shift: float = 0.0
    heterogeneity: float = 0.5
    partition_bits: Optional[int] = None
    target_shift_set: TargetShiftSet = "hard"
    seed: int = 42


class MechanismStressDGP:
    """Exact binary-state DGP for the mechanism-aligned GANICE stress test.

    The predictive covariate is ``Z in {0, 1}^d`` and the treatment is binary.
    The implementation keeps separate random streams for covariates, latent
    potential outcomes, and treatment assignment.  Consequently sparse-sample
    calls are nested prefixes, target shift cannot alter the observed sample,
    and changing overlap changes assignment but not either potential outcome.
    """

    outcome_lower = -3.0
    outcome_upper = 3.0
    noise_variance = 0.45**2

    def __init__(self, config: MechanismStressConfig) -> None:
        if config.conditioning_dim < 2:
            raise ValueError("conditioning_dim must be at least two")
        if config.max_conditioning_dim < config.conditioning_dim:
            raise ValueError(
                "max_conditioning_dim must be no smaller than conditioning_dim"
            )
        if config.master_observed_size < 1:
            raise ValueError("master_observed_size must be positive")
        if not (0.0 < config.overlap_kappa <= 1.0):
            raise ValueError("overlap_kappa must lie in (0, 1]")
        if not (0.0 <= config.target_shift < 1.0):
            raise ValueError("target_shift must lie in [0, 1)")
        if not (0.0 <= config.heterogeneity <= 1.0):
            raise ValueError("heterogeneity must lie in [0, 1]")
        if config.target_shift_set not in (
            "hard",
            "negative_control",
            "H",
            "H0",
        ):
            raise ValueError(
                "target_shift_set must be 'hard'/'H' or "
                "'negative_control'/'H0'"
            )
        partition_bits = (
            config.conditioning_dim
            if config.partition_bits is None
            else config.partition_bits
        )
        if not (0 <= partition_bits <= config.conditioning_dim):
            raise ValueError(
                "partition_bits must lie between zero and conditioning_dim"
            )
        self.config = config

    @property
    def feature_dim(self) -> int:
        return self.config.conditioning_dim

    @property
    def x_dim(self) -> int:
        """Alias used by the existing synthetic benchmark adapters."""

        return self.feature_dim

    @property
    def d_w(self) -> int:
        return self.feature_dim + 1

    @property
    def partition_bits(self) -> int:
        if self.config.partition_bits is None:
            return self.feature_dim
        return self.config.partition_bits

    @property
    def num_covariate_states(self) -> int:
        return 2**self.feature_dim

    @property
    def num_joint_states(self) -> int:
        return 2 * self.num_covariate_states

    @staticmethod
    def _validate_n(n: int) -> int:
        n_int = int(n)
        if n_int != n or n_int < 1:
            raise ValueError("n must be a positive integer")
        return n_int

    def _validate_z(self, z: NDArray) -> NDArray[np.int64]:
        z_array = ensure_row_matrix(z)
        if z_array.shape[1] != self.feature_dim:
            raise ValueError(
                f"z must have {self.feature_dim} columns, got {z_array.shape[1]}"
            )
        if not np.all(np.isfinite(z_array)):
            raise ValueError("z must be finite")
        if not np.all(np.isin(z_array, [0.0, 1.0])):
            raise ValueError("z must contain only zero and one")
        return z_array.astype(np.int64)

    @staticmethod
    def _validate_treatment(
        treatment: NDArray | int,
        n_rows: int,
    ) -> NDArray[np.int64]:
        raw_treatment = np.asarray(treatment).reshape(-1)
        if (
            np.any(~np.isfinite(raw_treatment))
            or not np.all(np.isin(raw_treatment, [0, 1]))
        ):
            raise ValueError("treatment must contain only zero and one")
        treatment_array = raw_treatment.astype(np.int64)
        if treatment_array.size == 1:
            treatment_array = np.full(
                n_rows,
                int(treatment_array.item()),
                dtype=np.int64,
            )
        if treatment_array.shape[0] != n_rows:
            raise ValueError("treatment must be scalar or align with z")
        return treatment_array

    def encode_w(
        self,
        z: NDArray,
        treatment: NDArray | int,
    ) -> NDArray[np.float32]:
        """Encode the exact joint state ``W=(Z,T)`` without jittering bits."""

        z_array = self._validate_z(z)
        treatment_array = self._validate_treatment(treatment, z_array.shape[0])
        return np.column_stack([z_array, treatment_array]).astype(np.float32)

    def enumerate_covariate_states(self) -> NDArray[np.int64]:
        """Return all binary vectors in fixed lexicographic order."""

        indices = np.arange(self.num_covariate_states, dtype=np.int64)
        shifts = np.arange(self.feature_dim - 1, -1, -1, dtype=np.int64)
        return ((indices[:, None] >> shifts[None, :]) & 1).astype(np.int64)

    @staticmethod
    def walsh_c1(z: NDArray) -> NDArray[np.int64]:
        z_array = np.asarray(z, dtype=np.int64)
        if z_array.ndim == 1:
            z_array = z_array[None, :]
        parity = np.mod(np.sum(z_array, axis=1), 2)
        return np.where(parity == 0, 1, -1).astype(np.int64)

    @staticmethod
    def walsh_c2(z: NDArray) -> NDArray[np.int64]:
        z_array = np.asarray(z, dtype=np.int64)
        if z_array.ndim == 1:
            z_array = z_array[None, :]
        weights = np.arange(1, z_array.shape[1] + 1, dtype=np.int64)
        parity = np.mod(z_array @ weights, 2)
        return np.where(parity == 0, 1, -1).astype(np.int64)

    @staticmethod
    def hard_cell_mask(z: NDArray) -> NDArray[np.bool_]:
        z_array = np.asarray(z, dtype=np.int64)
        if z_array.ndim == 1:
            z_array = z_array[None, :]
        if z_array.shape[1] < 2:
            raise ValueError("the hard-cell definition requires at least two bits")
        return (z_array[:, 0] == 1) & (z_array[:, 1] == 1)

    @staticmethod
    def negative_control_mask(z: NDArray) -> NDArray[np.bool_]:
        z_array = np.asarray(z, dtype=np.int64)
        if z_array.ndim == 1:
            z_array = z_array[None, :]
        if z_array.shape[1] < 2:
            raise ValueError(
                "the negative-control definition requires at least two bits"
            )
        return (z_array[:, 0] == 0) & (z_array[:, 1] == 0)

    def target_shift_mask(self, z: NDArray) -> NDArray[np.bool_]:
        if self.config.target_shift_set in ("negative_control", "H0"):
            return self.negative_control_mask(z)
        return self.hard_cell_mask(z)

    def covariate_state_index(self, z: NDArray) -> NDArray[np.int64]:
        z_array = self._validate_z(z)
        powers = 2 ** np.arange(
            self.feature_dim - 1,
            -1,
            -1,
            dtype=np.int64,
        )
        return (z_array @ powers).astype(np.int64)

    def joint_state_index(
        self,
        z: NDArray,
        treatment: NDArray | int,
    ) -> NDArray[np.int64]:
        z_array = self._validate_z(z)
        treatment_array = self._validate_treatment(treatment, z_array.shape[0])
        return 2 * self.covariate_state_index(z_array) + treatment_array

    def partition_state_index(
        self,
        z: NDArray,
        treatment: NDArray | int,
        *,
        partition_bits: Optional[int] = None,
    ) -> NDArray[np.int64]:
        """Index ``C_r(z,t)=(z_1,...,z_r,t)`` with treatment always locked."""

        z_array = self._validate_z(z)
        treatment_array = self._validate_treatment(treatment, z_array.shape[0])
        r = self.partition_bits if partition_bits is None else int(partition_bits)
        if not (0 <= r <= self.feature_dim):
            raise ValueError(
                "partition_bits must lie between zero and conditioning_dim"
            )
        if r == 0:
            prefix_index = np.zeros(z_array.shape[0], dtype=np.int64)
        else:
            powers = 2 ** np.arange(r - 1, -1, -1, dtype=np.int64)
            prefix_index = z_array[:, :r] @ powers
        return (2 * prefix_index + treatment_array).astype(np.int64)

    def propensity(self, z: NDArray) -> NDArray[np.float64]:
        """Return the exact observational probability ``P(T=1|Z=z)``."""

        z_array = self._validate_z(z)
        preferred = (self.walsh_c1(z_array) == 1).astype(np.int64)
        rare_probability = 0.5 * self.config.overlap_kappa
        return np.where(
            preferred == 1,
            rare_probability,
            1.0 - rare_probability,
        ).astype(np.float64)

    def observed_covariate_probability(self, z: NDArray) -> NDArray[np.float64]:
        z_array = self._validate_z(z)
        return np.full(
            z_array.shape[0],
            1.0 / self.num_covariate_states,
            dtype=np.float64,
        )

    def target_covariate_probability(self, z: NDArray) -> NDArray[np.float64]:
        z_array = self._validate_z(z)
        delta = self.config.target_shift
        base = (1.0 - delta) / self.num_covariate_states
        shifted = delta / (self.num_covariate_states // 4)
        return (
            base + shifted * self.target_shift_mask(z_array).astype(np.float64)
        )

    def observed_state_probability(
        self,
        z: NDArray,
        treatment: NDArray | int,
    ) -> NDArray[np.float64]:
        z_array = self._validate_z(z)
        treatment_array = self._validate_treatment(treatment, z_array.shape[0])
        p_one = self.propensity(z_array)
        p_treatment = np.where(treatment_array == 1, p_one, 1.0 - p_one)
        return self.observed_covariate_probability(z_array) * p_treatment

    def target_state_probability(
        self,
        z: NDArray,
        treatment: NDArray | int,
    ) -> NDArray[np.float64]:
        z_array = self._validate_z(z)
        self._validate_treatment(treatment, z_array.shape[0])
        return 0.5 * self.target_covariate_probability(z_array)

    def target_to_observed_ratio(
        self,
        z: NDArray,
        treatment: NDArray | int,
    ) -> NDArray[np.float64]:
        q = self.target_state_probability(z, treatment)
        pi = self.observed_state_probability(z, treatment)
        return q / pi

    # Compatibility with estimators whose sampler API uses "density ratio".
    target_density_ratio = target_to_observed_ratio

    def enumerate_target_states(self) -> dict[str, NDArray]:
        """Enumerate all exact states and their ``q``, ``pi``, and ratios."""

        covariate_states = self.enumerate_covariate_states()
        z = np.repeat(covariate_states, 2, axis=0)
        treatment = np.tile(np.asarray([0, 1], dtype=np.int64), len(covariate_states))
        q = self.target_state_probability(z, treatment)
        pi = self.observed_state_probability(z, treatment)
        return {
            "state_id": self.joint_state_index(z, treatment),
            "z": z,
            "x": z.astype(np.float32),
            "t": treatment,
            "w": self.encode_w(z, treatment),
            "q": q,
            "pi": pi,
            "q_over_pi": q / pi,
        }

    # A neutral alias is convenient for audits comparing both designs.
    enumerate_states = enumerate_target_states

    def target_state_masses(self) -> NDArray[np.float64]:
        return self.enumerate_target_states()["q"].copy()

    target_state_weights = target_state_masses

    def observed_state_masses(self) -> NDArray[np.float64]:
        return self.enumerate_target_states()["pi"].copy()

    def exact_overlap_constant(self) -> float:
        table = self.enumerate_target_states()
        target_positive = table["q"] > 0.0
        return float(np.min(table["pi"][target_positive] / table["q"][target_positive]))

    overlap_lower_bound = exact_overlap_constant

    def exact_importance_ess_ratio(self) -> float:
        table = self.enumerate_target_states()
        second_moment = np.sum(table["q"] ** 2 / table["pi"])
        return float(1.0 / second_moment)

    target_design_ess_ratio = exact_importance_ess_ratio

    def _full_binary_master(self, n: int, seed: int) -> NDArray[np.int64]:
        rng = make_rng(seed)
        full = rng.integers(
            0,
            2,
            size=(n, self.config.max_conditioning_dim),
            dtype=np.int64,
        )
        return full[:, : self.feature_dim]

    @staticmethod
    def _latent_components(
        u1: NDArray[np.float64],
        u2: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        a = np.sqrt(2.0) * np.sin(2.0 * np.pi * u1)
        b = (4.0 / np.sqrt(5.0)) * np.sin(2.0 * np.pi * u2) ** 3
        return a, b

    def _state_terms(
        self,
        z: NDArray,
        treatment: NDArray | int,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        z_array = self._validate_z(z)
        treatment_array = self._validate_treatment(treatment, z_array.shape[0])
        c1 = self.walsh_c1(z_array).astype(np.float64)
        c2 = self.walsh_c2(z_array).astype(np.float64)
        hard = self.hard_cell_mask(z_array).astype(np.float64)
        baseline = 0.4 * c1 + 0.2 * c2
        mean = baseline + treatment_array * (
            0.8 + 0.35 * self.config.heterogeneity * c1
        )
        theta_0 = np.pi / 12.0 + (np.pi / 4.0) * hard
        delta = (np.pi / 12.0) * (1.0 + c2)
        theta = (
            theta_0
            + self.config.heterogeneity * treatment_array * delta
        )
        return mean, theta, delta

    def potential_from_latents(
        self,
        z: NDArray,
        treatment: NDArray | int,
        u1: NDArray,
        u2: NDArray,
    ) -> NDArray[np.float64]:
        """Evaluate the smooth latent-pushforward map without clipping."""

        z_array = self._validate_z(z)
        treatment_array = self._validate_treatment(treatment, z_array.shape[0])
        u1_array = np.asarray(u1, dtype=np.float64).reshape(-1)
        u2_array = np.asarray(u2, dtype=np.float64).reshape(-1)
        if u1_array.size != z_array.shape[0] or u2_array.size != z_array.shape[0]:
            raise ValueError("u1 and u2 must align with z")
        if (
            np.any(~np.isfinite(u1_array))
            or np.any(~np.isfinite(u2_array))
            or np.any((u1_array < 0.0) | (u1_array > 1.0))
            or np.any((u2_array < 0.0) | (u2_array > 1.0))
        ):
            raise ValueError("u1 and u2 must lie in [0, 1]")
        a, b = self._latent_components(u1_array, u2_array)
        mean, theta, _ = self._state_terms(z_array, treatment_array)
        return mean + 0.45 * (np.cos(theta) * a + np.sin(theta) * b)

    def potential_means(self, z: NDArray) -> NDArray[np.float32]:
        z_array = self._validate_z(z)
        baseline = (
            0.4 * self.walsh_c1(z_array)
            + 0.2 * self.walsh_c2(z_array)
        ).astype(np.float64)
        effect = (
            0.8
            + 0.35
            * self.config.heterogeneity
            * self.walsh_c1(z_array)
        )
        return np.column_stack([baseline, baseline + effect]).astype(np.float32)

    def sample_potential(
        self,
        z: NDArray,
        treatment: NDArray | int,
        n_per_x: int = 1,
        seed: Optional[int] = None,
    ) -> NDArray[np.float32]:
        z_array = self._validate_z(z)
        n_per_x = self._validate_n(n_per_x)
        treatment_array = self._validate_treatment(treatment, z_array.shape[0])
        base_seed = self.config.seed + 30_000 if seed is None else int(seed)
        rng = make_rng(base_seed)
        u1 = rng.uniform(size=(z_array.shape[0], n_per_x))
        u2 = rng.uniform(size=(z_array.shape[0], n_per_x))
        repeated_z = np.repeat(z_array, n_per_x, axis=0)
        repeated_treatment = np.repeat(treatment_array, n_per_x)
        values = self.potential_from_latents(
            repeated_z,
            repeated_treatment,
            u1.reshape(-1),
            u2.reshape(-1),
        )
        return values.reshape(z_array.shape[0], n_per_x, 1).astype(np.float32)

    def sample_observed(
        self,
        n: int,
        seed: Optional[int] = None,
    ) -> dict[str, NDArray]:
        """Sample a prefix of a deterministic master observational realization."""

        n = self._validate_n(n)
        base_seed = self.config.seed if seed is None else int(seed)
        master_size = max(n, self.config.master_observed_size)
        z_master = self._full_binary_master(master_size, base_seed + 101)

        # Generate all potential-outcome randomness before assignment.
        latent_rng = make_rng(base_seed + 211)
        u1_master = latent_rng.uniform(size=master_size)
        u2_master = latent_rng.uniform(size=master_size)
        y0_master = self.potential_from_latents(
            z_master,
            0,
            u1_master,
            u2_master,
        )
        y1_master = self.potential_from_latents(
            z_master,
            1,
            u1_master,
            u2_master,
        )

        propensity_master = self.propensity(z_master)
        assignment_uniform_master = make_rng(base_seed + 307).uniform(
            size=master_size
        )
        treatment_master = (
            assignment_uniform_master < propensity_master
        ).astype(np.int64)
        factual_master = np.where(
            treatment_master == 1,
            y1_master,
            y0_master,
        )

        z = z_master[:n]
        treatment = treatment_master[:n]
        pi = self.observed_state_probability(z, treatment)
        q = self.target_state_probability(z, treatment)
        return {
            "z": z,
            "x": z.astype(np.float32),
            "t": treatment,
            "w": self.encode_w(z, treatment),
            "y": factual_master[:n, None].astype(np.float32),
            "y0": y0_master[:n, None].astype(np.float32),
            "y1": y1_master[:n, None].astype(np.float32),
            "u1": u1_master[:n].astype(np.float64),
            "u2": u2_master[:n].astype(np.float64),
            "assignment_uniform": assignment_uniform_master[:n].astype(
                np.float64
            ),
            "propensity": propensity_master[:n].astype(np.float64),
            "state_id": self.joint_state_index(z, treatment),
            "pi": pi,
            "q": q,
            "q_over_pi": q / pi,
        }

    def _sample_target_z(self, n: int, seed: int) -> NDArray[np.int64]:
        n = self._validate_n(n)
        # A common mixture construction couples all target-shift levels while
        # preserving the exact (1-delta) Uniform + delta Uniform(H) law.
        uniform_z = self._full_binary_master(n, seed + 401)
        shifted_z = self._full_binary_master(n, seed + 409)
        if self.config.target_shift_set in ("negative_control", "H0"):
            shifted_z[:, :2] = 0
        else:
            shifted_z[:, :2] = 1
        mixture_uniform = make_rng(seed + 419).uniform(size=n)
        use_shifted = mixture_uniform < self.config.target_shift
        return np.where(use_shifted[:, None], shifted_z, uniform_z)

    def sample_target_x(
        self,
        n: int,
        seed: Optional[int] = None,
    ) -> NDArray[np.float32]:
        base_seed = self.config.seed + 10_000 if seed is None else int(seed)
        return self._sample_target_z(n, base_seed).astype(np.float32)

    def sample_target(
        self,
        n: int,
        seed: Optional[int] = None,
    ) -> dict[str, NDArray]:
        n = self._validate_n(n)
        base_seed = self.config.seed + 20_000 if seed is None else int(seed)
        z = self._sample_target_z(n, base_seed)
        treatment = make_rng(base_seed + 431).integers(
            0,
            2,
            size=n,
            dtype=np.int64,
        )
        return {
            "z": z,
            "x": z.astype(np.float32),
            "t": treatment,
            "w": self.encode_w(z, treatment),
            "state_id": self.joint_state_index(z, treatment),
            "q": self.target_state_probability(z, treatment),
        }

    def sample_target_w(
        self,
        n: int,
        seed: Optional[int] = None,
    ) -> NDArray[np.float32]:
        return self.sample_target(n, seed=seed)["w"]

    def state_support(
        self,
        z: NDArray,
        treatment: NDArray | int,
    ) -> NDArray[np.float64]:
        """Return the exact support endpoints for each requested state."""

        z_array = self._validate_z(z)
        treatment_array = self._validate_treatment(treatment, z_array.shape[0])
        mean, theta, _ = self._state_terms(z_array, treatment_array)
        radius = 0.45 * (
            np.abs(np.cos(theta)) * np.sqrt(2.0)
            + np.abs(np.sin(theta)) * 4.0 / np.sqrt(5.0)
        )
        return np.column_stack([mean - radius, mean + radius])

    def outcome_diagnostics(
        self,
        *,
        w1_draws_per_state: int = 4096,
        seed: int = 91_773,
    ) -> dict[str, object]:
        """Return exact moment/support diagnostics and a fixed-draw W1 audit."""

        draws = self._validate_n(w1_draws_per_state)
        z = self.enumerate_covariate_states()
        q_z = self.target_covariate_probability(z)
        control_mean, _, _ = self._state_terms(z, 0)
        treated_mean, _, _ = self._state_terms(z, 1)
        cate = treated_mean - control_mean
        cate_mean = float(np.sum(q_z * cate))
        cate_variance = float(np.sum(q_z * (cate - cate_mean) ** 2))

        between_arm_w1 = np.empty(z.shape[0], dtype=np.float64)
        for state_index in range(z.shape[0]):
            state_seed = int(seed) + 1009 * state_index
            control = self.sample_potential(
                z[state_index: state_index + 1],
                0,
                n_per_x=draws,
                seed=state_seed,
            )[0, :, 0]
            treated = self.sample_potential(
                z[state_index: state_index + 1],
                1,
                n_per_x=draws,
                seed=state_seed,
            )[0, :, 0]
            between_arm_w1[state_index] = float(
                np.mean(np.abs(np.sort(treated) - np.sort(control)))
            )

        table = self.enumerate_target_states()
        support = self.state_support(table["z"], table["t"])
        return {
            "true_cate_mean": cate_mean,
            "true_cate_sd": float(np.sqrt(cate_variance)),
            "mean_between_arm_w1": float(np.sum(q_z * between_arm_w1)),
            "between_arm_w1_by_covariate_state": between_arm_w1.tolist(),
            "between_arm_w1_draws_per_state": draws,
            "conditional_variance_min": self.noise_variance,
            "conditional_variance_max": self.noise_variance,
            "support_min": float(np.min(support[:, 0])),
            "support_max": float(np.max(support[:, 1])),
            "state_support": support.tolist(),
        }

    def design_diagnostics(
        self,
        observed: Optional[dict[str, NDArray]] = None,
        *,
        partition_bits: Optional[int] = None,
    ) -> dict[str, object]:
        """Return exact design masses and, when supplied, realized cell counts."""

        r = self.partition_bits if partition_bits is None else int(partition_bits)
        if not (0 <= r <= self.feature_dim):
            raise ValueError(
                "partition_bits must lie between zero and conditioning_dim"
            )
        table = self.enumerate_target_states()
        partition_ids = self.partition_state_index(
            table["z"],
            table["t"],
            partition_bits=r,
        )
        num_partition_cells = 2 ** (r + 1)
        target_partition_mass = np.bincount(
            partition_ids,
            weights=table["q"],
            minlength=num_partition_cells,
        )
        observed_partition_mass = np.bincount(
            partition_ids,
            weights=table["pi"],
            minlength=num_partition_cells,
        )
        target_hard_mass = float(
            np.sum(
                self.target_covariate_probability(
                    self.enumerate_covariate_states()
                )
                * self.target_shift_mask(
                    self.enumerate_covariate_states()
                )
            )
        )
        diagnostics: dict[str, object] = {
            "conditioning_dim": self.feature_dim,
            "num_covariate_states": self.num_covariate_states,
            "num_joint_states": self.num_joint_states,
            "partition_bits": r,
            "num_partition_cells": num_partition_cells,
            "treatment_locked_in_partition": True,
            "exact_state_partition": r == self.feature_dim,
            "distinct_exact_state_aliasing_at_full_resolution": False,
            "overlap_kappa": self.config.overlap_kappa,
            "target_shift": self.config.target_shift,
            "target_shift_set": self.config.target_shift_set,
            "heterogeneity": self.config.heterogeneity,
            "exact_overlap_constant": self.exact_overlap_constant(),
            "exact_importance_ess_ratio": self.exact_importance_ess_ratio(),
            "target_shift_set_mass": target_hard_mass,
            "state_id": table["state_id"].tolist(),
            "q_by_state": table["q"].tolist(),
            "pi_by_state": table["pi"].tolist(),
            "q_over_pi_by_state": table["q_over_pi"].tolist(),
            "target_partition_mass": target_partition_mass.tolist(),
            "observed_partition_mass": observed_partition_mass.tolist(),
        }
        if observed is None:
            return diagnostics

        z_observed = self._validate_z(observed["z"])
        treatment_observed = self._validate_treatment(
            observed["t"],
            z_observed.shape[0],
        )
        exact_ids = self.joint_state_index(z_observed, treatment_observed)
        cell_ids = self.partition_state_index(
            z_observed,
            treatment_observed,
            partition_bits=r,
        )
        exact_counts = np.bincount(
            exact_ids,
            minlength=self.num_joint_states,
        ).astype(np.int64)
        cell_counts = np.bincount(
            cell_ids,
            minlength=num_partition_cells,
        ).astype(np.int64)
        treatment_counts = np.bincount(
            treatment_observed,
            minlength=2,
        ).astype(np.int64)
        diagnostics.update(
            {
                "n_train": int(z_observed.shape[0]),
                "treatment_counts": treatment_counts.tolist(),
                "exact_state_counts": exact_counts.tolist(),
                "partition_cell_counts": cell_counts.tolist(),
                "empty_exact_states": int(np.count_nonzero(exact_counts == 0)),
                "minimum_exact_state_count": int(exact_counts.min()),
                "maximum_exact_state_count": int(exact_counts.max()),
                "empty_partition_cells": int(np.count_nonzero(cell_counts == 0)),
                "minimum_partition_cell_count": int(cell_counts.min()),
                "maximum_partition_cell_count": int(cell_counts.max()),
                "target_mass_empty_exact_states": float(
                    np.sum(table["q"][exact_counts == 0])
                ),
                "target_mass_empty_partition_cells": float(
                    np.sum(target_partition_mass[cell_counts == 0])
                ),
            }
        )
        return diagnostics
