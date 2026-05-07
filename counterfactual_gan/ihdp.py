from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from io import StringIO
from pathlib import Path
import csv
import subprocess

import numpy as np

from .utils import ensure_row_matrix, make_rng


IHDP_CONTINUOUS_COLUMNS = ("bw", "b.head", "preterm", "birth.o", "nnhealth", "momage")
IHDP_RACE_COLUMNS = ("momwhite", "momblack", "momhisp")
IHDP_FEATURE_COLUMNS = (
    "bw",
    "b.head",
    "preterm",
    "birth.o",
    "nnhealth",
    "momage",
    "sex",
    "twin",
    "b.marr",
    "mom.lths",
    "mom.hs",
    "mom.scoll",
    "cig",
    "first",
    "booze",
    "drugs",
    "work.dur",
    "prenatal",
    "ark",
    "ein",
    "har",
    "mia",
    "pen",
    "tex",
    "was",
)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


@lru_cache(maxsize=4)
def _load_r_dataframe_csv(data_path: str, object_name: str) -> dict[str, np.ndarray]:
    """Load a small R data.frame through Rscript without adding a Python RData dependency."""

    path = Path(data_path).expanduser().resolve()
    script = (
        f"load({str(path)!r}); "
        f"write.csv({object_name}, row.names=FALSE, na='')"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    reader = csv.DictReader(StringIO(completed.stdout))
    columns: dict[str, list[float]] = {}
    for row in reader:
        for key, value in row.items():
            columns.setdefault(key, []).append(float("nan") if value == "" else float(value))
    if not columns:
        raise ValueError(f"no columns were loaded from {path}")
    return {key: np.asarray(value, dtype=np.float32) for key, value in columns.items()}


def load_ihdp_hill_sim_data(data_path: str | Path = "Data/IHDP/sim.data") -> dict[str, np.ndarray]:
    """Load and preprocess the Hill/Shalit IHDP covariates.

    The raw Hill file has 985 randomized IHDP units. Following Hill (2011) and
    Shalit et al. (2017), the observational benchmark removes the non-white
    treated children, leaving 747 units: 139 treated and 608 controls. Race
    indicators are then excluded, producing the standard 25 covariates.
    """

    raw = _load_r_dataframe_csv(str(data_path), "imp1")
    missing = [name for name in ("treat", *IHDP_FEATURE_COLUMNS, *IHDP_RACE_COLUMNS) if name not in raw]
    if missing:
        raise ValueError(f"missing IHDP columns: {missing}")
    keep = (raw["treat"] == 0.0) | ((raw["treat"] == 1.0) & (raw["momwhite"] == 1.0))
    x_raw = np.column_stack([raw[name][keep] for name in IHDP_FEATURE_COLUMNS]).astype(np.float32)
    treatment = raw["treat"][keep].astype(np.int64)
    if x_raw.shape != (747, 25):
        raise ValueError(f"unexpected preprocessed IHDP shape {x_raw.shape}; expected (747, 25)")
    treated = int(treatment.sum())
    if treated != 139:
        raise ValueError(f"unexpected treated count {treated}; expected 139")
    return {"x_raw": x_raw, "treatment": treatment}


def _stratified_split(
    treatment: np.ndarray,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, np.ndarray]:
    rng = make_rng(seed)
    indices_by_arm = [np.flatnonzero(treatment == arm) for arm in (0, 1)]
    splits = {"train": [], "validation": [], "test": []}
    for arm_indices in indices_by_arm:
        permuted = rng.permutation(arm_indices)
        n = permuted.size
        n_train = int(round(train_fraction * n))
        n_validation = int(round(validation_fraction * n))
        n_train = min(n_train, n)
        n_validation = min(n_validation, n - n_train)
        splits["train"].append(permuted[:n_train])
        splits["validation"].append(permuted[n_train : n_train + n_validation])
        splits["test"].append(permuted[n_train + n_validation :])
    output = {}
    for name, parts in splits.items():
        joined = np.concatenate(parts).astype(np.int64)
        output[name] = rng.permutation(joined)
    return output


@dataclass(slots=True)
class _IHDPDistParams:
    b0: np.ndarray
    btau: np.ndarray
    b_pi: np.ndarray
    b_delta: np.ndarray
    b_sigma: np.ndarray
    b_sigma_alt: np.ndarray


def _draw_params(rng: np.random.Generator, x_dim: int) -> _IHDPDistParams:
    mean_scale = 1.0 / np.sqrt(float(x_dim))
    distribution_scale = 2.5 / np.sqrt(float(x_dim))
    return _IHDPDistParams(
        b0=rng.normal(0.0, mean_scale, size=x_dim).astype(np.float32),
        btau=rng.normal(0.0, mean_scale, size=x_dim).astype(np.float32),
        b_pi=rng.normal(0.0, distribution_scale, size=(2, x_dim)).astype(np.float32),
        b_delta=rng.normal(0.0, distribution_scale, size=(2, x_dim)).astype(np.float32),
        b_sigma=rng.normal(0.0, distribution_scale, size=(2, x_dim)).astype(np.float32),
        b_sigma_alt=rng.normal(0.0, distribution_scale, size=(2, x_dim)).astype(np.float32),
    )


@dataclass(slots=True)
class IHDPDistDGP:
    """IHDP-Dist semi-synthetic binary-treatment benchmark.

    The preprocessing follows the Hill/Shalit IHDP benchmark. The stochastic
    outcome law follows the IHDP-Dist construction in the GANICE draft: the
    conditional mean is an IHDP-style nonlinear response, while the full law is
    a heteroskedastic normal/Student-t mixture.
    """

    data_path: Path = Path("Data/IHDP/sim.data")
    seed: int = 7
    split_seed: int = 11
    train_fraction: float = 0.63
    validation_fraction: float = 0.27
    student_df: float = 5.0
    raw: dict[str, np.ndarray] = field(init=False)
    split_indices: dict[str, np.ndarray] = field(init=False)
    continuous_indices: np.ndarray = field(init=False)
    x_mean: np.ndarray = field(init=False)
    x_std: np.ndarray = field(init=False)
    x: np.ndarray = field(init=False)
    treatment: np.ndarray = field(init=False)
    params: _IHDPDistParams = field(init=False)

    def __post_init__(self) -> None:
        self.raw = load_ihdp_hill_sim_data(self.data_path)
        self.treatment = self.raw["treatment"]
        self.split_indices = _stratified_split(
            self.treatment,
            seed=self.split_seed,
            train_fraction=self.train_fraction,
            validation_fraction=self.validation_fraction,
        )
        self.continuous_indices = np.asarray(
            [IHDP_FEATURE_COLUMNS.index(name) for name in IHDP_CONTINUOUS_COLUMNS],
            dtype=np.int64,
        )
        x_raw = self.raw["x_raw"].astype(np.float32)
        train_x = x_raw[self.split_indices["train"]]
        self.x_mean = np.zeros(x_raw.shape[1], dtype=np.float32)
        self.x_std = np.ones(x_raw.shape[1], dtype=np.float32)
        self.x_mean[self.continuous_indices] = train_x[:, self.continuous_indices].mean(axis=0)
        std = train_x[:, self.continuous_indices].std(axis=0)
        self.x_std[self.continuous_indices] = np.where(std < 1e-6, 1.0, std)
        self.x = self.transform_x(x_raw)
        self.params = _draw_params(make_rng(self.seed), self.feature_dim)

    @property
    def feature_dim(self) -> int:
        return len(IHDP_FEATURE_COLUMNS)

    @property
    def d_w(self) -> int:
        return self.feature_dim + 1

    @staticmethod
    def treatment_embedding(treatment: np.ndarray | int) -> np.ndarray:
        treatment_array = np.asarray(treatment, dtype=np.float32)
        return 0.25 + 0.5 * treatment_array

    def transform_x(self, x_raw: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_raw, dtype=np.float32)
        transformed = x_arr.copy()
        transformed[:, self.continuous_indices] = (
            transformed[:, self.continuous_indices] - self.x_mean[self.continuous_indices]
        ) / self.x_std[self.continuous_indices]
        return transformed.astype(np.float32)

    def split(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        if name not in self.split_indices:
            raise ValueError(f"unknown IHDP split {name!r}")
        idx = self.split_indices[name]
        return self.x[idx].astype(np.float32), self.treatment[idx].astype(np.int64)

    def encode_w(self, x: np.ndarray, treatment: np.ndarray | int) -> np.ndarray:
        x_arr = ensure_row_matrix(x)
        treatment_embed = self.treatment_embedding(treatment).reshape(-1, 1)
        if treatment_embed.shape[0] == 1 and x_arr.shape[0] > 1:
            treatment_embed = np.repeat(treatment_embed, x_arr.shape[0], axis=0)
        return np.column_stack([x_arr, treatment_embed[:, 0]]).astype(np.float32)

    def decode_w(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w_arr = ensure_row_matrix(w)
        return w_arr[:, : self.feature_dim].astype(np.float32), (w_arr[:, -1] > 0.5).astype(np.int64)

    def sample_target_w(self, n: int, seed: int | None = None) -> np.ndarray:
        rng = make_rng(seed)
        x_test, _ = self.split("test")
        x = x_test[rng.integers(0, x_test.shape[0], size=n)]
        treatment = rng.binomial(1, 0.5, size=n).astype(np.int64)
        return self.encode_w(x, treatment)

    def potential_means(self, x: np.ndarray) -> np.ndarray:
        x_arr = ensure_row_matrix(x).astype(np.float64)
        p = self.params
        m0 = np.exp(0.2 * (x_arr @ p.b0.astype(np.float64))) - 1.0
        tau = 1.0 + 0.5 * np.tanh(x_arr @ p.btau.astype(np.float64))
        return np.column_stack([m0, m0 + tau]).astype(np.float32)

    def _mixture_terms(self, x: np.ndarray, treatment: np.ndarray) -> tuple[np.ndarray, ...]:
        x_arr = ensure_row_matrix(x).astype(np.float64)
        treatment_arr = np.asarray(treatment, dtype=np.int64).reshape(-1)
        means = self.potential_means(x_arr).astype(np.float64)
        selected_mean = means[np.arange(x_arr.shape[0]), treatment_arr]
        p = self.params
        pi = np.zeros(x_arr.shape[0], dtype=np.float64)
        delta = np.zeros_like(pi)
        sigma_1 = np.zeros_like(pi)
        sigma_2 = np.zeros_like(pi)
        for arm in (0, 1):
            mask = treatment_arr == arm
            if not np.any(mask):
                continue
            x_arm = x_arr[mask]
            pi[mask] = _sigmoid(x_arm @ p.b_pi[arm].astype(np.float64))
            delta[mask] = 0.5 + 0.5 * np.tanh(x_arm @ p.b_delta[arm].astype(np.float64))
            sigma_1[mask] = 0.2 + 0.3 * _sigmoid(x_arm @ p.b_sigma[arm].astype(np.float64))
            sigma_2[mask] = 0.3 + 0.5 * _sigmoid(x_arm @ p.b_sigma_alt[arm].astype(np.float64))
        loc_1 = selected_mean + (1.0 - pi) * delta
        loc_2 = selected_mean - pi * delta
        return pi, loc_1, sigma_1, loc_2, sigma_2

    def sample_potential(
        self,
        x: np.ndarray,
        treatment: np.ndarray | int,
        n_per_x: int = 1,
        seed: int | None = None,
    ) -> np.ndarray:
        x_arr = ensure_row_matrix(x)
        n = x_arr.shape[0]
        treatment_arr = np.asarray(treatment, dtype=np.int64).reshape(-1)
        if treatment_arr.size == 1:
            treatment_arr = np.full(n, int(treatment_arr.item()), dtype=np.int64)
        if treatment_arr.shape[0] != n:
            raise ValueError("treatment must be scalar or align with x")
        repeated_x = np.repeat(x_arr, n_per_x, axis=0)
        repeated_t = np.repeat(treatment_arr, n_per_x)
        rng = make_rng(seed)
        pi, loc_1, sigma_1, loc_2, sigma_2 = self._mixture_terms(repeated_x, repeated_t)
        use_normal = rng.uniform(0.0, 1.0, size=repeated_x.shape[0]) < pi
        values = np.empty(repeated_x.shape[0], dtype=np.float64)
        values[use_normal] = rng.normal(loc_1[use_normal], sigma_1[use_normal])
        values[~use_normal] = loc_2[~use_normal] + sigma_2[~use_normal] * rng.standard_t(
            self.student_df,
            size=int((~use_normal).sum()),
        )
        return values.reshape(n, n_per_x, 1).astype(np.float32)

    def sample_conditional(self, w: np.ndarray, n: int, seed: int | None = None) -> np.ndarray:
        x, treatment = self.decode_w(w)
        return self.sample_potential(x[:1], int(treatment[0]), n_per_x=n, seed=seed).reshape(n, 1)

    def observed_split(self, name: str, seed: int | None = None) -> dict[str, np.ndarray]:
        x, treatment = self.split(name)
        y = self.sample_potential(x, treatment, n_per_x=1, seed=seed).reshape(-1, 1)
        return {
            "x": x,
            "t": treatment,
            "w": self.encode_w(x, treatment),
            "y": y.astype(np.float32),
            "mu": self.potential_means(x),
        }

    def outcome_bounds(self, observed_y: np.ndarray, margin: float = 0.75) -> tuple[float, float]:
        y = np.asarray(observed_y, dtype=np.float64).reshape(-1)
        lower = float(np.quantile(y, 0.01) - margin)
        upper = float(np.quantile(y, 0.99) + margin)
        if upper - lower < 1.0:
            midpoint = 0.5 * (lower + upper)
            lower = midpoint - 0.5
            upper = midpoint + 0.5
        return lower, upper
