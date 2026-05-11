from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .utils import ensure_row_matrix, make_rng


JOBS_FEATURE_COLUMNS = (
    "age",
    "education",
    "black",
    "hispanic",
    "married",
    "nodegree",
    "re75",
)
JOBS_CONTINUOUS_INDICES = np.array([0, 1, 6], dtype=np.int64)


def earnings_to_model_scale(earnings: NDArray) -> NDArray:
    return np.arcsinh(np.asarray(earnings, dtype=np.float64) / 1000.0).astype(np.float32)


def model_scale_to_earnings(values: NDArray) -> NDArray:
    return (1000.0 * np.sinh(np.asarray(values, dtype=np.float64))).astype(np.float64)


def _split_indices(
    n: int, rng: np.random.Generator, train_fraction: float, validation_fraction: float
) -> Dict[str, NDArray]:
    permuted = rng.permutation(np.arange(n, dtype=np.int64))
    n_train = int(round(train_fraction * n))
    n_validation = int(round(validation_fraction * n))
    n_train = min(n_train, n)
    n_validation = min(n_validation, n - n_train)
    return {
        "train": permuted[:n_train],
        "validation": permuted[n_train: n_train + n_validation],
        "test": permuted[n_train + n_validation:],
    }


def _read_jobs_file(path: Path, expected_columns: int) -> NDArray:
    array = np.loadtxt(path, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != expected_columns:
        raise ValueError(f"{path} has shape {array.shape}; expected (*, {expected_columns})")
    return array


@dataclass
class JobsLaLondeData:
    """NSW/PSID Jobs-LaLonde loader with Shalit/GANITE-style splitting.

    The randomized NSW sample is split within treatment arm into
    train/validation/test subsets. PSID controls are split separately. The
    observational training set uses NSW treated units and PSID controls by
    default; setting ``include_nsw_control_in_observed`` also includes the NSW
    control split, matching the full 3212-unit Shalit/GANITE split while still
    keeping held-out RCT units out of training.
    """

    data_dir: str | Path = "Data/Jobs"
    split_seed: int = 7
    train_fraction: float = 0.56
    validation_fraction: float = 0.24
    include_nsw_control_in_observed: bool = True

    def __post_init__(self) -> None:
        data_dir = Path(self.data_dir)
        nsw_treated = _read_jobs_file(data_dir / "nsw_treated.txt", expected_columns=9)
        nsw_control = _read_jobs_file(data_dir / "nsw_control.txt", expected_columns=9)
        psid_control = _read_jobs_file(data_dir / "psid_controls.txt", expected_columns=10)

        if nsw_treated.shape[0] != 297 or nsw_control.shape[0] != 425 or psid_control.shape[0] != 2490:
            raise ValueError(
                "unexpected Jobs row counts; expected NSW treated 297, NSW control 425, PSID controls 2490"
            )
        if not np.all(nsw_treated[:, 0] == 1.0):
            raise ValueError("nsw_treated.txt must contain only treated rows")
        if not np.all(nsw_control[:, 0] == 0.0) or not np.all(psid_control[:, 0] == 0.0):
            raise ValueError("control files must contain only control rows")

        self._nsw_treated_x_raw = nsw_treated[:, [1, 2, 3, 4, 5, 6, 7]].astype(np.float32)
        self._nsw_treated_y_earn = nsw_treated[:, 8].astype(np.float64)
        self._nsw_control_x_raw = nsw_control[:, [1, 2, 3, 4, 5, 6, 7]].astype(np.float32)
        self._nsw_control_y_earn = nsw_control[:, 8].astype(np.float64)
        # PSID has an additional RE74 column. The common feature set drops RE74
        # and keeps RE75 in the final feature position.
        self._psid_control_x_raw = psid_control[:, [1, 2, 3, 4, 5, 6, 8]].astype(np.float32)
        self._psid_control_y_earn = psid_control[:, 9].astype(np.float64)

        rng = make_rng(self.split_seed)
        self._splits = {
            "nsw_treated": _split_indices(
                self._nsw_treated_x_raw.shape[0],
                rng,
                self.train_fraction,
                self.validation_fraction,
            ),
            "nsw_control": _split_indices(
                self._nsw_control_x_raw.shape[0],
                rng,
                self.train_fraction,
                self.validation_fraction,
            ),
            "psid_control": _split_indices(
                self._psid_control_x_raw.shape[0],
                rng,
                self.train_fraction,
                self.validation_fraction,
            ),
        }

        train_x_raw = self._observed_x_raw("train")
        self.x_mean = np.zeros(self.feature_dim, dtype=np.float32)
        self.x_std = np.ones(self.feature_dim, dtype=np.float32)
        self.x_mean[JOBS_CONTINUOUS_INDICES] = train_x_raw[:, JOBS_CONTINUOUS_INDICES].mean(axis=0)
        train_std = train_x_raw[:, JOBS_CONTINUOUS_INDICES].std(axis=0)
        self.x_std[JOBS_CONTINUOUS_INDICES] = np.where(train_std < 1e-6, 1.0, train_std)

    @property
    def feature_dim(self) -> int:
        return len(JOBS_FEATURE_COLUMNS)

    @property
    def d_w(self) -> int:
        return self.feature_dim + 1

    @staticmethod
    def treatment_embedding(treatment: NDArray | int) -> NDArray:
        treatment_array = np.asarray(treatment, dtype=np.float32)
        return 0.25 + 0.5 * treatment_array

    def transform_x(self, x_raw: NDArray) -> NDArray:
        x_arr = np.asarray(x_raw, dtype=np.float32).copy()
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
        x_arr[:, JOBS_CONTINUOUS_INDICES] = (
            x_arr[:, JOBS_CONTINUOUS_INDICES] - self.x_mean[JOBS_CONTINUOUS_INDICES]
        ) / self.x_std[JOBS_CONTINUOUS_INDICES]
        return x_arr.astype(np.float32)

    def encode_w(self, x: NDArray, treatment: NDArray | int) -> NDArray:
        x_arr = ensure_row_matrix(x)
        treatment_embed = self.treatment_embedding(treatment).reshape(-1, 1)
        if treatment_embed.shape[0] == 1 and x_arr.shape[0] > 1:
            treatment_embed = np.repeat(treatment_embed, x_arr.shape[0], axis=0)
        return np.column_stack([x_arr, treatment_embed[:, 0]]).astype(np.float32)

    def decode_w(self, w: NDArray) -> Tuple[NDArray, NDArray]:
        w_arr = ensure_row_matrix(w)
        return w_arr[:, : self.feature_dim].astype(np.float32), (w_arr[:, -1] > 0.5).astype(np.int64)

    def _source_split(self, source: str, split: str) -> NDArray:
        if split not in ("train", "validation", "test"):
            raise ValueError(f"unknown Jobs split {split!r}")
        return self._splits[source][split]

    def _observed_x_raw(self, split: str) -> NDArray:
        parts = [self._nsw_treated_x_raw[self._source_split("nsw_treated", split)]]
        if self.include_nsw_control_in_observed:
            parts.append(self._nsw_control_x_raw[self._source_split("nsw_control", split)])
        parts.append(self._psid_control_x_raw[self._source_split("psid_control", split)])
        return np.concatenate(parts, axis=0).astype(np.float32)

    def observed_split(self, split: str) -> Dict[str, NDArray]:
        treated_idx = self._source_split("nsw_treated", split)
        psid_idx = self._source_split("psid_control", split)
        x_parts = [self._nsw_treated_x_raw[treated_idx]]
        t_parts = [np.ones(treated_idx.shape[0], dtype=np.int64)]
        y_parts = [self._nsw_treated_y_earn[treated_idx]]
        source_parts = [np.full(treated_idx.shape[0], "nsw_treated", dtype=object)]

        if self.include_nsw_control_in_observed:
            control_idx = self._source_split("nsw_control", split)
            x_parts.append(self._nsw_control_x_raw[control_idx])
            t_parts.append(np.zeros(control_idx.shape[0], dtype=np.int64))
            y_parts.append(self._nsw_control_y_earn[control_idx])
            source_parts.append(np.full(control_idx.shape[0], "nsw_control", dtype=object))

        x_parts.append(self._psid_control_x_raw[psid_idx])
        t_parts.append(np.zeros(psid_idx.shape[0], dtype=np.int64))
        y_parts.append(self._psid_control_y_earn[psid_idx])
        source_parts.append(np.full(psid_idx.shape[0], "psid_control", dtype=object))

        x_raw = np.concatenate(x_parts, axis=0).astype(np.float32)
        treatment = np.concatenate(t_parts, axis=0).astype(np.int64)
        y_earn = np.concatenate(y_parts, axis=0).astype(np.float64)
        source = np.concatenate(source_parts, axis=0)
        x = self.transform_x(x_raw)
        y = earnings_to_model_scale(y_earn).reshape(-1, 1)
        return {
            "x": x,
            "x_raw": x_raw,
            "t": treatment,
            "w": self.encode_w(x, treatment),
            "y": y.astype(np.float32),
            "y_earnings": y_earn.astype(np.float64),
            "source": source,
        }

    def rct_split(self, split: str) -> Dict[str, NDArray]:
        treated_idx = self._source_split("nsw_treated", split)
        control_idx = self._source_split("nsw_control", split)
        x_raw = np.concatenate(
            [self._nsw_treated_x_raw[treated_idx], self._nsw_control_x_raw[control_idx]],
            axis=0,
        ).astype(np.float32)
        treatment = np.concatenate(
            [
                np.ones(treated_idx.shape[0], dtype=np.int64),
                np.zeros(control_idx.shape[0], dtype=np.int64),
            ],
            axis=0,
        )
        y_earn = np.concatenate(
            [self._nsw_treated_y_earn[treated_idx], self._nsw_control_y_earn[control_idx]],
            axis=0,
        ).astype(np.float64)
        x = self.transform_x(x_raw)
        return {
            "x": x,
            "x_raw": x_raw,
            "t": treatment,
            "w": self.encode_w(x, treatment),
            "y": earnings_to_model_scale(y_earn).reshape(-1, 1).astype(np.float32),
            "y_earnings": y_earn,
        }

    def sample_target_w(self, n: int, seed: Optional[int] = None, split: str = "test") -> NDArray:
        rng = make_rng(seed)
        rct = self.rct_split(split)
        x = rct["x"][rng.integers(0, rct["x"].shape[0], size=n)]
        treatment = rng.binomial(1, 0.5, size=n).astype(np.int64)
        return self.encode_w(x, treatment)

    def outcome_bounds(self, observed_y: NDArray, margin: float = 0.35) -> Tuple[float, float]:
        y = np.asarray(observed_y, dtype=np.float64).reshape(-1)
        lower = 0.0
        upper = float(np.quantile(y, 0.995) + margin)
        if upper <= lower + 1.0:
            upper = lower + 1.0
        return lower, upper

    def rct_att_earnings(self, split: str = "test") -> float:
        rct = self.rct_split(split)
        treated = rct["y_earnings"][rct["t"] == 1]
        control = rct["y_earnings"][rct["t"] == 0]
        return float(treated.mean() - control.mean())

    def source_counts(self, split: str = "train") -> Dict[str, int]:
        observed = self.observed_split(split)
        return {source: int(np.sum(observed["source"] == source)) for source in np.unique(observed["source"])}
