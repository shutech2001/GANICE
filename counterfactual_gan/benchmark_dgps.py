from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .utils import ensure_row_matrix, make_rng


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(slots=True)
class GANITEBenchmarkDGP:
    outcome_noise: float = 0.05
    outcome_bound: float = 1.0

    @property
    def feature_dim(self) -> int:
        return 1

    @property
    def d_w(self) -> int:
        return 2

    @staticmethod
    def treatment_embedding(t: np.ndarray | int) -> np.ndarray:
        t_array = np.asarray(t, dtype=np.float32)
        return 0.25 + 0.5 * t_array

    def treatment_propensity(self, x: np.ndarray) -> np.ndarray:
        x = ensure_row_matrix(x)
        raw = _sigmoid(2.4 * (x[:, 0] - 0.45) + 0.35 * np.sin(2.0 * np.pi * x[:, 0]))
        return np.clip(0.30 + 0.40 * raw, 0.30, 0.70).astype(np.float32)

    def potential_means(self, x: np.ndarray) -> np.ndarray:
        x = ensure_row_matrix(x)
        mu0 = _sigmoid(-0.6 + 2.1 * x[:, 0] - 1.4 * (x[:, 0] - 0.65) ** 2)
        tau = 0.18 + 0.28 * np.sin(np.pi * x[:, 0]) - 0.10 * x[:, 0]
        mu1 = np.clip(mu0 + tau, 0.0, 1.0)
        return np.stack([mu0, mu1], axis=1).astype(np.float32)

    def sample_x(self, n: int, seed: int | None = None) -> np.ndarray:
        rng = make_rng(seed)
        return rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)

    def sample_target_w(self, n: int, seed: int | None = None) -> np.ndarray:
        rng = make_rng(seed)
        x = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
        t = rng.binomial(1, 0.5, size=n).astype(np.int64)
        return self.encode_w(x, t)

    def encode_w(self, x: np.ndarray, t: np.ndarray | int) -> np.ndarray:
        x = ensure_row_matrix(x)
        t_embed = self.treatment_embedding(t).reshape(-1, 1)
        return np.column_stack([x[:, 0], t_embed[:, 0]]).astype(np.float32)

    def decode_w(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = ensure_row_matrix(w)
        x = w[:, :1].astype(np.float32)
        t = (w[:, 1] > 0.5).astype(np.int64)
        return x, t

    def q_obs_density(self, w: np.ndarray) -> np.ndarray:
        x, t = self.decode_w(w)
        p1 = self.treatment_propensity(x)
        return np.where(t == 1, p1, 1.0 - p1).astype(np.float32)

    def q_target_density(self, w: np.ndarray) -> np.ndarray:
        w = ensure_row_matrix(w)
        return np.full(w.shape[0], 0.5, dtype=np.float32)

    @property
    def kappa(self) -> float:
        return 0.55

    def sample_observed(self, n: int, seed: int | None = None) -> dict[str, np.ndarray]:
        rng = make_rng(seed)
        x = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
        means = self.potential_means(x)
        p1 = self.treatment_propensity(x)
        t = rng.binomial(1, p1).astype(np.int64)
        y = means[np.arange(n), t][:, None] + rng.normal(0.0, self.outcome_noise, size=(n, 1))
        y = np.clip(y, 0.0, self.outcome_bound).astype(np.float32)
        w = self.encode_w(x, t)
        return {"x": x, "t": t, "w": w, "y": y, "mu": means}

    def sample_conditional(self, w: np.ndarray, n: int, seed: int | None = None) -> np.ndarray:
        rng = make_rng(seed)
        x, t = self.decode_w(w)
        means = self.potential_means(x)
        y = means[0, int(t[0])] + rng.normal(0.0, self.outcome_noise, size=(n, 1))
        return np.clip(y, 0.0, self.outcome_bound).astype(np.float32)


@dataclass(slots=True)
class SCIGANBenchmarkDGP:
    num_treatments: int = 2
    outcome_noise: float = 0.1
    curve_scale: float = 2.0
    seed: int = 17
    params: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = make_rng(self.seed)
        self.params = rng.uniform(0.5, 1.4, size=(self.num_treatments, 3)).astype(np.float32)

    @property
    def feature_dim(self) -> int:
        return 1

    @property
    def d_w(self) -> int:
        return 3

    @staticmethod
    def treatment_embedding(t: np.ndarray | int) -> np.ndarray:
        t_array = np.asarray(t, dtype=np.float32)
        return 0.25 + 0.5 * t_array

    def _treatment_features(self, x: np.ndarray, treatment: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = ensure_row_matrix(x)
        p = self.params[treatment]
        a = 0.35 + p[0] * x[:, 0]
        b = 0.65 + p[1] * x[:, 0]
        c = 0.75 + p[2] * x[:, 0]
        return a, b, c

    def optimal_dosage(self, x: np.ndarray, treatment: int) -> np.ndarray:
        a, b, c = self._treatment_features(x, treatment)
        if treatment == 0:
            opt = b / (2.0 * c)
        else:
            opt = c / (2.0 * b)
        return np.clip(opt, 0.0, 1.0).astype(np.float32)

    def response_mean(self, x: np.ndarray, treatment: int, dosage: np.ndarray | float) -> np.ndarray:
        x = ensure_row_matrix(x)
        dosage_array = np.asarray(dosage, dtype=np.float32).reshape(-1)
        if dosage_array.size == 1:
            dosage_array = np.full(x.shape[0], dosage_array.item(), dtype=np.float32)
        a, b, c = self._treatment_features(x, treatment)
        d = dosage_array
        if treatment == 0:
            value = self.curve_scale * (a + 1.2 * b * d - 1.2 * c * d**2)
        else:
            ratio = b / c
            value = self.curve_scale * (0.5 + a + np.sin(np.pi * ratio * d))
        return value.astype(np.float32)

    def potential_means(self, x: np.ndarray, dosage: np.ndarray) -> np.ndarray:
        x = ensure_row_matrix(x)
        dosage = np.asarray(dosage, dtype=np.float32).reshape(-1)
        outputs = [self.response_mean(x, treatment, dosage) for treatment in range(self.num_treatments)]
        return np.stack(outputs, axis=1).astype(np.float32)

    def sample_x(self, n: int, seed: int | None = None) -> np.ndarray:
        rng = make_rng(seed)
        return rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)

    def treatment_propensity(self, x: np.ndarray) -> np.ndarray:
        x = ensure_row_matrix(x)
        best0 = self.response_mean(x, 0, self.optimal_dosage(x, 0))
        best1 = self.response_mean(x, 1, self.optimal_dosage(x, 1))
        raw = _sigmoid(0.45 * (best1 - best0))
        return np.clip(0.30 + 0.40 * raw, 0.30, 0.70).astype(np.float32)

    def encode_w(self, x: np.ndarray, t: np.ndarray | int, dosage: np.ndarray | float) -> np.ndarray:
        x = ensure_row_matrix(x)
        t_embed = self.treatment_embedding(t).reshape(-1, 1)
        dosage = np.asarray(dosage, dtype=np.float32).reshape(-1, 1)
        if dosage.shape[0] == 1 and x.shape[0] > 1:
            dosage = np.repeat(dosage, x.shape[0], axis=0)
        return np.column_stack([x[:, 0], t_embed[:, 0], dosage[:, 0]]).astype(np.float32)

    def decode_w(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w = ensure_row_matrix(w)
        x = w[:, :1].astype(np.float32)
        t = (w[:, 1] > 0.5).astype(np.int64)
        dosage = w[:, 2].astype(np.float32)
        return x, t, dosage

    def sample_target_w(self, n: int, seed: int | None = None) -> np.ndarray:
        rng = make_rng(seed)
        x = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
        t = rng.binomial(1, 0.5, size=n).astype(np.int64)
        d = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
        return self.encode_w(x, t, d)

    def q_obs_density(self, w: np.ndarray) -> np.ndarray:
        x, t, _ = self.decode_w(w)
        p1 = self.treatment_propensity(x)
        return np.where(t == 1, p1, 1.0 - p1).astype(np.float32)

    def q_target_density(self, w: np.ndarray) -> np.ndarray:
        w = ensure_row_matrix(w)
        return np.full(w.shape[0], 0.5, dtype=np.float32)

    @property
    def kappa(self) -> float:
        return 0.55

    def sample_observed(self, n: int, seed: int | None = None) -> dict[str, np.ndarray]:
        rng = make_rng(seed)
        x = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
        p1 = self.treatment_propensity(x)
        t = rng.binomial(1, p1).astype(np.int64)
        dosage = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
        means = self.potential_means(x, dosage)
        y = means[np.arange(n), t][:, None] + rng.normal(0.0, self.outcome_noise, size=(n, 1))
        y = y.astype(np.float32)
        w = self.encode_w(x, t, dosage)
        return {"x": x, "t": t, "d": dosage, "w": w, "y": y, "mu": means}

    def sample_conditional(self, w: np.ndarray, n: int, seed: int | None = None) -> np.ndarray:
        rng = make_rng(seed)
        x, t, dosage = self.decode_w(w)
        mean = self.response_mean(x, int(t[0]), dosage[0])
        y = mean[0] + rng.normal(0.0, self.outcome_noise, size=(n, 1))
        return y.astype(np.float32)
