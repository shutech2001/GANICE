from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .utils import ensure_2d, ensure_row_matrix, make_rng


@dataclass(slots=True)
class FiniteStateCausalDGP:
    x_support: tuple[int, ...] = (0, 1, 2)
    treatments: tuple[int, ...] = (0, 1)
    latent_dim: int = 2
    outcome_bound: float = 3.0
    x_probs: np.ndarray = field(
        default_factory=lambda: np.array([0.25, 0.45, 0.30], dtype=np.float64)
    )
    treatment_probs: np.ndarray = field(
        default_factory=lambda: np.array([0.30, 0.65, 0.45], dtype=np.float64)
    )
    num_states: int = field(init=False)
    target_q: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.x_probs = np.asarray(self.x_probs, dtype=np.float64)
        self.x_probs = self.x_probs / self.x_probs.sum()
        self.treatment_probs = np.asarray(self.treatment_probs, dtype=np.float64)
        if self.treatment_probs.shape[0] != len(self.x_support):
            raise ValueError("treatment_probs must align with x_support")
        self.num_states = len(self.x_support) * len(self.treatments)
        target = []
        for x_idx in self.x_support:
            for treatment in self.treatments:
                target.append(self.x_probs[x_idx] * 0.5)
        self.target_q = np.asarray(target, dtype=np.float64)
        self.target_q = self.target_q / self.target_q.sum()

    def state_index(self, x: int, treatment: int) -> int:
        return x * len(self.treatments) + treatment

    def inverse_state_index(self, state: int) -> tuple[int, int]:
        return divmod(state, len(self.treatments))

    def observational_pi(self) -> np.ndarray:
        pi = []
        for x_idx in self.x_support:
            p_x = self.x_probs[x_idx]
            p_t1 = self.treatment_probs[x_idx]
            pi.extend([p_x * (1.0 - p_t1), p_x * p_t1])
        return np.asarray(pi, dtype=np.float64)

    def importance_weights(self) -> np.ndarray:
        return self.target_q / self.observational_pi()

    def generator_map(self, x: int, treatment: int, u: np.ndarray) -> np.ndarray:
        u = ensure_2d(u)
        u1 = u[:, 0]
        u2 = u[:, 1] if u.shape[1] > 1 else u[:, 0]
        loc = -0.9 + 0.85 * x + 0.95 * treatment - 0.35 * x * treatment
        scale = 0.25 + 0.08 * x + 0.12 * treatment
        oscillation = 0.40 * np.sin(2.0 * np.pi * u1 + 0.35 * x - 0.2 * treatment)
        oscillation += 0.18 * np.cos(2.0 * np.pi * u2 * (1.0 + treatment))
        trend = scale * (u1 + u2 - 1.0)
        y = loc + trend + oscillation
        return np.clip(y[:, None], -self.outcome_bound, self.outcome_bound).astype(np.float32)

    def sample_conditional(self, x: int, treatment: int, n: int, seed: int | None = None) -> np.ndarray:
        rng = make_rng(seed)
        u = rng.uniform(0.0, 1.0, size=(n, self.latent_dim))
        return self.generator_map(x, treatment, u)

    def sample_state(self, state: int, n: int, seed: int | None = None) -> np.ndarray:
        x, treatment = self.inverse_state_index(state)
        return self.sample_conditional(x, treatment, n, seed=seed)

    def sample_observed(self, n: int, seed: int | None = None) -> dict[str, np.ndarray]:
        rng = make_rng(seed)
        x = rng.choice(np.array(self.x_support), size=n, p=self.x_probs)
        treatment_prob = self.treatment_probs[x]
        t = rng.binomial(1, treatment_prob, size=n).astype(np.int64)
        u = rng.uniform(0.0, 1.0, size=(n, self.latent_dim))
        y = np.zeros((n, 1), dtype=np.float32)
        for x_idx in self.x_support:
            for treatment in self.treatments:
                mask = (x == x_idx) & (t == treatment)
                if mask.any():
                    y[mask] = self.generator_map(x_idx, treatment, u[mask])
        state = np.array([self.state_index(int(xi), int(ti)) for xi, ti in zip(x, t, strict=True)])
        return {"x": x.astype(np.int64), "t": t, "state": state, "y": y}

    def true_state_mean(self, state: int, n_mc: int = 4096) -> float:
        return float(self.sample_state(state, n_mc, seed=10_000 + state).mean())


@dataclass(slots=True)
class ContinuousCausalDGP:
    latent_dim: int = 2
    outcome_bound: float = 3.0

    @property
    def d_w(self) -> int:
        return 2

    def sample_target_w(self, n: int, seed: int | None = None) -> np.ndarray:
        rng = make_rng(seed)
        return rng.uniform(0.0, 1.0, size=(n, self.d_w)).astype(np.float32)

    def sample_x_observed(self, n: int, rng: np.random.Generator) -> np.ndarray:
        u = rng.uniform(0.0, 1.0, size=n)
        return ((-0.6 + np.sqrt(0.36 + 1.6 * u)) / 0.8).astype(np.float32)

    def sample_t_observed(self, n: int, rng: np.random.Generator) -> np.ndarray:
        u = rng.uniform(0.0, 1.0, size=n)
        return ((1.3 - np.sqrt(1.69 - 1.2 * u)) / 0.6).astype(np.float32)

    def q_obs_density(self, w: np.ndarray) -> np.ndarray:
        w = ensure_row_matrix(w)
        x = w[:, 0]
        t = w[:, 1]
        return (0.6 + 0.8 * x) * (1.3 - 0.6 * t)

    def q_target_density(self, w: np.ndarray) -> np.ndarray:
        w = ensure_row_matrix(w)
        return np.ones(w.shape[0], dtype=np.float32)

    @property
    def kappa(self) -> float:
        return 0.42

    def importance_weights(self, w: np.ndarray) -> np.ndarray:
        return self.q_target_density(w) / self.q_obs_density(w)

    def generator_map(self, w: np.ndarray, u: np.ndarray) -> np.ndarray:
        w = ensure_row_matrix(w)
        u = ensure_2d(u)
        x = w[:, 0]
        t = w[:, 1]
        u1 = u[:, 0]
        u2 = u[:, 1] if u.shape[1] > 1 else u[:, 0]
        loc = 1.1 * np.sin(np.pi * x) + 0.9 * (t - 0.5) + 0.4 * x * t
        scale = 0.18 + 0.15 * x + 0.10 * t
        oscillation = 0.35 * np.sin(2.0 * np.pi * (u1 + 0.25 * x - 0.15 * t))
        oscillation += 0.16 * np.cos(2.0 * np.pi * u2 + np.pi * x * t)
        trend = scale * (u1 + u2 - 1.0)
        y = loc + trend + oscillation
        return np.clip(y[:, None], -self.outcome_bound, self.outcome_bound).astype(np.float32)

    def sample_conditional(self, w: np.ndarray, n: int, seed: int | None = None) -> np.ndarray:
        rng = make_rng(seed)
        w = np.repeat(ensure_row_matrix(w), n, axis=0)
        u = rng.uniform(0.0, 1.0, size=(n, self.latent_dim))
        return self.generator_map(w, u)

    def sample_observed(self, n: int, seed: int | None = None) -> dict[str, np.ndarray]:
        rng = make_rng(seed)
        x = self.sample_x_observed(n, rng)
        t = self.sample_t_observed(n, rng)
        w = np.column_stack([x, t]).astype(np.float32)
        u = rng.uniform(0.0, 1.0, size=(n, self.latent_dim))
        y = self.generator_map(w, u)
        return {"w": w, "y": y}
