from __future__ import annotations

from typing import Callable

import numpy as np


def sample_wasserstein_1_1d(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    a = np.sort(np.asarray(sample_a, dtype=np.float64).reshape(-1))
    b = np.sort(np.asarray(sample_b, dtype=np.float64).reshape(-1))
    if a.size == 0 or b.size == 0:
        raise ValueError("sample_wasserstein_1_1d requires non-empty samples")
    grid_size = max(a.size, b.size)
    probs = (np.arange(grid_size, dtype=np.float64) + 0.5) / grid_size
    qa = np.quantile(a, probs, method="linear")
    qb = np.quantile(b, probs, method="linear")
    return float(np.mean(np.abs(qa - qb)))


def finite_state_conditional_w1(
    state_weights: np.ndarray,
    true_sampler: Callable[[int, int], np.ndarray],
    learned_sampler: Callable[[int, int], np.ndarray],
    n_per_state: int,
) -> float:
    state_weights = np.asarray(state_weights, dtype=np.float64)
    total = 0.0
    for state in range(state_weights.size):
        y_true = true_sampler(state, n_per_state)
        y_learned = learned_sampler(state, n_per_state)
        total += state_weights[state] * sample_wasserstein_1_1d(y_true, y_learned)
    return float(total)


def continuous_conditional_w1_grid(
    w_grid: np.ndarray,
    true_sampler: Callable[[np.ndarray, int], np.ndarray],
    learned_sampler: Callable[[np.ndarray, int], np.ndarray],
    n_per_w: int,
) -> float:
    total = 0.0
    for w in np.asarray(w_grid, dtype=np.float32):
        y_true = true_sampler(w, n_per_w)
        y_learned = learned_sampler(w, n_per_w)
        total += sample_wasserstein_1_1d(y_true, y_learned)
    return float(total / len(w_grid))
