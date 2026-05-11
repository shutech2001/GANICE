from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
from numpy.typing import NDArray


def _as_nonempty_1d(values: NDArray, name: str) -> NDArray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return array


def sample_wasserstein_1_1d(sample_a: NDArray, sample_b: NDArray) -> float:
    a = np.sort(_as_nonempty_1d(sample_a, "sample_a"))
    b = np.sort(_as_nonempty_1d(sample_b, "sample_b"))
    grid_size = max(a.size, b.size)
    probs = (np.arange(grid_size, dtype=np.float64) + 0.5) / grid_size
    qa = np.quantile(a, probs, method="linear")
    qb = np.quantile(b, probs, method="linear")
    return float(np.mean(np.abs(qa - qb)))


def mean_abs_between_1d(sample_a: NDArray, sample_b: NDArray) -> float:
    a = np.sort(_as_nonempty_1d(sample_a, "sample_a"))
    b = np.sort(_as_nonempty_1d(sample_b, "sample_b"))
    prefix = np.concatenate([[0.0], np.cumsum(a)])
    ranks = np.searchsorted(a, b, side="right")
    left = b * ranks - prefix[ranks]
    right = (prefix[-1] - prefix[ranks]) - b * (a.size - ranks)
    return float(np.mean(left + right) / float(a.size))


def mean_abs_within_1d(sample: NDArray) -> float:
    values = np.sort(_as_nonempty_1d(sample, "sample"))
    n = values.size
    coeff = 2.0 * np.arange(n, dtype=np.float64) - float(n) + 1.0
    return float(2.0 * np.sum(coeff * values) / float(n * n))


def crps_empirical_1d(predicted: NDArray, observed: NDArray) -> float:
    return float(mean_abs_between_1d(predicted, observed) - 0.5 * mean_abs_within_1d(predicted))


def energy_distance_1d(sample_a: NDArray, sample_b: NDArray) -> float:
    value = (
        2.0 * mean_abs_between_1d(sample_a, sample_b)
        - mean_abs_within_1d(sample_a)
        - mean_abs_within_1d(sample_b)
    )
    return float(max(0.0, value))


def empirical_cdf_values_1d(reference: NDArray, points: NDArray) -> NDArray:
    ref = np.sort(_as_nonempty_1d(reference, "reference"))
    query = np.asarray(points, dtype=np.float64).reshape(-1)
    return np.searchsorted(ref, query, side="right").astype(np.float64) / float(ref.size)


def ks_2samp_1d(sample_a: NDArray, sample_b: NDArray) -> float:
    a = _as_nonempty_1d(sample_a, "sample_a")
    b = _as_nonempty_1d(sample_b, "sample_b")
    grid = np.unique(np.concatenate([a, b]))
    diff = np.abs(empirical_cdf_values_1d(a, grid) - empirical_cdf_values_1d(b, grid))
    return float(np.max(diff))


def cvm_2samp_1d(sample_a: NDArray, sample_b: NDArray) -> float:
    a = _as_nonempty_1d(sample_a, "sample_a")
    b = _as_nonempty_1d(sample_b, "sample_b")
    diff_a = empirical_cdf_values_1d(a, a) - empirical_cdf_values_1d(b, a)
    diff_b = empirical_cdf_values_1d(a, b) - empirical_cdf_values_1d(b, b)
    return float(0.5 * (np.mean(diff_a**2) + np.mean(diff_b**2)))


def mmd2_gaussian_median_1d(sample_a: NDArray, sample_b: NDArray) -> float:
    a = _as_nonempty_1d(sample_a, "sample_a")
    b = _as_nonempty_1d(sample_b, "sample_b")
    pooled = np.concatenate([a, b])
    distances = np.abs(pooled[:, None] - pooled[None, :])
    positive = distances[distances > 1e-12]
    bandwidth = float(np.median(positive)) if positive.size else float(np.std(pooled))
    if not np.isfinite(bandwidth) or bandwidth <= 1e-12:
        bandwidth = 1.0
    values = []
    for scale in (0.5, 1.0, 2.0):
        ell = max(scale * bandwidth, 1e-12)
        k_aa = np.exp(-((a[:, None] - a[None, :]) ** 2) / (2.0 * ell**2)).mean()
        k_bb = np.exp(-((b[:, None] - b[None, :]) ** 2) / (2.0 * ell**2)).mean()
        k_ab = np.exp(-((a[:, None] - b[None, :]) ** 2) / (2.0 * ell**2)).mean()
        values.append(k_aa + k_bb - 2.0 * k_ab)
    return float(max(0.0, np.mean(values)))


def quantile_squared_error_sum_1d(
    predicted: NDArray,
    truth: NDArray,
    quantiles: NDArray,
) -> float:
    q = np.asarray(quantiles, dtype=np.float64)
    pred_q = np.quantile(_as_nonempty_1d(predicted, "predicted"), q, method="linear")
    true_q = np.quantile(_as_nonempty_1d(truth, "truth"), q, method="linear")
    return float(np.sum((pred_q - true_q) ** 2))


def tail_mean_error_1d(predicted: NDArray, truth: NDArray, levels: NDArray) -> float:
    pred = _as_nonempty_1d(predicted, "predicted")
    true = _as_nonempty_1d(truth, "truth")
    total = 0.0
    for alpha in np.asarray(levels, dtype=np.float64):
        pred_q = float(np.quantile(pred, alpha, method="linear"))
        true_q = float(np.quantile(true, alpha, method="linear"))
        pred_lower = pred[pred <= pred_q]
        true_lower = true[true <= true_q]
        pred_upper = pred[pred >= pred_q]
        true_upper = true[true >= true_q]
        total += abs(float(pred_lower.mean()) - float(true_lower.mean()))
        total += abs(float(pred_upper.mean()) - float(true_upper.mean()))
    return float(total / float(len(levels)))


def central_interval_coverage_width_1d(
    predicted: NDArray,
    truth: NDArray,
    coverages: NDArray,
) -> Tuple[Dict[float, float], Dict[float, float]]:
    pred = _as_nonempty_1d(predicted, "predicted")
    true = _as_nonempty_1d(truth, "truth")
    coverage_values: Dict[float, float] = {}
    width_values: Dict[float, float] = {}
    for coverage in np.asarray(coverages, dtype=np.float64):
        lo = float(np.quantile(pred, (1.0 - coverage) / 2.0, method="linear"))
        hi = float(np.quantile(pred, (1.0 + coverage) / 2.0, method="linear"))
        coverage_values[float(coverage)] = float(np.mean((true >= lo) & (true <= hi)))
        width_values[float(coverage)] = float(hi - lo)
    return coverage_values, width_values


def pit_histogram_1d(predicted: NDArray, truth: NDArray, bins: int = 10) -> NDArray:
    values = empirical_cdf_values_1d(predicted, truth)
    counts, _ = np.histogram(values, bins=np.linspace(0.0, 1.0, bins + 1))
    return counts.astype(np.float64) / max(values.size, 1)


def continuous_conditional_w1_grid(
    w_grid: NDArray,
    true_sampler: Callable[[NDArray, int], NDArray],
    learned_sampler: Callable[[NDArray, int], NDArray],
    n_per_w: int,
) -> float:
    total = 0.0
    for w in np.asarray(w_grid, dtype=np.float32):
        y_true = true_sampler(w, n_per_w)
        y_learned = learned_sampler(w, n_per_w)
        total += sample_wasserstein_1_1d(y_true, y_learned)
    return float(total / len(w_grid))
