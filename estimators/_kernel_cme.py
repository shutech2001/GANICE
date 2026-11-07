from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import pdist


FloatArray = NDArray[np.float64]


def as_2d_float(x: NDArray) -> FloatArray:
    array = np.asarray(x, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("expected a one- or two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError("array values must all be finite")
    return array


def as_covariates(x: NDArray, *, x_dim: int) -> FloatArray:
    """Validate covariates while accepting one row as a one-dimensional input."""

    if x_dim < 1:
        raise ValueError("x_dim must be positive")
    array = np.asarray(x, dtype=np.float64)
    if array.ndim == 0:
        if x_dim != 1:
            raise ValueError(f"a scalar covariate is incompatible with x_dim={x_dim}")
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        if x_dim == 1:
            array = array[:, None]
        elif array.size == x_dim:
            array = array[None, :]
        else:
            raise ValueError(f"one-dimensional x must contain exactly {x_dim} entries")
    if array.ndim != 2:
        raise ValueError("x must be a one- or two-dimensional array")
    if array.shape[1] != x_dim:
        raise ValueError(f"x has {array.shape[1]} columns, expected {x_dim}")
    if not np.all(np.isfinite(array)):
        raise ValueError("x values must all be finite")
    return array


def median_pairwise_bandwidth(
    values: NDArray,
    *,
    seed: int,
    max_pairs: Optional[int] = 20_000,
    include_zero: bool = False,
) -> float:
    """Median Euclidean distance, using a deterministic pair sample if requested.

    Passing ``max_pairs=None`` computes the median over every unordered pair.
    ``include_zero=True`` follows a literal all-pairs heuristic when duplicate
    observations occur; a positive-distance fallback handles a zero median.
    """

    array = as_2d_float(values)
    n = array.shape[0]
    if n < 2:
        return 1.0
    if max_pairs is not None and max_pairs < 1:
        raise ValueError("max_pairs must be positive or None")
    total_pairs = n * (n - 1) // 2
    if max_pairs is None or total_pairs <= max_pairs:
        distances = pdist(array, metric="euclidean")
    else:
        rng = np.random.default_rng(seed)
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n - 1, size=max_pairs)
        j = j + (j >= i)
        distances = np.linalg.norm(array[i] - array[j], axis=1)
    finite = distances[np.isfinite(distances)]
    usable = finite if include_zero else finite[finite > 0.0]
    if usable.size == 0:
        return 1.0
    bandwidth = float(np.median(usable))
    if bandwidth <= 0.0:
        positive = finite[finite > 0.0]
        if positive.size == 0:
            return 1.0
        bandwidth = float(np.median(positive))
    return float(max(bandwidth, np.finfo(np.float64).eps ** 0.25))


def rbf_kernel(left: NDArray, right: NDArray, bandwidth: float) -> FloatArray:
    left_array = as_2d_float(left)
    right_array = as_2d_float(right)
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive and finite")
    left_sq = np.sum(left_array * left_array, axis=1, keepdims=True)
    right_sq = np.sum(right_array * right_array, axis=1, keepdims=True).T
    squared_distance = np.maximum(left_sq + right_sq - 2.0 * left_array @ right_array.T, 0.0)
    return np.exp(-0.5 * squared_distance / float(bandwidth * bandwidth))


def project_rows_to_simplex(values: NDArray) -> FloatArray:
    """Euclidean projection of every row onto the probability simplex."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("values must be a non-empty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("values must all be finite")
    ordered = np.sort(matrix, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    indices = np.arange(1, matrix.shape[1] + 1, dtype=np.float64)
    positive = ordered - cumulative / indices > 0.0
    rho = np.maximum(np.sum(positive, axis=1) - 1, 0)
    theta = cumulative[np.arange(matrix.shape[0]), rho] / (rho + 1.0)
    projected = np.maximum(matrix - theta[:, None], 0.0)
    normalizer = projected.sum(axis=1, keepdims=True)
    return projected / np.maximum(normalizer, np.finfo(np.float64).tiny)


@dataclass(frozen=True, slots=True)
class RKHSProjectionResult:
    """Result of projecting signed empirical weights to an RKHS-nearest law."""

    weights: FloatArray
    residuals: FloatArray
    dual_gaps: FloatArray
    iterations: NDArray[np.int64]
    converged: NDArray[np.bool_]


def project_rows_to_rkhs_simplex(
    values: NDArray,
    outcome_gram: NDArray,
    *,
    max_iter: int = 128,
    tolerance: float = 1e-8,
) -> RKHSProjectionResult:
    r"""Project signed weights to probability laws in the outcome RKHS.

    For each signed row ``beta`` and outcome Gram matrix ``L``, this solves

    ``min_{p in simplex} 0.5 * (p - beta) @ L @ (p - beta)``.

    The deterministic Frank--Wolfe iterations use an exact line search.  This
    is the finite-support, kernel-herding-style projection relevant to a CME:
    unlike Euclidean simplex projection, it minimizes error between the target
    and projected *kernel mean embeddings*.  Each iteration is linear in the
    support size after the initial Gram multiplications.
    """

    signed = np.asarray(values, dtype=np.float64)
    gram = np.asarray(outcome_gram, dtype=np.float64)
    if signed.ndim != 2 or signed.shape[1] == 0:
        raise ValueError("values must be a matrix with non-empty support")
    if not np.all(np.isfinite(signed)):
        raise ValueError("values must all be finite")
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("outcome_gram must be square")
    if gram.shape[0] != signed.shape[1]:
        raise ValueError("outcome_gram size must match the weight support")
    if not np.all(np.isfinite(gram)):
        raise ValueError("outcome_gram values must all be finite")
    if not np.allclose(gram, gram.T, rtol=1e-10, atol=1e-12):
        raise ValueError("outcome_gram must be symmetric")
    if np.any(np.diag(gram) < -1e-12):
        raise ValueError("outcome_gram cannot have a negative diagonal")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)):
        raise TypeError("max_iter must be an integer")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if isinstance(tolerance, (bool, np.bool_)):
        raise TypeError("tolerance must be a real number")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be non-negative and finite")

    n_rows, support_size = signed.shape
    if n_rows == 0:
        return RKHSProjectionResult(
            weights=np.empty_like(signed),
            residuals=np.empty(0, dtype=np.float64),
            dual_gaps=np.empty(0, dtype=np.float64),
            iterations=np.empty(0, dtype=np.int64),
            converged=np.empty(0, dtype=np.bool_),
        )

    kernel = 0.5 * (gram + gram.T)
    diagonal = np.diag(kernel)
    target_scores = signed @ kernel

    rows = np.arange(n_rows)
    # Warm-start from the previous Euclidean adapter.  It is feasible, usually
    # has a compact active set, and ensures every accepted RKHS line-search step
    # can only improve on the former adapter's RKHS objective.
    weights = project_rows_to_simplex(signed)
    current_scores = weights @ kernel
    iterations = np.zeros(n_rows, dtype=np.int64)
    dual_gaps = np.full(n_rows, np.inf, dtype=np.float64)
    denominator_floor = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(diagonal))))

    for _ in range(int(max_iter)):
        gradient = current_scores - target_scores
        fw_vertices = np.argmin(gradient, axis=1)
        weighted_gradient = np.sum(weights * gradient, axis=1)
        dual_gaps = np.maximum(weighted_gradient - gradient[rows, fw_vertices], 0.0)
        active = dual_gaps > tolerance
        if not np.any(active):
            break

        active_rows = rows[active]
        active_weights = weights[active]
        active_scores = current_scores[active]
        active_gradient = gradient[active]
        active_fw_vertices = fw_vertices[active]
        current_norm2 = np.sum(active_weights * active_scores, axis=1)

        # Away steps can remove previously selected atoms and substantially
        # accelerate convergence over plain kernel herding while retaining the
        # same deterministic Frank--Wolfe dual certificate.
        support = active_weights > 1e-14
        away_vertices = np.argmax(np.where(support, active_gradient, -np.inf), axis=1)
        away_gaps = (
            active_gradient[np.arange(active_rows.size), away_vertices]
            - weighted_gradient[active]
        )
        use_away = (
            away_gaps > dual_gaps[active]
        ) & (
            active_weights[np.arange(active_rows.size), away_vertices] < 1.0 - 1e-14
        )

        fw_local = np.flatnonzero(~use_away)
        if fw_local.size:
            fw_rows = active_rows[fw_local]
            vertices = active_fw_vertices[fw_local]
            scores = active_scores[fw_local]
            row_weights = active_weights[fw_local]
            direction_norm2 = (
                diagonal[vertices]
                - 2.0 * scores[np.arange(fw_local.size), vertices]
                + current_norm2[fw_local]
            )
            step = np.ones(fw_local.size, dtype=np.float64)
            stable = direction_norm2 > denominator_floor
            step[stable] = np.clip(
                dual_gaps[fw_rows][stable] / direction_norm2[stable],
                0.0,
                1.0,
            )
            weights[fw_rows] *= 1.0 - step[:, None]
            weights[fw_rows, vertices] += step
            current_scores[fw_rows] = (
                (1.0 - step[:, None]) * scores
                + step[:, None] * kernel[vertices]
            )

        away_local = np.flatnonzero(use_away)
        if away_local.size:
            away_rows = active_rows[away_local]
            vertices = away_vertices[away_local]
            scores = active_scores[away_local]
            row_weights = active_weights[away_local]
            direction_norm2 = (
                diagonal[vertices]
                - 2.0 * scores[np.arange(away_local.size), vertices]
                + current_norm2[away_local]
            )
            vertex_weights = row_weights[np.arange(away_local.size), vertices]
            max_step = vertex_weights / np.maximum(1.0 - vertex_weights, 1e-14)
            step = max_step.copy()
            stable = direction_norm2 > denominator_floor
            step[stable] = np.minimum(
                away_gaps[away_local][stable] / direction_norm2[stable],
                max_step[stable],
            )
            weights[away_rows] *= 1.0 + step[:, None]
            weights[away_rows, vertices] -= step
            current_scores[away_rows] = (
                (1.0 + step[:, None]) * scores
                - step[:, None] * kernel[vertices]
            )
        iterations[active] += 1

    # Remove only roundoff-level violations and recompute all diagnostics from
    # the returned weights.  This keeps probability validity explicit.
    weights = np.maximum(weights, 0.0)
    weights /= weights.sum(axis=1, keepdims=True)
    current_scores = weights @ kernel
    gradient = current_scores - target_scores
    dual_gaps = np.maximum(
        np.sum(weights * gradient, axis=1) - np.min(gradient, axis=1),
        0.0,
    )
    target_norm2 = np.sum(signed * target_scores, axis=1)
    projected_norm2 = np.sum(weights * current_scores, axis=1)
    cross_inner = np.sum(weights * target_scores, axis=1)
    residual2 = np.maximum(projected_norm2 - 2.0 * cross_inner + target_norm2, 0.0)
    residuals = np.sqrt(residual2)
    return RKHSProjectionResult(
        weights=weights,
        residuals=residuals,
        dual_gaps=dual_gaps,
        iterations=iterations,
        converged=dual_gaps <= tolerance,
    )


@dataclass(slots=True)
class KernelCMEArm:
    """Treatment-arm kernel conditional mean embedding fitted by vector-valued KRR."""

    x: FloatArray
    y: FloatArray
    x_bandwidth: float
    ridge: float
    ridge_times_n: bool
    _cholesky: tuple[FloatArray, bool]

    @classmethod
    def fit(
        cls,
        x: NDArray,
        y: NDArray,
        *,
        x_bandwidth: float,
        ridge: float,
        ridge_times_n: bool,
    ) -> "KernelCMEArm":
        x_array = as_2d_float(x)
        y_array = as_2d_float(y)
        if x_array.shape[0] != y_array.shape[0] or x_array.shape[0] == 0:
            raise ValueError("an arm must contain aligned, non-empty x and y arrays")
        gram = rbf_kernel(x_array, x_array, x_bandwidth)
        regularization = float(ridge) * (x_array.shape[0] if ridge_times_n else 1.0)
        gram.flat[:: gram.shape[0] + 1] += max(regularization, 1e-10)
        factor = cho_factor(gram, lower=True, check_finite=False)
        return cls(
            x=x_array,
            y=y_array,
            x_bandwidth=float(x_bandwidth),
            ridge=float(ridge),
            ridge_times_n=bool(ridge_times_n),
            _cholesky=factor,
        )

    def signed_weights(self, query_x: NDArray) -> FloatArray:
        query = as_2d_float(query_x)
        if query.shape[1] != self.x.shape[1]:
            raise ValueError(
                f"query_x has {query.shape[1]} columns, expected {self.x.shape[1]}"
            )
        cross = rbf_kernel(query, self.x, self.x_bandwidth)
        return cho_solve(self._cholesky, cross.T, check_finite=False).T

    def probability_weights(self, query_x: NDArray) -> FloatArray:
        """Closest empirical-distribution weights to the signed KRR coefficients."""

        return project_rows_to_simplex(self.signed_weights(query_x))

    def sample(
        self,
        query_x: NDArray,
        *,
        n_per_x: int,
        rng: np.random.Generator,
    ) -> FloatArray:
        if n_per_x < 1:
            raise ValueError("n_per_x must be positive")
        weights = self.probability_weights(query_x)
        samples = np.empty((weights.shape[0], n_per_x, self.y.shape[1]), dtype=np.float64)
        for row, probabilities in enumerate(weights):
            selected = rng.choice(self.y.shape[0], size=n_per_x, replace=True, p=probabilities)
            samples[row] = self.y[selected]
        return samples

    def projected_mean(self, query_x: NDArray) -> FloatArray:
        return self.probability_weights(query_x) @ self.y


def deterministic_arm_subsample(
    x: NDArray,
    y: NDArray,
    *,
    max_samples: Optional[int],
    seed: int,
) -> tuple[FloatArray, FloatArray]:
    x_array = as_2d_float(x)
    y_array = as_2d_float(y)
    if max_samples is None or x_array.shape[0] <= max_samples:
        return x_array, y_array
    if max_samples < 2:
        raise ValueError("max_arm_samples must be at least two")
    indices = np.random.default_rng(seed).choice(x_array.shape[0], size=max_samples, replace=False)
    indices.sort()
    return x_array[indices], y_array[indices]
