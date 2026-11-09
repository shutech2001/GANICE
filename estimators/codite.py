from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

from ._kernel_cme import (
    KernelCMEArm,
    RKHSProjectionResult,
    as_2d_float,
    as_covariates,
    deterministic_arm_subsample,
    median_pairwise_bandwidth,
    project_rows_to_rkhs_simplex,
    rbf_kernel,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(slots=True)
class CoDiTEConfig:
    """Numerical choices for the binary/scalar MMD-associated CoDiTE.

    Park et al. do not report numerical bandwidths, ridge values, a tuning
    grid, or a label-resample count.  The common ``ridge`` and
    ``x_bandwidth`` fields are backward-compatible fallbacks; the arm-specific
    fields override them when supplied.  ``max_arm_samples`` is a
    repository-level evaluation-budget approximation and must be ``None`` for
    the paper's full-sample KCD test.
    """

    x_dim: int
    num_treatments: int = 2
    ridge: float = 1e-3
    ridge_control: Optional[float] = None
    ridge_treated: Optional[float] = None
    x_bandwidth: Optional[float] = None
    x_bandwidth_control: Optional[float] = None
    x_bandwidth_treated: Optional[float] = None
    outcome_bandwidth: Optional[float] = None
    max_arm_samples: Optional[int] = None
    propensity_ridge: float = 1e-2
    propensity_bandwidth: Optional[float] = None
    propensity_clip: float = 1e-6
    propensity_max_iter: int = 250
    adapter_projection_max_iter: int = 256
    adapter_projection_tolerance: float = 5e-4
    seed: int = 211


@dataclass(slots=True)
class CoDiTEDiagnostics:
    """Fit metadata and cumulative evaluation-adapter diagnostics.

    ``adapter_query_count`` counts arm/query pairs passed through the adapter.
    ``negative_raw_fraction`` is over their signed KRR coefficients, and
    ``rkhs_projection_residual`` is the mean RKHS norm between each raw CME and
    its projected empirical probability law.
    """

    arm_sizes: dict[int, int] = field(default_factory=dict)
    fitted_arm_sizes: dict[int, int] = field(default_factory=dict)
    x_bandwidths: dict[int, float] = field(default_factory=dict)
    ridges: dict[int, float] = field(default_factory=dict)
    outcome_bandwidth: float = 0.0
    adapter_query_count: int = 0
    negative_raw_fraction: float = 0.0
    rkhs_projection_residual: float = 0.0
    rkhs_projection_residual_max: float = 0.0
    rkhs_projection_dual_gap: float = 0.0
    rkhs_projection_converged_fraction: float = 1.0
    sampling_adapter: str = (
        "Tolerance-controlled deterministic away-step Frank-Wolfe projection of "
        "the arm-specific signed CME/KRR coefficients onto the empirical probability "
        "simplex, minimizing distance in the Gaussian outcome-kernel RKHS"
    )


@dataclass(frozen=True, slots=True)
class CoDiTEKCDTestResult:
    """Algorithm 1 result with the documented finite-sample empty-arm guard."""

    statistic: float
    resampled_statistics: FloatArray
    p_value: float
    reject: bool
    alpha: float
    num_resamples: int
    propensity_scores: FloatArray
    propensity_model: str
    propensity_optimizer_success: Optional[bool]
    propensity_optimizer_message: Optional[str]
    degenerate_label_redraws: int


@dataclass(slots=True)
class _KernelLogisticPropensity:
    """Deterministic kernel-logistic propensity model used by Algorithm 1.

    Park et al. suggest kernel logistic regression as an example but do not
    specify its kernel, bandwidth, or regularization.  This implementation
    uses the same Gaussian input-kernel family as the CME and exposes all
    otherwise-unspecified numerical choices through ``CoDiTEConfig``.
    """

    x: FloatArray
    bandwidth: float
    coefficients: FloatArray
    intercept: float
    optimizer_success: bool
    optimizer_message: str

    @classmethod
    def fit(
        cls,
        x: FloatArray,
        treatment: IntArray,
        *,
        bandwidth: float,
        ridge: float,
        max_iter: int,
    ) -> "_KernelLogisticPropensity":
        if np.unique(treatment).size != 2:
            raise ValueError("kernel logistic regression requires both treatment arms")
        gram = rbf_kernel(x, x, bandwidth)
        # Optimize in an orthonormal empirical-RKHS feature basis rather than
        # directly in the often ill-conditioned dual coefficients.  If
        # K = Q diag(s) Q^T and Phi = Q diag(sqrt(s)), then f = Phi w and the
        # RKHS penalty is ||w||^2.  The returned alpha reconstructs
        # f(x) = k(x, X)^T alpha for predictions at new covariates.
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
        cutoff = max(
            np.finfo(np.float64).eps * gram.shape[0] * float(eigenvalues[-1]),
            np.finfo(np.float64).tiny,
        )
        retained = eigenvalues > cutoff
        retained_values = eigenvalues[retained]
        retained_vectors = eigenvectors[:, retained]
        features = retained_vectors * np.sqrt(retained_values)[None, :]
        labels = treatment.astype(np.float64)
        initial = np.zeros(features.shape[1] + 1, dtype=np.float64)
        treated_fraction = float(np.clip(labels.mean(), 1e-8, 1.0 - 1e-8))
        initial[0] = np.log(treated_fraction / (1.0 - treated_fraction))

        def objective(parameters: FloatArray) -> tuple[float, FloatArray]:
            intercept = parameters[0]
            feature_coefficients = parameters[1:]
            logits = intercept + features @ feature_coefficients
            probabilities = expit(logits)
            residual = probabilities - labels
            value = np.mean(np.logaddexp(0.0, logits) - labels * logits)
            value += 0.5 * ridge * float(
                feature_coefficients @ feature_coefficients
            )
            gradient = np.empty_like(parameters)
            gradient[0] = residual.mean()
            gradient[1:] = (
                features.T @ residual / labels.size
                + ridge * feature_coefficients
            )
            return float(value), gradient

        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": int(max_iter), "ftol": 1e-12, "gtol": 1e-7},
        )
        parameters = np.asarray(result.x, dtype=np.float64)
        if not np.all(np.isfinite(parameters)):
            raise RuntimeError("propensity optimization produced non-finite coefficients")
        coefficients = retained_vectors @ (
            parameters[1:] / np.sqrt(retained_values)
        )
        return cls(
            x=x,
            bandwidth=float(bandwidth),
            coefficients=coefficients,
            intercept=float(parameters[0]),
            optimizer_success=bool(result.success),
            optimizer_message=str(result.message),
        )

    def predict(self, x: FloatArray) -> FloatArray:
        logits = self.intercept + rbf_kernel(x, self.x, self.bandwidth) @ self.coefficients
        return np.asarray(expit(logits), dtype=np.float64)


class CoDiTE:
    """MMD-associated CoDiTE estimator of Park et al. (2021), Section 4.

    The paper's estimator is an arm-specific conditional mean embedding (CME),
    fitted by vector-valued kernel ridge regression with ``K + n * ridge * I``.
    ``predict_codite_mmd``, ``predict_codite_mmd2``, ``kcd_statistic``, and
    ``witness`` use those signed coefficients exactly.  ``kcd_test`` follows
    the paper's propensity-conditional treatment-label resampling Algorithm 1,
    with a reported finite-sample guard for label draws containing only one arm.

    A population CME represents its conditional law, but its finite-sample
    signed KRR estimate need not be the embedding of a probability measure.
    The experiment runner nevertheless requires samples, so
    ``sample_potential`` is an explicitly labelled adapter: for each treatment
    arm it uses deterministic Frank--Wolfe iterations to approximately solve,
    to configured tolerance, for the empirical probability law whose Gaussian
    outcome-kernel mean embedding is closest to the signed estimate, then
    samples observed arm outcomes with those weights.  This evaluation adapter
    is not a native CoDiTE estimand or an additional claim made in the original
    paper.  The paper's separate Section 5 U-statistic regression for selected
    scalar summaries is not this CME estimator and is intentionally not
    represented by the sampling adapter.
    """

    def __init__(self, config: CoDiTEConfig) -> None:
        if config.x_dim < 1:
            raise ValueError("x_dim must be positive")
        if config.num_treatments != 2:
            raise ValueError("Park et al. (2021) formulate CoDiTE for binary treatment")
        if not np.isfinite(config.ridge) or config.ridge <= 0.0:
            raise ValueError("ridge must be positive and finite")
        for name in ("ridge_control", "ridge_treated"):
            value = getattr(config, name)
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "x_bandwidth",
            "x_bandwidth_control",
            "x_bandwidth_treated",
            "outcome_bandwidth",
            "propensity_bandwidth",
        ):
            value = getattr(config, name)
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be positive and finite")
        if config.max_arm_samples is not None and config.max_arm_samples < 2:
            raise ValueError("max_arm_samples must be at least two")
        if not np.isfinite(config.propensity_ridge) or config.propensity_ridge <= 0.0:
            raise ValueError("propensity_ridge must be positive and finite")
        if not 0.0 < config.propensity_clip < 0.5:
            raise ValueError("propensity_clip must lie strictly between zero and one half")
        if isinstance(config.propensity_max_iter, (bool, np.bool_)) or not isinstance(
            config.propensity_max_iter,
            (int, np.integer),
        ):
            raise TypeError("propensity_max_iter must be an integer")
        if config.propensity_max_iter < 1:
            raise ValueError("propensity_max_iter must be positive")
        if isinstance(config.adapter_projection_max_iter, (bool, np.bool_)) or not isinstance(
            config.adapter_projection_max_iter,
            (int, np.integer),
        ):
            raise TypeError("adapter_projection_max_iter must be an integer")
        if config.adapter_projection_max_iter < 1:
            raise ValueError("adapter_projection_max_iter must be positive")
        if isinstance(config.adapter_projection_tolerance, (bool, np.bool_)):
            raise TypeError("adapter_projection_tolerance must be a real number")
        if (
            not np.isfinite(config.adapter_projection_tolerance)
            or config.adapter_projection_tolerance < 0.0
        ):
            raise ValueError("adapter_projection_tolerance must be non-negative and finite")
        self.config = config
        self.arms: dict[int, KernelCMEArm] = {}
        self.x_train: Optional[FloatArray] = None
        self.treatment_train: Optional[IntArray] = None
        self.y_train: Optional[FloatArray] = None
        self.outcome_bandwidth = 1.0
        self.diagnostics = CoDiTEDiagnostics()
        self._outcome_grams: dict[tuple[int, int], NDArray[np.float64]] = {}
        self._adapter_negative_count = 0
        self._adapter_coefficient_count = 0
        self._adapter_residual_sum = 0.0
        self._adapter_residual_max = 0.0
        self._adapter_dual_gap_sum = 0.0
        self._adapter_converged_count = 0
        self._adapter_query_count = 0

    def _ridge_for_arm(self, arm: int) -> float:
        override = self.config.ridge_control if arm == 0 else self.config.ridge_treated
        return float(self.config.ridge if override is None else override)

    def _configured_bandwidth_for_arm(self, arm: int) -> Optional[float]:
        override = (
            self.config.x_bandwidth_control
            if arm == 0
            else self.config.x_bandwidth_treated
        )
        return self.config.x_bandwidth if override is None else float(override)

    def fit(self, x: NDArray, treatment: NDArray, y: NDArray) -> "CoDiTE":
        x_array = as_covariates(x, x_dim=self.config.x_dim)
        raw_treatment = np.asarray(treatment)
        if raw_treatment.ndim > 2 or (
            raw_treatment.ndim == 2 and raw_treatment.shape[1] != 1
        ):
            raise ValueError("treatment must be one-dimensional")
        raw_treatment = raw_treatment.reshape(-1)
        if not np.all(np.isin(raw_treatment, [0, 1])):
            raise ValueError("CoDiTE requires treatment values 0 and 1")
        treatment_array = raw_treatment.astype(np.int64)
        y_array = as_2d_float(y)
        if x_array.shape[0] != treatment_array.shape[0] or x_array.shape[0] != y_array.shape[0]:
            raise ValueError("x, treatment, and y must have the same number of rows")
        if y_array.shape[1] != 1:
            raise ValueError("Park et al. (2021) formulate CoDiTE for scalar outcomes")
        if np.unique(treatment_array).size != 2:
            raise ValueError("both treatment arms must be present")

        # Retain the original, unsubsampled observations: the paper's global
        # KCD statistic is evaluated at every training covariate, and its
        # conditional-resampling test reuses the aligned (X, Z, Y) rows.
        self.x_train = x_array.copy()
        self.treatment_train = treatment_array.copy()
        self.y_train = y_array.copy()
        self.arms = {}
        self._outcome_grams = {}
        self.diagnostics = CoDiTEDiagnostics()
        self._adapter_negative_count = 0
        self._adapter_coefficient_count = 0
        self._adapter_residual_sum = 0.0
        self._adapter_residual_max = 0.0
        self._adapter_dual_gap_sum = 0.0
        self._adapter_converged_count = 0
        self._adapter_query_count = 0
        for arm in (0, 1):
            mask = treatment_array == arm
            self.diagnostics.arm_sizes[arm] = int(mask.sum())
            arm_x, arm_y = deterministic_arm_subsample(
                x_array[mask],
                y_array[mask],
                max_samples=self.config.max_arm_samples,
                seed=self.config.seed + 101 * arm,
            )
            bandwidth = self._configured_bandwidth_for_arm(arm)
            if bandwidth is None:
                bandwidth = median_pairwise_bandwidth(arm_x, seed=self.config.seed + 1_000 + arm)
            ridge = self._ridge_for_arm(arm)
            self.arms[arm] = KernelCMEArm.fit(
                arm_x,
                arm_y,
                x_bandwidth=float(bandwidth),
                ridge=ridge,
                ridge_times_n=True,
            )
            self.diagnostics.fitted_arm_sizes[arm] = arm_x.shape[0]
            self.diagnostics.x_bandwidths[arm] = float(bandwidth)
            self.diagnostics.ridges[arm] = ridge

        if self.config.outcome_bandwidth is None:
            self.outcome_bandwidth = median_pairwise_bandwidth(y_array, seed=self.config.seed + 2_000)
        else:
            self.outcome_bandwidth = float(self.config.outcome_bandwidth)
        self.diagnostics.outcome_bandwidth = self.outcome_bandwidth
        return self

    def _check_fitted(self) -> None:
        if set(self.arms) != {0, 1}:
            raise RuntimeError("CoDiTE must be fit before prediction")

    def _outcome_gram(self, left_arm: int, right_arm: int) -> NDArray[np.float64]:
        key = (left_arm, right_arm)
        if key not in self._outcome_grams:
            self._outcome_grams[key] = rbf_kernel(
                self.arms[left_arm].y,
                self.arms[right_arm].y,
                self.outcome_bandwidth,
            )
        return self._outcome_grams[key]

    def _record_adapter_diagnostics(
        self,
        signed_weights: NDArray[np.float64],
        projection: RKHSProjectionResult,
    ) -> None:
        self._adapter_negative_count += int(np.count_nonzero(signed_weights < 0.0))
        self._adapter_coefficient_count += int(signed_weights.size)
        self._adapter_residual_sum += float(projection.residuals.sum())
        if projection.residuals.size:
            self._adapter_residual_max = max(
                self._adapter_residual_max,
                float(projection.residuals.max()),
            )
        self._adapter_dual_gap_sum += float(projection.dual_gaps.sum())
        self._adapter_converged_count += int(np.count_nonzero(projection.converged))
        self._adapter_query_count += int(signed_weights.shape[0])

        diagnostics = self.diagnostics
        diagnostics.adapter_query_count = self._adapter_query_count
        if self._adapter_coefficient_count:
            diagnostics.negative_raw_fraction = (
                self._adapter_negative_count / self._adapter_coefficient_count
            )
        if self._adapter_query_count:
            diagnostics.rkhs_projection_residual = (
                self._adapter_residual_sum / self._adapter_query_count
            )
            diagnostics.rkhs_projection_residual_max = self._adapter_residual_max
            diagnostics.rkhs_projection_dual_gap = (
                self._adapter_dual_gap_sum / self._adapter_query_count
            )
            diagnostics.rkhs_projection_converged_fraction = (
                self._adapter_converged_count / self._adapter_query_count
            )

    def _adapter_probability_weights(
        self,
        arm: int,
        query: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluation-adapter weights; native CoDiTE methods never call this."""

        signed_weights = self.arms[arm].signed_weights(query)
        projection = project_rows_to_rkhs_simplex(
            signed_weights,
            self._outcome_gram(arm, arm),
            max_iter=self.config.adapter_projection_max_iter,
            tolerance=self.config.adapter_projection_tolerance,
        )
        self._record_adapter_diagnostics(signed_weights, projection)
        return projection.weights

    def _codite_mmd2(self, query: FloatArray) -> FloatArray:
        beta_0 = self.arms[0].signed_weights(query)
        beta_1 = self.arms[1].signed_weights(query)
        term_00 = np.einsum(
            "bi,ij,bj->b",
            beta_0,
            self._outcome_gram(0, 0),
            beta_0,
            optimize=True,
        )
        term_11 = np.einsum(
            "bi,ij,bj->b",
            beta_1,
            self._outcome_gram(1, 1),
            beta_1,
            optimize=True,
        )
        term_01 = np.einsum(
            "bi,ij,bj->b",
            beta_0,
            self._outcome_gram(0, 1),
            beta_1,
            optimize=True,
        )
        return np.maximum(term_00 + term_11 - 2.0 * term_01, 0.0)

    def predict_codite_mmd2(self, x: NDArray) -> NDArray[np.float32]:
        """Squared conditional MMD in Lemma 4.1 of Park et al. (2021)."""

        self._check_fitted()
        query = as_covariates(x, x_dim=self.config.x_dim)
        return self._codite_mmd2(query).astype(np.float32)

    def predict_codite_mmd(self, x: NDArray) -> NDArray[np.float32]:
        """The paper's native MMD-associated CoDiTE ``U_MMD(x)``."""

        self._check_fitted()
        query = as_covariates(x, x_dim=self.config.x_dim)
        return np.sqrt(self._codite_mmd2(query)).astype(np.float32)

    def kcd_statistic(self, x: Optional[NDArray] = None) -> float:
        """Global plug-in KCD statistic: the sample mean of conditional MMD squared.

        With no argument, use every original training covariate as specified
        in the paper.  An explicit covariate design is accepted for diagnostic
        evaluation of its empirical design average; that average represents
        the population KCD criterion only when the design represents ``P_X``.
        """

        self._check_fitted()
        if x is None:
            if self.x_train is None:
                raise RuntimeError(
                    "CoDiTE must be fit before computing the KCD statistic"
                )
            query = self.x_train
        else:
            query = as_covariates(x, x_dim=self.config.x_dim)
        if query.shape[0] == 0:
            raise ValueError("x must contain at least one row")
        return float(np.mean(self._codite_mmd2(query)))

    def witness(
        self,
        x: NDArray,
        outcome: NDArray,
        *,
        paired: bool = False,
    ) -> NDArray[np.float32]:
        """Evaluate ``mu_1(x)(y) - mu_0(x)(y)`` using raw signed CME weights.

        By default every query is evaluated at every supplied outcome and the
        result has shape ``(n_query, n_outcome)``.  Set ``paired=True`` to
        evaluate row ``i`` only at outcome ``i``; equal array lengths alone do
        not silently change the meaning of the call.
        """

        self._check_fitted()
        if not isinstance(paired, (bool, np.bool_)):
            raise TypeError("paired must be boolean")
        query = as_covariates(x, x_dim=self.config.x_dim)
        outcome_array = as_2d_float(outcome)
        if outcome_array.shape[1] != 1:
            raise ValueError("outcome evaluation points must be scalar")
        beta_0 = self.arms[0].signed_weights(query)
        beta_1 = self.arms[1].signed_weights(query)
        eval_0 = beta_0 @ rbf_kernel(self.arms[0].y, outcome_array, self.outcome_bandwidth)
        eval_1 = beta_1 @ rbf_kernel(self.arms[1].y, outcome_array, self.outcome_bandwidth)
        values = eval_1 - eval_0
        if paired:
            if outcome_array.shape[0] != query.shape[0]:
                raise ValueError(
                    "paired witness evaluation requires equal row counts"
                )
            return np.diag(values).astype(np.float32)
        return values.astype(np.float32)

    def kcd_test(
        self,
        x: Optional[NDArray] = None,
        treatment: Optional[NDArray] = None,
        y: Optional[NDArray] = None,
        *,
        propensity_scores: Optional[NDArray] = None,
        num_resamples: int = 499,
        alpha: float = 0.05,
        seed: Optional[int] = None,
    ) -> CoDiTEKCDTestResult:
        """Follow Algorithm 1 with an explicit finite-sample empty-arm guard.

        This is not an ordinary permutation or outcome bootstrap.  Each null
        treatment label is drawn independently from ``Bernoulli(e_hat(x_i))``;
        both arm-specific CMEs are then refit and the KCD statistic is
        recomputed.  The strict ``>`` comparison in the published algorithm is
        reproduced literally.  A finite-sample draw containing only one arm
        cannot define the two CME fits; such a draw is redrawn, and the number
        of these explicitly conditioned-away draws is returned in
        ``degenerate_label_redraws``.
        """

        if isinstance(num_resamples, (bool, np.bool_)) or not isinstance(
            num_resamples,
            (int, np.integer),
        ):
            raise TypeError("num_resamples must be an integer")
        if num_resamples < 1:
            raise ValueError("num_resamples must be positive")
        if isinstance(alpha, (bool, np.bool_)) or not np.isfinite(alpha):
            raise TypeError("alpha must be a finite real number")
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must lie strictly between zero and one")
        if self.config.max_arm_samples is not None:
            raise ValueError(
                "the paper's KCD test uses every arm observation; "
                "set max_arm_samples=None"
            )

        supplied = (x is not None, treatment is not None, y is not None)
        if any(supplied) and not all(supplied):
            raise ValueError(
                "x, treatment, and y must either all be supplied or all omitted"
            )
        if all(supplied):
            assert x is not None and treatment is not None and y is not None
            self.fit(x, treatment, y)
        else:
            self._check_fitted()
        if (
            self.x_train is None
            or self.treatment_train is None
            or self.y_train is None
        ):
            raise RuntimeError("CoDiTE must be fit before running the KCD test")
        x_array = self.x_train
        treatment_array = self.treatment_train
        y_array = self.y_train
        observed_statistic = self.kcd_statistic()

        propensity_model_name: str
        propensity_success: Optional[bool]
        propensity_message: Optional[str]
        if propensity_scores is None:
            propensity_bandwidth = self.config.propensity_bandwidth
            if propensity_bandwidth is None:
                propensity_bandwidth = median_pairwise_bandwidth(
                    x_array,
                    seed=self.config.seed + 3_000,
                )
            propensity_model = _KernelLogisticPropensity.fit(
                x_array,
                treatment_array,
                bandwidth=float(propensity_bandwidth),
                ridge=self.config.propensity_ridge,
                max_iter=self.config.propensity_max_iter,
            )
            probabilities = propensity_model.predict(x_array)
            probabilities = np.clip(
                probabilities,
                self.config.propensity_clip,
                1.0 - self.config.propensity_clip,
            )
            propensity_model_name = "Gaussian-kernel logistic regression"
            propensity_success = propensity_model.optimizer_success
            propensity_message = propensity_model.optimizer_message
        else:
            probabilities = np.asarray(propensity_scores, dtype=np.float64).reshape(-1)
            if probabilities.shape[0] != x_array.shape[0]:
                raise ValueError("propensity_scores must contain one value per row")
            if not np.all(np.isfinite(probabilities)):
                raise ValueError("propensity_scores must all be finite")
            if np.any(probabilities <= 0.0) or np.any(probabilities >= 1.0):
                raise ValueError(
                    "propensity_scores must lie strictly between zero and one"
                )
            propensity_model_name = "user supplied"
            propensity_success = None
            propensity_message = None

        # Algorithm 1 treats k_0, k_1, l, lambda_0, and lambda_1 as fixed
        # inputs.  Freeze any data-selected bandwidths from the observed fit
        # instead of silently re-selecting them after every label resample.
        resample_config = replace(
            self.config,
            ridge_control=self.diagnostics.ridges[0],
            ridge_treated=self.diagnostics.ridges[1],
            x_bandwidth_control=self.diagnostics.x_bandwidths[0],
            x_bandwidth_treated=self.diagnostics.x_bandwidths[1],
            outcome_bandwidth=self.outcome_bandwidth,
            max_arm_samples=None,
        )
        rng = np.random.default_rng(
            self.config.seed + 4_000 if seed is None else seed
        )
        null_statistics = np.empty(int(num_resamples), dtype=np.float64)
        degenerate_redraws = 0
        for resample in range(int(num_resamples)):
            for _ in range(10_000):
                null_treatment = (
                    rng.random(x_array.shape[0]) < probabilities
                ).astype(np.int64)
                if np.unique(null_treatment).size == 2:
                    break
                degenerate_redraws += 1
            else:
                raise RuntimeError(
                    "conditional label resampling repeatedly produced an empty arm"
                )
            null_estimator = CoDiTE(
                replace(resample_config, seed=self.config.seed + 5_000 + resample)
            ).fit(x_array, null_treatment, y_array)
            null_statistics[resample] = null_estimator.kcd_statistic(x_array)

        p_value = float(
            (1 + np.count_nonzero(null_statistics > observed_statistic))
            / (int(num_resamples) + 1)
        )
        return CoDiTEKCDTestResult(
            statistic=observed_statistic,
            resampled_statistics=null_statistics,
            p_value=p_value,
            reject=bool(p_value < float(alpha)),
            alpha=float(alpha),
            num_resamples=int(num_resamples),
            propensity_scores=probabilities.copy(),
            propensity_model=propensity_model_name,
            propensity_optimizer_success=propensity_success,
            propensity_optimizer_message=propensity_message,
            degenerate_label_redraws=degenerate_redraws,
        )

    def predict_potential_outcomes(self, x: NDArray) -> NDArray[np.float32]:
        self._check_fitted()
        query = as_covariates(x, x_dim=self.config.x_dim)
        means = np.column_stack(
            [
                (self._adapter_probability_weights(arm, query) @ self.arms[arm].y)[:, 0]
                for arm in (0, 1)
            ]
        )
        return means.astype(np.float32)

    def sample_potential(
        self,
        x: NDArray,
        treatment: NDArray | int,
        n_per_x: int = 1,
        seed: Optional[int] = None,
    ) -> NDArray[np.float32]:
        self._check_fitted()
        query = as_covariates(x, x_dim=self.config.x_dim)
        if isinstance(n_per_x, (bool, np.bool_)) or not isinstance(n_per_x, (int, np.integer)):
            raise TypeError("n_per_x must be an integer")
        if n_per_x < 1:
            raise ValueError("n_per_x must be positive")
        raw_treatment = np.asarray(treatment)
        if raw_treatment.ndim > 2 or (
            raw_treatment.ndim == 2 and raw_treatment.shape[1] != 1
        ):
            raise ValueError("treatment must be scalar or one-dimensional")
        raw_treatment = raw_treatment.reshape(-1)
        if raw_treatment.size == 1:
            raw_treatment = np.full(query.shape[0], raw_treatment.item())
        if raw_treatment.shape[0] != query.shape[0] or not np.all(np.isin(raw_treatment, [0, 1])):
            raise ValueError("treatment must be scalar 0/1 or align with x")
        treatment_array = raw_treatment.astype(np.int64)
        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        samples = np.empty((query.shape[0], n_per_x, 1), dtype=np.float64)
        for arm in (0, 1):
            mask = treatment_array == arm
            if np.any(mask):
                weights = self._adapter_probability_weights(arm, query[mask])
                support = self.arms[arm].y
                arm_rows = np.flatnonzero(mask)
                for local_row, probabilities in enumerate(weights):
                    selected = rng.choice(
                        support.shape[0],
                        size=n_per_x,
                        replace=True,
                        p=probabilities,
                    )
                    samples[arm_rows[local_row]] = support[selected]
        return samples.astype(np.float32)
