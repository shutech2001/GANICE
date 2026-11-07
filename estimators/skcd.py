from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, cast

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit

from ._kernel_cme import (
    KernelCMEArm,
    as_2d_float,
    as_covariates,
    median_pairwise_bandwidth,
    rbf_kernel,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(slots=True)
class SKCDConfig:
    """Configuration for the binary-treatment SKCD estimator.

    Jain et al. (2026) use two-fold cross-fitting, Gaussian kernels selected by
    the median heuristic, and unscaled ``K + 1e-3 I`` regularization for the
    outcome conditional mean embeddings.  Their experiments tune a LightGBM
    propensity model by Optuna on an internal 80/20 split with ten-round early
    stopping.  That is the default backend here.  A deterministic
    ridge-regularized logistic backend remains available for ablations and
    tiny samples that cannot support the internal stratified split.
    """

    x_dim: int
    num_treatments: int = 2
    n_folds: int = 2
    ridge: float = 1e-3
    x_bandwidth: Optional[float] = None
    outcome_bandwidth: Optional[float] = None
    propensity_backend: str = "lightgbm_optuna"
    propensity_validation_fraction: float = 0.2
    propensity_optuna_trials: int = 30
    propensity_early_stopping_rounds: int = 10
    propensity_max_estimators: int = 2_000
    propensity_n_jobs: int = 1
    propensity_l2: float = 1e-2
    propensity_degree: int = 2
    propensity_clip: float = 1e-6
    propensity_max_iter: int = 250
    # The paper computes each median over every unordered pair.  A positive
    # integer opts into deterministic pair subsampling for very large data.
    median_max_pairs: Optional[int] = None
    bootstrap_batch_size: int = 16
    wald_gamma: float = 1.0 / 3.0
    wald_epsilon: Optional[float] = None
    wald_epsilon_floor: float = 1e-8
    seed: int = 313


@dataclass(slots=True)
class SKCDDiagnostics:
    fold_sizes: dict[int, int] = field(default_factory=dict)
    fold_arm_sizes: dict[int, dict[int, int]] = field(default_factory=dict)
    propensity_optimizer_success: dict[int, bool] = field(default_factory=dict)
    propensity_optimizer_messages: dict[int, str] = field(default_factory=dict)
    propensity_best_scores: dict[int, float] = field(default_factory=dict)
    propensity_best_iterations: dict[int, int] = field(default_factory=dict)
    propensity_best_params: dict[int, dict[str, object]] = field(default_factory=dict)
    propensity_tuning_sizes: dict[int, dict[str, int]] = field(default_factory=dict)
    oof_propensity_mean: float = 0.0
    oof_propensity_min: float = 0.0
    oof_propensity_max: float = 0.0
    x_bandwidth: float = 0.0
    outcome_bandwidth: float = 0.0
    bandwidth_selection: str = "exact full-sample all-pairs median heuristic"
    propensity_model: str = "cross-fitted LightGBM tuned by Optuna"
    sampling_adapter: str = (
        "Equal mixture of the two fold-specific arm outcome-nuisance empirical "
        "laws, each obtained by deterministic Euclidean simplex projection of "
        "that fold's signed CME coefficients"
    )


@dataclass(frozen=True, slots=True)
class SKCDTestResult:
    statistic: float
    critical_value: float
    p_value: float
    reject: bool
    alpha: float
    n_bootstrap: int
    test_type: str = "mmd"
    epsilon: Optional[float] = None


@dataclass(frozen=True, slots=True)
class SKCDWitnessBand:
    """MMD slice confidence band for the native SKCD witness."""

    witness: FloatArray
    lower: FloatArray
    upper: FloatArray
    half_width: FloatArray
    critical_values: FloatArray
    alpha: float
    n_bootstrap: int


@dataclass(slots=True)
class _RidgeLogistic:
    mean: FloatArray
    scale: FloatArray
    coefficients: FloatArray
    degree: int
    optimizer_success: bool
    optimizer_message: str

    @staticmethod
    def _expand(standardized_x: FloatArray, degree: int) -> FloatArray:
        if degree == 1:
            return standardized_x
        columns = [standardized_x]
        p = standardized_x.shape[1]
        interactions = [
            standardized_x[:, left] * standardized_x[:, right]
            for left in range(p)
            for right in range(left, p)
        ]
        if interactions:
            columns.append(np.column_stack(interactions))
        return np.column_stack(columns)

    @classmethod
    def fit(
        cls,
        x: FloatArray,
        treatment: IntArray,
        *,
        degree: int,
        l2: float,
        max_iter: int,
    ) -> "_RidgeLogistic":
        if np.unique(treatment).size != 2:
            raise ValueError("each propensity training fold must contain both treatment arms")
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale > 1e-10, scale, 1.0)
        features = cls._expand((x - mean) / scale, degree)
        design = np.column_stack([np.ones(x.shape[0], dtype=np.float64), features])
        labels = treatment.astype(np.float64)

        initial = np.zeros(design.shape[1], dtype=np.float64)
        treated_fraction = float(np.clip(labels.mean(), 1e-6, 1.0 - 1e-6))
        initial[0] = np.log(treated_fraction / (1.0 - treated_fraction))

        def objective(coefficients: FloatArray) -> tuple[float, FloatArray]:
            logits = design @ coefficients
            probabilities = expit(logits)
            value = np.mean(np.logaddexp(0.0, logits) - labels * logits)
            value += 0.5 * l2 * float(coefficients[1:] @ coefficients[1:])
            gradient = design.T @ (probabilities - labels) / float(labels.size)
            gradient[1:] += l2 * coefficients[1:]
            return float(value), gradient

        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-7},
        )
        coefficients = np.asarray(result.x, dtype=np.float64)
        if not np.all(np.isfinite(coefficients)):
            raise RuntimeError("propensity optimization produced non-finite coefficients")
        return cls(
            mean=mean,
            scale=scale,
            coefficients=coefficients,
            degree=degree,
            optimizer_success=bool(result.success),
            optimizer_message=str(result.message),
        )

    def predict(self, x: FloatArray) -> FloatArray:
        features = self._expand((x - self.mean) / self.scale, self.degree)
        design = np.column_stack([np.ones(x.shape[0], dtype=np.float64), features])
        return np.asarray(expit(design @ self.coefficients), dtype=np.float64)


def _stratified_train_validation_split(
    treatment: IntArray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[IntArray, IntArray]:
    """Seeded internal split that keeps both arms in training and validation."""

    rng = np.random.default_rng(seed)
    train_parts: list[IntArray] = []
    validation_parts: list[IntArray] = []
    for arm in (0, 1):
        indices = np.flatnonzero(treatment == arm).astype(np.int64)
        if indices.size < 2:
            raise ValueError(
                "LightGBM+Optuna propensity tuning needs at least two "
                "observations from each arm in every outer training fold"
            )
        rng.shuffle(indices)
        n_validation = int(round(validation_fraction * float(indices.size)))
        n_validation = min(max(n_validation, 1), indices.size - 1)
        validation_parts.append(indices[:n_validation])
        train_parts.append(indices[n_validation:])
    train = np.concatenate(train_parts).astype(np.int64)
    validation = np.concatenate(validation_parts).astype(np.int64)
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


@dataclass(slots=True)
class _LightGBMOptunaPropensity:
    """Paper-experiment propensity fit with deterministic broad Optuna search."""

    model: object
    best_score: float
    best_iteration: int
    best_params: dict[str, object]
    tuning_train_size: int
    tuning_validation_size: int

    @staticmethod
    def _binary_logloss(labels: IntArray, probabilities: FloatArray) -> float:
        clipped = np.clip(
            np.asarray(probabilities, dtype=np.float64),
            1e-15,
            1.0 - 1e-15,
        )
        targets = labels.astype(np.float64)
        return float(
            -np.mean(
                targets * np.log(clipped)
                + (1.0 - targets) * np.log1p(-clipped)
            )
        )

    @classmethod
    def fit(
        cls,
        x: FloatArray,
        treatment: IntArray,
        *,
        validation_fraction: float,
        n_trials: int,
        early_stopping_rounds: int,
        max_estimators: int,
        n_jobs: int,
        seed: int,
    ) -> "_LightGBMOptunaPropensity":
        try:
            import lightgbm as lgb
            import optuna
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError(
                "propensity_backend='lightgbm_optuna' requires the "
                "lightgbm and optuna project dependencies"
            ) from exc

        inner_train, inner_validation = _stratified_train_validation_split(
            treatment,
            validation_fraction=validation_fraction,
            seed=seed + 1,
        )
        previous_optuna_verbosity = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(
            seed=seed + 2,
            n_startup_trials=min(10, max(1, n_trials // 3)),
        )
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
        )

        fixed_params: dict[str, Any] = {
            "objective": "binary",
            "metric": "binary_logloss",
            "n_estimators": max_estimators,
            "random_state": seed,
            "bagging_seed": seed,
            "feature_fraction_seed": seed,
            "data_random_seed": seed,
            "deterministic": True,
            "force_col_wise": True,
            "n_jobs": n_jobs,
            "verbosity": -1,
        }

        def objective(trial: Any) -> float:
            params: dict[str, Any] = {
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    1e-3,
                    2e-1,
                    log=True,
                ),
                "num_leaves": trial.suggest_int(
                    "num_leaves",
                    7,
                    255,
                    log=True,
                ),
                "max_depth": trial.suggest_categorical(
                    "max_depth",
                    [-1, 3, 5, 7, 10, 15],
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples",
                    2,
                    100,
                    log=True,
                ),
                "min_split_gain": trial.suggest_float(
                    "min_split_gain",
                    0.0,
                    1.0,
                ),
                "feature_fraction": trial.suggest_float(
                    "feature_fraction",
                    0.5,
                    1.0,
                ),
                "bagging_fraction": trial.suggest_float(
                    "bagging_fraction",
                    0.5,
                    1.0,
                ),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha",
                    1e-8,
                    10.0,
                    log=True,
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda",
                    1e-8,
                    10.0,
                    log=True,
                ),
                "max_bin": trial.suggest_categorical(
                    "max_bin",
                    [63, 127, 255],
                ),
            }
            model = lgb.LGBMClassifier(**fixed_params, **params)
            model.fit(
                x[inner_train],
                treatment[inner_train],
                eval_X=x[inner_validation],
                eval_y=treatment[inner_validation],
                eval_metric="binary_logloss",
                callbacks=[
                    lgb.early_stopping(
                        early_stopping_rounds,
                        verbose=False,
                    ),
                    lgb.log_evaluation(period=0),
                ],
            )
            best_iteration = int(
                max(1, getattr(model, "best_iteration_", max_estimators))
            )
            trial.set_user_attr("best_iteration", best_iteration)
            probabilities = np.asarray(
                cast(
                    NDArray,
                    model.predict_proba(
                        x[inner_validation],
                        num_iteration=best_iteration,
                    ),
                )[:, 1],
                dtype=np.float64,
            )
            return cls._binary_logloss(
                treatment[inner_validation],
                probabilities,
            )

        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                show_progress_bar=False,
            )
        finally:
            optuna.logging.set_verbosity(previous_optuna_verbosity)
        best_params = dict(study.best_trial.params)
        best_iteration = int(study.best_trial.user_attrs["best_iteration"])
        final_params = dict(fixed_params)
        final_params.update(best_params)
        final_params["n_estimators"] = best_iteration
        final_model = lgb.LGBMClassifier(**final_params)
        final_model.fit(
            x,
            treatment,
            callbacks=[lgb.log_evaluation(period=0)],
        )
        return cls(
            model=final_model,
            best_score=float(study.best_value),
            best_iteration=best_iteration,
            best_params={**best_params, "n_estimators": best_iteration},
            tuning_train_size=int(inner_train.size),
            tuning_validation_size=int(inner_validation.size),
        )

    def predict(self, x: FloatArray) -> FloatArray:
        predict_proba = getattr(self.model, "predict_proba")
        return np.asarray(predict_proba(x)[:, 1], dtype=np.float64)


@dataclass(slots=True)
class _FoldNuisance:
    train_fold: int
    heldout_fold: int
    train_indices: IntArray
    heldout_indices: IntArray
    propensity: _RidgeLogistic | _LightGBMOptunaPropensity
    arms: dict[int, KernelCMEArm]
    arm_indices: dict[int, IntArray]


def _binary_treatment(values: NDArray, *, n_rows: int) -> IntArray:
    raw = np.asarray(values)
    if raw.ndim > 2 or (raw.ndim == 2 and raw.shape[1] != 1):
        raise ValueError("treatment must be one-dimensional")
    raw = raw.reshape(-1)
    if raw.size != n_rows:
        raise ValueError("x, treatment, and y must have the same number of rows")
    if not np.all(np.isin(raw, [0, 1])):
        raise ValueError("SKCD requires treatment values 0 and 1")
    return raw.astype(np.int64)


def _outcome_points(values: NDArray, *, outcome_dim: int) -> FloatArray:
    """Validate scalar or vector outcome evaluation points."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        if outcome_dim != 1:
            raise ValueError(
                f"a scalar outcome is incompatible with outcome_dim={outcome_dim}"
            )
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        if outcome_dim == 1:
            array = array[:, None]
        elif array.size == outcome_dim:
            array = array[None, :]
        else:
            raise ValueError(
                f"one-dimensional outcome must contain exactly {outcome_dim} entries"
            )
    if array.ndim != 2:
        raise ValueError("outcome must be a one- or two-dimensional array")
    if array.shape[1] != outcome_dim:
        raise ValueError(
            f"outcome has {array.shape[1]} columns, expected {outcome_dim}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("outcome values must all be finite")
    return array


def _stratified_two_folds(treatment: IntArray, *, seed: int) -> tuple[IntArray, IntArray]:
    rng = np.random.default_rng(seed)
    per_fold: list[list[IntArray]] = [[], []]
    for arm in (0, 1):
        indices = np.flatnonzero(treatment == arm).astype(np.int64)
        if indices.size < 2:
            raise ValueError("each treatment arm needs at least two observations for two-fold cross-fitting")
        rng.shuffle(indices)
        split = np.array_split(indices, 2)
        per_fold[0].append(split[0])
        per_fold[1].append(split[1])
    folds = []
    for parts in per_fold:
        fold = np.concatenate(parts).astype(np.int64)
        fold.sort()
        folds.append(fold)
    return folds[0], folds[1]


class SKCD:
    """Smoothed Kernel Conditional Discrepancy of Jain et al. (2026).

    The native estimator is the two-fold cross-fitted, Hilbert-valued AIPW/EIF
    estimator of the global SCoDiTE.  The MMD and regularized Wald tests, the
    witness, and the MMD slice confidence band evaluate that native estimand
    using the paper's exact arm-specific coefficient equations.

    SKCD is a global test, not a generative conditional-distribution model.  To
    participate in GANICE's distributional benchmark, ``sample_potential`` uses
    only its fitted outcome nuisances: it equally averages the two fold-specific
    arm outcome-nuisance laws.  Each fold's signed CME coefficients are first
    projected onto the probability simplex, after which the two valid empirical
    laws are mixed with weight one half.  Sampling this mixture is an evaluation
    adapter; it is not the SKCD statistic or a claim from Jain et al.
    """

    def __init__(self, config: SKCDConfig) -> None:
        if config.x_dim < 1:
            raise ValueError("x_dim must be positive")
        if config.num_treatments != 2:
            raise ValueError("SKCD is defined here only for binary treatment")
        if config.n_folds != 2:
            raise ValueError("Jain et al. (2026) SKCD uses exactly two folds")
        if not np.isfinite(config.ridge) or config.ridge <= 0.0:
            raise ValueError("ridge must be positive and finite")
        if config.x_bandwidth is not None and (
            not np.isfinite(config.x_bandwidth) or config.x_bandwidth <= 0.0
        ):
            raise ValueError("x_bandwidth must be positive and finite")
        if config.outcome_bandwidth is not None and (
            not np.isfinite(config.outcome_bandwidth) or config.outcome_bandwidth <= 0.0
        ):
            raise ValueError("outcome_bandwidth must be positive and finite")
        if config.propensity_backend not in {
            "lightgbm_optuna",
            "ridge_logistic",
        }:
            raise ValueError(
                "propensity_backend must be 'lightgbm_optuna' or 'ridge_logistic'"
            )
        if (
            not np.isfinite(config.propensity_validation_fraction)
            or not 0.0 < config.propensity_validation_fraction < 0.5
        ):
            raise ValueError(
                "propensity_validation_fraction must lie strictly between 0 and 0.5"
            )
        if (
            isinstance(config.propensity_optuna_trials, (bool, np.bool_))
            or not isinstance(config.propensity_optuna_trials, (int, np.integer))
            or config.propensity_optuna_trials < 1
        ):
            raise ValueError("propensity_optuna_trials must be a positive integer")
        if (
            isinstance(config.propensity_early_stopping_rounds, (bool, np.bool_))
            or not isinstance(
                config.propensity_early_stopping_rounds,
                (int, np.integer),
            )
            or config.propensity_early_stopping_rounds < 1
        ):
            raise ValueError(
                "propensity_early_stopping_rounds must be a positive integer"
            )
        if (
            isinstance(config.propensity_max_estimators, (bool, np.bool_))
            or not isinstance(config.propensity_max_estimators, (int, np.integer))
            or config.propensity_max_estimators < 2
        ):
            raise ValueError("propensity_max_estimators must be an integer at least two")
        if (
            isinstance(config.propensity_n_jobs, (bool, np.bool_))
            or not isinstance(config.propensity_n_jobs, (int, np.integer))
            or config.propensity_n_jobs == 0
        ):
            raise ValueError("propensity_n_jobs must be a nonzero integer")
        if not np.isfinite(config.propensity_l2) or config.propensity_l2 < 0.0:
            raise ValueError("propensity_l2 must be non-negative and finite")
        if config.propensity_degree not in (1, 2):
            raise ValueError("propensity_degree must be 1 or 2")
        if not 0.0 < config.propensity_clip < 0.5:
            raise ValueError("propensity_clip must lie strictly between 0 and 0.5")
        if config.propensity_max_iter < 1:
            raise ValueError("propensity_max_iter must be positive")
        if config.median_max_pairs is not None and config.median_max_pairs < 1:
            raise ValueError("median_max_pairs must be positive or None")
        if config.bootstrap_batch_size < 1:
            raise ValueError("bootstrap_batch_size must be positive")
        if not np.isfinite(config.wald_gamma) or config.wald_gamma <= 0.0:
            raise ValueError("wald_gamma must be positive and finite")
        if config.wald_epsilon is not None and (
            not np.isfinite(config.wald_epsilon)
            or not 0.0 < config.wald_epsilon <= 1.0
        ):
            raise ValueError("wald_epsilon must lie in (0, 1]")
        if (
            not np.isfinite(config.wald_epsilon_floor)
            or not 0.0 < config.wald_epsilon_floor <= 1.0
        ):
            raise ValueError("wald_epsilon_floor must lie in (0, 1]")

        self.config = config
        self.x_train: Optional[FloatArray] = None
        self.treatment_train: Optional[IntArray] = None
        self.y_train: Optional[FloatArray] = None
        self.folds: tuple[IntArray, IntArray] = (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
        self.nuisance_models: list[_FoldNuisance] = []
        self.x_bandwidth = 1.0
        self.outcome_bandwidth = 1.0
        self.diagnostics = SKCDDiagnostics()
        self._coefficient_cache: Optional[FloatArray] = None
        self._plugin_coefficient_cache: Optional[FloatArray] = None
        self._mmd_matrix_cache: Optional[FloatArray] = None
        self._wald_components_cache: Optional[
            tuple[FloatArray, FloatArray, FloatArray]
        ] = None

    def _check_fitted(self) -> tuple[FloatArray, IntArray, FloatArray]:
        if (
            self.x_train is None
            or self.treatment_train is None
            or self.y_train is None
            or len(self.nuisance_models) != 2
        ):
            raise RuntimeError("SKCD must be fit before prediction")
        return self.x_train, self.treatment_train, self.y_train

    def fit(self, x: NDArray, treatment: NDArray, y: NDArray) -> "SKCD":
        x_array = as_covariates(x, x_dim=self.config.x_dim)
        y_array = as_2d_float(y)
        treatment_array = _binary_treatment(treatment, n_rows=x_array.shape[0])
        if y_array.shape[0] != x_array.shape[0]:
            raise ValueError("x, treatment, and y must have the same number of rows")
        if np.unique(treatment_array).size != 2:
            raise ValueError("both treatment arms must be present")

        folds = _stratified_two_folds(treatment_array, seed=self.config.seed)
        if self.config.x_bandwidth is None:
            x_bandwidth = median_pairwise_bandwidth(
                x_array,
                seed=self.config.seed + 1_000,
                max_pairs=self.config.median_max_pairs,
                include_zero=True,
            )
        else:
            x_bandwidth = float(self.config.x_bandwidth)
        if self.config.outcome_bandwidth is None:
            outcome_bandwidth = median_pairwise_bandwidth(
                y_array,
                seed=self.config.seed + 2_000,
                max_pairs=self.config.median_max_pairs,
                include_zero=True,
            )
        else:
            outcome_bandwidth = float(self.config.outcome_bandwidth)

        diagnostics = SKCDDiagnostics(
            x_bandwidth=x_bandwidth,
            outcome_bandwidth=outcome_bandwidth,
            propensity_model=(
                "cross-fitted LightGBM tuned by Optuna on an internal "
                f"stratified {100.0 * (1.0 - self.config.propensity_validation_fraction):g}/"
                f"{100.0 * self.config.propensity_validation_fraction:g} split"
                if self.config.propensity_backend == "lightgbm_optuna"
                else (
                    "cross-fitted ridge logistic regression with standardized "
                    "polynomial features"
                )
            ),
            bandwidth_selection=(
                "exact full-sample all-pairs median heuristic"
                if self.config.median_max_pairs is None
                else (
                    "full-sample median heuristic with at most "
                    f"{self.config.median_max_pairs} deterministic sampled pairs"
                )
            ),
        )
        for fold_id, indices in enumerate(folds):
            diagnostics.fold_sizes[fold_id] = int(indices.size)
            diagnostics.fold_arm_sizes[fold_id] = {
                arm: int(np.sum(treatment_array[indices] == arm)) for arm in (0, 1)
            }

        nuisance_models: list[_FoldNuisance] = []
        propensity: _LightGBMOptunaPropensity | _RidgeLogistic
        oof_propensity = np.empty(x_array.shape[0], dtype=np.float64)
        for train_fold in (0, 1):
            heldout_fold = 1 - train_fold
            train_indices = folds[train_fold]
            heldout_indices = folds[heldout_fold]
            if self.config.propensity_backend == "lightgbm_optuna":
                propensity = _LightGBMOptunaPropensity.fit(
                    x_array[train_indices],
                    treatment_array[train_indices],
                    validation_fraction=self.config.propensity_validation_fraction,
                    n_trials=self.config.propensity_optuna_trials,
                    early_stopping_rounds=self.config.propensity_early_stopping_rounds,
                    max_estimators=self.config.propensity_max_estimators,
                    n_jobs=self.config.propensity_n_jobs,
                    seed=self.config.seed + 10_000 + 1_000 * train_fold,
                )
                diagnostics.propensity_optimizer_success[train_fold] = True
                diagnostics.propensity_optimizer_messages[train_fold] = (
                    "Optuna study completed; final LightGBM refit on the full "
                    "outer training fold"
                )
                diagnostics.propensity_best_scores[train_fold] = (
                    propensity.best_score
                )
                diagnostics.propensity_best_iterations[train_fold] = (
                    propensity.best_iteration
                )
                diagnostics.propensity_best_params[train_fold] = dict(
                    propensity.best_params
                )
                diagnostics.propensity_tuning_sizes[train_fold] = {
                    "outer_train": int(train_indices.size),
                    "inner_train": propensity.tuning_train_size,
                    "inner_validation": propensity.tuning_validation_size,
                }
            else:
                propensity = _RidgeLogistic.fit(
                    x_array[train_indices],
                    treatment_array[train_indices],
                    degree=self.config.propensity_degree,
                    l2=self.config.propensity_l2,
                    max_iter=self.config.propensity_max_iter,
                )
                diagnostics.propensity_optimizer_success[train_fold] = (
                    propensity.optimizer_success
                )
                diagnostics.propensity_optimizer_messages[train_fold] = (
                    propensity.optimizer_message
                )
            oof_propensity[heldout_indices] = np.clip(
                propensity.predict(x_array[heldout_indices]),
                self.config.propensity_clip,
                1.0 - self.config.propensity_clip,
            )

            arms: dict[int, KernelCMEArm] = {}
            arm_indices: dict[int, IntArray] = {}
            for arm in (0, 1):
                indices = train_indices[treatment_array[train_indices] == arm]
                arm_indices[arm] = indices
                arms[arm] = KernelCMEArm.fit(
                    x_array[indices],
                    y_array[indices],
                    x_bandwidth=x_bandwidth,
                    ridge=self.config.ridge,
                    ridge_times_n=False,
                )
            nuisance_models.append(
                _FoldNuisance(
                    train_fold=train_fold,
                    heldout_fold=heldout_fold,
                    train_indices=train_indices,
                    heldout_indices=heldout_indices,
                    propensity=propensity,
                    arms=arms,
                    arm_indices=arm_indices,
                )
            )

        diagnostics.oof_propensity_mean = float(oof_propensity.mean())
        diagnostics.oof_propensity_min = float(oof_propensity.min())
        diagnostics.oof_propensity_max = float(oof_propensity.max())

        self.x_train = x_array
        self.treatment_train = treatment_array
        self.y_train = y_array
        self.folds = folds
        self.nuisance_models = nuisance_models
        self.x_bandwidth = x_bandwidth
        self.outcome_bandwidth = outcome_bandwidth
        self.diagnostics = diagnostics
        self._coefficient_cache = None
        self._plugin_coefficient_cache = None
        self._mmd_matrix_cache = None
        self._wald_components_cache = None
        return self

    def _adapter_law(self, x: FloatArray, arm: int) -> tuple[FloatArray, FloatArray]:
        self._check_fitted()
        probability_blocks: list[FloatArray] = []
        support_blocks: list[FloatArray] = []
        for nuisance in self.nuisance_models:
            probability_blocks.append(0.5 * nuisance.arms[arm].probability_weights(x))
            support_blocks.append(nuisance.arms[arm].y)
        probabilities = np.concatenate(probability_blocks, axis=1)
        support = np.concatenate(support_blocks, axis=0)
        return probabilities, support

    def predict_potential_outcomes(self, x: NDArray) -> NDArray[np.float32]:
        self._check_fitted()
        query = as_covariates(x, x_dim=self.config.x_dim)
        assert self.y_train is not None
        outcome_dim = self.y_train.shape[1]
        means = np.empty((query.shape[0], 2, outcome_dim), dtype=np.float64)
        for arm in (0, 1):
            probabilities, support = self._adapter_law(query, arm)
            means[:, arm, :] = probabilities @ support
        if outcome_dim == 1:
            return means[:, :, 0].astype(np.float32)
        return means.astype(np.float32)

    def sample_potential(
        self,
        x: NDArray,
        treatment: NDArray | int,
        n_per_x: int = 1,
        seed: Optional[int] = None,
    ) -> NDArray[np.float32]:
        _, _, y_train = self._check_fitted()
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
        if raw_treatment.size != query.shape[0] or not np.all(np.isin(raw_treatment, [0, 1])):
            raise ValueError("treatment must be scalar 0/1 or align with x")
        treatment_array = raw_treatment.astype(np.int64)

        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        samples = np.empty(
            (query.shape[0], int(n_per_x), y_train.shape[1]),
            dtype=np.float64,
        )
        for arm in (0, 1):
            mask = treatment_array == arm
            if not np.any(mask):
                continue
            probabilities, support = self._adapter_law(query[mask], arm)
            for row, row_probabilities in enumerate(probabilities):
                selected = rng.choice(
                    support.shape[0],
                    size=int(n_per_x),
                    replace=True,
                    p=row_probabilities,
                )
                samples[np.flatnonzero(mask)[row], :, :] = support[selected]
        return samples.astype(np.float32)

    def _coefficient_matrix(self) -> FloatArray:
        x_train, treatment_train, _ = self._check_fitted()
        if self._coefficient_cache is not None:
            return self._coefficient_cache
        n = x_train.shape[0]
        coefficients = np.zeros((n, n), dtype=np.float64)
        clip = self.config.propensity_clip
        for nuisance in self.nuisance_models:
            heldout = nuisance.heldout_indices
            n_heldout = heldout.size
            propensity = np.clip(
                nuisance.propensity.predict(x_train[heldout]),
                clip,
                1.0 - clip,
            )
            a = treatment_train[heldout].astype(np.float64)
            normalizer = 2.0 * float(n_heldout)
            diagonal = a / propensity - (1.0 - a) / (1.0 - propensity)
            coefficients[heldout, heldout] = diagonal / normalizer

            treated_indices = nuisance.arm_indices[1]
            treated_weights = nuisance.arms[1].signed_weights(x_train[heldout])
            treated_augmentation = 1.0 - a / propensity
            coefficients[np.ix_(heldout, treated_indices)] += (
                treated_augmentation[:, None] * treated_weights / normalizer
            )

            control_indices = nuisance.arm_indices[0]
            control_weights = nuisance.arms[0].signed_weights(x_train[heldout])
            control_augmentation = (1.0 - a) / (1.0 - propensity) - 1.0
            coefficients[np.ix_(heldout, control_indices)] += (
                control_augmentation[:, None] * control_weights / normalizer
            )
        self._coefficient_cache = coefficients
        return coefficients

    def _plugin_coefficient_matrix(self) -> FloatArray:
        """Exact arm-specific plug-in matrix ``E`` from the paper.

        This matrix is needed to center the fold-specific estimated influence
        functions in the reported Wald covariance estimator.  The exact
        entry-wise equation is used because the manuscript's later fast block
        shortcut does not preserve separate treated and control CME weights.
        """

        x_train, _, _ = self._check_fitted()
        if self._plugin_coefficient_cache is not None:
            return self._plugin_coefficient_cache
        n = x_train.shape[0]
        coefficients = np.zeros((n, n), dtype=np.float64)
        for nuisance in self.nuisance_models:
            heldout = nuisance.heldout_indices
            normalizer = 2.0 * float(heldout.size)
            treated_indices = nuisance.arm_indices[1]
            control_indices = nuisance.arm_indices[0]
            coefficients[np.ix_(heldout, treated_indices)] += (
                nuisance.arms[1].signed_weights(x_train[heldout]) / normalizer
            )
            coefficients[np.ix_(heldout, control_indices)] -= (
                nuisance.arms[0].signed_weights(x_train[heldout]) / normalizer
            )
        self._plugin_coefficient_cache = coefficients
        return coefficients

    def _wald_components(
        self,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return the EIF Gram, observed projection, and bootstrap map.

        If ``q_i = phi_i / sqrt(2 n_s)`` for observation ``i`` in held-out
        fold ``s``, the returned objects are

        ``H[i,j] = <q_i,q_j>``,
        ``v[i] = <q_i, psi_bar>``, and
        ``R @ xi = (<q_i, Delta_xi>)_i``.

        This is algebraically equivalent to the manuscript's regularized
        covariance operator but works directly from its exact ``C`` and ``E``
        equations, avoiding the inconsistent arm-collapsing block shortcut.
        """

        x_train, _, y_train = self._check_fitted()
        if self._wald_components_cache is not None:
            return self._wald_components_cache

        n = x_train.shape[0]
        coefficients = self._coefficient_matrix()
        plugin = self._plugin_coefficient_matrix()
        covariate_gram = rbf_kernel(x_train, x_train, self.x_bandwidth)
        outcome_gram = rbf_kernel(y_train, y_train, self.outcome_bandwidth)

        fold_id = np.empty(n, dtype=np.int64)
        row_component = np.zeros_like(coefficients)
        fold_center: list[FloatArray] = []
        for index, fold in enumerate(self.folds):
            n_fold = float(fold.size)
            fold_id[fold] = index
            row_component[fold] = np.sqrt(2.0 * n_fold) * coefficients[fold]
            center = np.zeros_like(plugin)
            center[fold] = np.sqrt(2.0 / n_fold) * plugin[fold]
            fold_center.append(center)

        # q_i = one-row(row_component_i) - fold_center_s.
        row_gram = covariate_gram * (
            row_component @ outcome_gram @ row_component.T
        )
        row_center = np.empty((n, 2), dtype=np.float64)
        center_gram = np.empty((2, 2), dtype=np.float64)
        for left_fold, center in enumerate(fold_center):
            transformed = covariate_gram @ center @ outcome_gram
            row_center[:, left_fold] = np.sum(
                row_component * transformed,
                axis=1,
            )
            for right_fold, other_center in enumerate(fold_center):
                center_gram[left_fold, right_fold] = float(
                    np.sum(center * (covariate_gram @ other_center @ outcome_gram))
                )

        selected_row_center = row_center[:, fold_id]
        eif_gram = (
            row_gram
            - selected_row_center
            - selected_row_center.T
            + center_gram[fold_id[:, None], fold_id[None, :]]
        )
        eif_gram = 0.5 * (eif_gram + eif_gram.T)

        transformed_estimate = covariate_gram @ coefficients @ outcome_gram
        observed_projection = np.sum(
            row_component * transformed_estimate,
            axis=1,
        )
        center_estimate = np.asarray(
            [
                float(np.sum(center * transformed_estimate))
                for center in fold_center
            ],
            dtype=np.float64,
        )
        observed_projection -= center_estimate[fold_id]

        # For Delta_xi = sum_j xi_j row_j(C), build the linear map from
        # multipliers to all <q_i, Delta_xi> inner products.
        bootstrap_projection = covariate_gram * (
            row_component @ outcome_gram @ coefficients.T
        )
        center_rows = np.empty((2, n), dtype=np.float64)
        for index, center in enumerate(fold_center):
            transformed = covariate_gram @ center @ outcome_gram
            center_rows[index] = np.sum(coefficients * transformed, axis=1)
        bootstrap_projection -= center_rows[fold_id]

        self._wald_components_cache = (
            eif_gram,
            observed_projection,
            bootstrap_projection,
        )
        return self._wald_components_cache

    def witness(
        self,
        x: NDArray,
        outcome: NDArray,
        *,
        paired: bool = False,
    ) -> NDArray[np.float32]:
        """Evaluate the native cross-fitted AIPW SCoDiTE witness.

        The default is the Cartesian grid of all supplied covariate and outcome
        points.  Set ``paired=True`` to evaluate row-aligned ``(x_i, y_i)``
        pairs.
        """

        x_train, _, y_train = self._check_fitted()
        query = as_covariates(x, x_dim=self.config.x_dim)
        outcome_array = _outcome_points(
            outcome,
            outcome_dim=y_train.shape[1],
        )
        covariate_kernel = rbf_kernel(query, x_train, self.x_bandwidth)
        outcome_kernel = rbf_kernel(y_train, outcome_array, self.outcome_bandwidth)
        values = covariate_kernel @ self._coefficient_matrix() @ outcome_kernel
        if paired:
            if outcome_array.shape[0] != query.shape[0]:
                raise ValueError(
                    "paired witness evaluation requires equal row counts for x and outcome"
                )
            return np.diag(values).astype(np.float32)
        return values.astype(np.float32)

    def predict_witness(
        self,
        x: NDArray,
        outcome: NDArray,
        *,
        paired: bool = False,
    ) -> NDArray[np.float32]:
        return self.witness(x, outcome, paired=paired)

    def _mmd_quadratic_matrix(self) -> FloatArray:
        x_train, _, y_train = self._check_fitted()
        if self._mmd_matrix_cache is not None:
            return self._mmd_matrix_cache
        coefficients = self._coefficient_matrix()
        covariate_gram = rbf_kernel(x_train, x_train, self.x_bandwidth)
        outcome_gram = rbf_kernel(y_train, y_train, self.outcome_bandwidth)
        embedded_inner_products = coefficients @ outcome_gram @ coefficients.T
        matrix = covariate_gram * embedded_inner_products
        matrix = 0.5 * (matrix + matrix.T)
        self._mmd_matrix_cache = matrix
        return matrix

    def global_mmd2(self) -> float:
        """Squared RKHS norm of the native cross-fitted AIPW SCoDiTE estimate."""

        value = float(self._mmd_quadratic_matrix().sum())
        return max(0.0, value)

    def mmd_test_statistic(self) -> float:
        x_train, _, _ = self._check_fitted()
        return float(x_train.shape[0]) * self.global_mmd2()

    @staticmethod
    def _validate_inference_arguments(
        n_bootstrap: int,
        alpha: float,
    ) -> tuple[int, float]:
        if isinstance(n_bootstrap, (bool, np.bool_)) or not isinstance(
            n_bootstrap,
            (int, np.integer),
        ):
            raise TypeError("n_bootstrap must be an integer")
        if n_bootstrap < 1:
            raise ValueError("n_bootstrap must be positive")
        if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one")
        return int(n_bootstrap), float(alpha)

    def _draw_fold_multipliers(
        self,
        rng: np.random.Generator,
        n_draws: int,
    ) -> FloatArray:
        x_train, _, _ = self._check_fitted()
        multipliers = np.zeros((n_draws, x_train.shape[0]), dtype=np.float64)
        for fold in self.folds:
            probabilities = np.full(
                fold.size,
                1.0 / float(fold.size),
                dtype=np.float64,
            )
            counts = rng.multinomial(fold.size, probabilities, size=n_draws)
            multipliers[:, fold] = counts - 1.0
        return multipliers

    def global_mmd_test(
        self,
        *,
        n_bootstrap: int = 1_000,
        alpha: float = 0.05,
        seed: Optional[int] = None,
    ) -> SKCDTestResult:
        """Fixed-nuisance, split-multinomial SKCD-MMD test."""

        x_train, _, _ = self._check_fitted()
        n_bootstrap, alpha = self._validate_inference_arguments(
            n_bootstrap,
            alpha,
        )

        n = x_train.shape[0]
        matrix = self._mmd_quadratic_matrix()
        statistic = max(0.0, float(n) * float(matrix.sum()))
        rng = np.random.default_rng(self.config.seed + 50_000 if seed is None else seed)
        bootstrap = np.empty(n_bootstrap, dtype=np.float64)
        batch_size = self.config.bootstrap_batch_size
        for start in range(0, n_bootstrap, batch_size):
            stop = min(start + batch_size, n_bootstrap)
            multipliers = self._draw_fold_multipliers(rng, stop - start)
            values = float(n) * np.sum((multipliers @ matrix) * multipliers, axis=1)
            bootstrap[start:stop] = np.maximum(values, 0.0)

        critical_value = float(np.quantile(bootstrap, 1.0 - alpha, method="higher"))
        p_value = float((1 + np.sum(bootstrap >= statistic)) / (n_bootstrap + 1))
        return SKCDTestResult(
            statistic=statistic,
            critical_value=critical_value,
            p_value=p_value,
            reject=bool(statistic > critical_value),
            alpha=alpha,
            n_bootstrap=n_bootstrap,
            test_type="mmd",
        )

    def _resolve_wald_epsilon(
        self,
        epsilon: Optional[float],
        eif_gram: FloatArray,
    ) -> float:
        selected = self.config.wald_epsilon if epsilon is None else epsilon
        if selected is not None:
            if isinstance(selected, (bool, np.bool_)) or (
                not np.isfinite(selected) or not 0.0 < selected <= 1.0
            ):
                raise ValueError("epsilon must lie in (0, 1]")
            return float(selected)

        # Jain et al.'s experimental heuristic.  Their consistency theorem is
        # for fixed epsilon, so this data-dependent choice is reported rather
        # than presented as theory-mandated.
        covariance_trace = max(0.0, float(np.trace(eif_gram)))
        raw = (
            self.config.wald_gamma
            * covariance_trace
            / (1.0 + self.config.wald_gamma * covariance_trace)
        )
        return float(
            np.clip(
                raw,
                self.config.wald_epsilon_floor,
                1.0,
            )
        )

    def _wald_factor(
        self,
        epsilon: Optional[float],
    ) -> tuple[float, tuple[FloatArray, bool], FloatArray, FloatArray]:
        eif_gram, observed_projection, bootstrap_projection = (
            self._wald_components()
        )
        selected_epsilon = self._resolve_wald_epsilon(epsilon, eif_gram)
        regularized = (
            selected_epsilon * np.eye(eif_gram.shape[0], dtype=np.float64)
            + (1.0 - selected_epsilon) * eif_gram
        )
        regularized = 0.5 * (regularized + regularized.T)
        factor = cho_factor(
            regularized,
            lower=True,
            check_finite=False,
        )
        return (
            selected_epsilon,
            factor,
            observed_projection,
            bootstrap_projection,
        )

    def wald_test_statistic(
        self,
        *,
        epsilon: Optional[float] = None,
    ) -> float:
        """Regularized SKCD-Wald statistic ``n <Omega_n psi_bar, psi_bar>``."""

        x_train, _, _ = self._check_fitted()
        (
            selected_epsilon,
            factor,
            observed_projection,
            _,
        ) = self._wald_factor(epsilon)
        norm_squared = self.global_mmd2()
        inverse_projection = cho_solve(
            factor,
            observed_projection,
            check_finite=False,
        )
        adjusted = (
            norm_squared
            - (1.0 - selected_epsilon)
            * float(observed_projection @ inverse_projection)
        ) / selected_epsilon
        return max(0.0, float(x_train.shape[0]) * adjusted)

    def global_wald_test(
        self,
        *,
        n_bootstrap: int = 1_000,
        alpha: float = 0.05,
        seed: Optional[int] = None,
        epsilon: Optional[float] = None,
    ) -> SKCDTestResult:
        """Fixed-nuisance, split-multinomial SKCD-Wald test."""

        x_train, _, _ = self._check_fitted()
        n_bootstrap, alpha = self._validate_inference_arguments(
            n_bootstrap,
            alpha,
        )
        n = x_train.shape[0]
        (
            selected_epsilon,
            factor,
            observed_projection,
            bootstrap_projection,
        ) = self._wald_factor(epsilon)
        norm_squared = self.global_mmd2()
        inverse_projection = cho_solve(
            factor,
            observed_projection,
            check_finite=False,
        )
        observed_adjusted = (
            norm_squared
            - (1.0 - selected_epsilon)
            * float(observed_projection @ inverse_projection)
        ) / selected_epsilon
        statistic = max(0.0, float(n) * observed_adjusted)

        matrix = self._mmd_quadratic_matrix()
        rng = np.random.default_rng(
            self.config.seed + 60_000 if seed is None else seed
        )
        bootstrap = np.empty(n_bootstrap, dtype=np.float64)
        batch_size = self.config.bootstrap_batch_size
        for start in range(0, n_bootstrap, batch_size):
            stop = min(start + batch_size, n_bootstrap)
            multipliers = self._draw_fold_multipliers(rng, stop - start)
            norm_values = np.sum(
                (multipliers @ matrix) * multipliers,
                axis=1,
            )
            projected = multipliers @ bootstrap_projection.T
            inverse_projected = cho_solve(
                factor,
                projected.T,
                check_finite=False,
            ).T
            corrections = np.sum(projected * inverse_projected, axis=1)
            values = (
                float(n)
                / selected_epsilon
                * (
                    norm_values
                    - (1.0 - selected_epsilon) * corrections
                )
            )
            bootstrap[start:stop] = np.maximum(values, 0.0)

        critical_value = float(
            np.quantile(bootstrap, 1.0 - alpha, method="higher")
        )
        p_value = float(
            (1 + np.sum(bootstrap >= statistic)) / (n_bootstrap + 1)
        )
        return SKCDTestResult(
            statistic=statistic,
            critical_value=critical_value,
            p_value=p_value,
            reject=bool(statistic > critical_value),
            alpha=alpha,
            n_bootstrap=n_bootstrap,
            test_type="wald",
            epsilon=selected_epsilon,
        )

    def witness_confidence_band(
        self,
        x: NDArray,
        outcome: NDArray,
        *,
        n_bootstrap: int = 1_000,
        alpha: float = 0.05,
        seed: Optional[int] = None,
    ) -> SKCDWitnessBand:
        """Paper's MMD slice band, uniform over outcomes for each fixed ``x``.

        The constant half-width ``sqrt(q_x / n)`` uses the unit diagonal of
        this implementation's Gaussian outcome kernel.
        """

        x_train, _, y_train = self._check_fitted()
        query = as_covariates(x, x_dim=self.config.x_dim)
        outcome_array = _outcome_points(
            outcome,
            outcome_dim=y_train.shape[1],
        )
        n_bootstrap, alpha = self._validate_inference_arguments(
            n_bootstrap,
            alpha,
        )
        n = x_train.shape[0]
        witness = np.asarray(
            self.witness(query, outcome_array, paired=False),
            dtype=np.float64,
        )
        coefficients = self._coefficient_matrix()
        outcome_gram = rbf_kernel(y_train, y_train, self.outcome_bandwidth)
        embedded_inner_products = (
            coefficients @ outcome_gram @ coefficients.T
        )
        query_kernel = rbf_kernel(query, x_train, self.x_bandwidth)

        rng = np.random.default_rng(
            self.config.seed + 70_000 if seed is None else seed
        )
        bootstrap = np.empty(
            (n_bootstrap, query.shape[0]),
            dtype=np.float64,
        )
        batch_size = self.config.bootstrap_batch_size
        slice_matrices = [
            np.outer(kernel_row, kernel_row) * embedded_inner_products
            for kernel_row in query_kernel
        ]
        for start in range(0, n_bootstrap, batch_size):
            stop = min(start + batch_size, n_bootstrap)
            multipliers = self._draw_fold_multipliers(rng, stop - start)
            for query_index, matrix in enumerate(slice_matrices):
                values = float(n) * np.sum(
                    (multipliers @ matrix) * multipliers,
                    axis=1,
                )
                bootstrap[start:stop, query_index] = np.maximum(values, 0.0)

        critical_values = np.asarray(
            np.quantile(
                bootstrap,
                1.0 - alpha,
                axis=0,
                method="higher",
            ),
            dtype=np.float64,
        )
        half_width = np.sqrt(np.maximum(critical_values, 0.0) / float(n))
        lower = witness - half_width[:, None]
        upper = witness + half_width[:, None]
        return SKCDWitnessBand(
            witness=witness,
            lower=lower,
            upper=upper,
            half_width=half_width,
            critical_values=critical_values,
            alpha=alpha,
            n_bootstrap=n_bootstrap,
        )

    def test_global_mmd(
        self,
        *,
        n_bootstrap: int = 1_000,
        alpha: float = 0.05,
        seed: Optional[int] = None,
    ) -> SKCDTestResult:
        return self.global_mmd_test(n_bootstrap=n_bootstrap, alpha=alpha, seed=seed)
