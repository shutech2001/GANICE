"""Run four additional methods on the binary-treatment benchmarks and merge.

This runner reconstructs the original seeds and splits, fits only CoDiTE,
DNet, Causal-DRF, and SKCD, and merges their summaries with the benchmark
tables under ``outputs``. Repetition results stay in memory; only paper
figures, tables, and numeric CSV files are written.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
import importlib
import inspect
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, ClassVar, NotRequired, Protocol, TypedDict, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from estimators.ihdp import IHDPDistDGP  # noqa: E402
from estimators.jobs import JobsLaLondeData  # noqa: E402
from scripts import run_ganice_experiment as core  # noqa: E402


ADDITIONAL_METHODS = ("codite", "dnet", "causal_drf", "skcd")
ALL_BINARY_METHODS = (
    "ganite",
    "po_flow",
    "diff_po",
    "infs",
    "dr_learner",
    *ADDITIONAL_METHODS,
    "ganice",
)
TCGA_METHODS = ("scigan", "vcnet", "drnet", "ganice")
METHOD_LABELS = {
    "codite": "CoDiTE",
    "dnet": "DNet",
    "causal_drf": "Causal-DRF",
    "skcd": "SKCD (adapted)",
}
FIT_SEED_OFFSETS = {
    "codite": 71,
    "dnet": 73,
    "causal_drf": 79,
    "skcd": 83,
}
IHDP_QTE_SEED_OFFSETS = {
    "codite": 37_000,
    "dnet": 38_000,
    "causal_drf": 39_000,
    "skcd": 40_000,
}
JOBS_EVALUATION_SEED_OFFSETS = {
    "codite": 66_000,
    "dnet": 67_000,
    "causal_drf": 68_000,
    "skcd": 69_000,
}


Sampler = Callable[[NDArray, int, int, int | None], NDArray]


class _DataclassConfig(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]


class RepetitionResult(TypedDict):
    replication: int
    methods: dict[str, dict[str, float]]
    qte: NotRequired[core.QTECurveData]
    arm_samples: NotRequired[dict[str, NDArray]]


def _load_json(path: Path) -> Any:
    """Read historical results during migration; new runs do not write JSON."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _copy_existing_benchmark(existing_root: Path, output_root: Path, benchmark: str) -> None:
    """Copy only the artifacts used by the paper, when writing to another root."""
    source = existing_root / benchmark
    if not source.is_dir():
        raise FileNotFoundError(f"missing existing benchmark tree: {source}")
    target = output_root / benchmark
    target.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return
    stems = {
        "IHDP": {
            "metrics",
            "ihdp_dist_table",
            "ihdp_additional_metrics",
            "ihdp_qte_error",
            "ihdp_objective_ablation",
            "ihdp_interval_widths",
            "ihdp_pit_histograms",
        },
        "Jobs": {
            "metrics",
            "jobs_lalonde_table",
            "jobs_additional_metrics",
            "jobs_rct_cdf",
            "jobs_rct_cdf_treated",
            "jobs_rct_cdf_control",
            "jobs_rct_cdf_mean",
            "jobs_interval_widths",
            "jobs_pit_histograms",
        },
        "TCGA": {
            "metrics",
            "tcga_dose_table",
            "tcga_additional_metrics",
            "tcga_dose_quantile_band",
            "tcga_dose_quantile_band_mean",
            "tcga_dose_quantile_band_curves",
            "tcga_interval_widths",
            "tcga_pit_histograms",
        },
    }[benchmark]
    for path in source.iterdir():
        if path.is_file() and path.stem in stems and path.suffix in {".csv", ".tex", ".pdf"}:
            shutil.copy2(path, target / path.name)


def _read_summary(source_dir: Path, benchmark: str) -> dict[str, dict[str, float]]:
    path = source_dir / "metrics.csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            return {
                row["method"]: {
                    key: float(value) for key, value in row.items() if key != "method" and value not in (None, "")
                }
                for row in csv.DictReader(handle)
            }
    key = {"IHDP": "ihdp_dist", "Jobs": "jobs_lalonde", "TCGA": "tcga_dose"}[benchmark]
    summary = _load_json(source_dir / "metrics.json")[key]
    return summary["summary"] if benchmark == "TCGA" else summary


def _estimator_classes() -> dict[str, tuple[type, type]]:
    """Load lazily so the runner can be compiled before sibling agents finish."""

    module_specs = {
        "codite": ("estimators.codite", "CoDiTEConfig", "CoDiTE"),
        "dnet": ("estimators.dnet", "DNetConfig", "DNet"),
        "causal_drf": ("estimators.causal_drf", "CausalDRFConfig", "CausalDRF"),
        "skcd": ("estimators.skcd", "SKCDConfig", "SKCD"),
    }
    loaded: dict[str, tuple[type, type]] = {}
    for method, (module_name, config_name, estimator_name) in module_specs.items():
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            raise RuntimeError(
                f"{module_name} is unavailable; finish installing/implementing all four estimator modules first"
            ) from error
        loaded[method] = (getattr(module, config_name), getattr(module, estimator_name))
    return loaded


def _config_kwargs(
    config_class: type,
    *,
    method: str,
    x_dim: int,
    outcome_lower: float,
    outcome_upper: float,
    seed: int,
    quick: bool,
    benchmark: str,
) -> dict[str, object]:
    """Use public config fields while remaining tolerant of harmless API growth."""

    parameters = inspect.signature(config_class).parameters
    candidates: dict[str, object] = {
        "x_dim": x_dim,
        "num_treatments": 2,
        "seed": seed,
        "device": "cpu",
    }
    # DNet is unbounded in Wu et al. (2023), so do not silently turn the
    # benchmark's plotting/evaluation bounds into an architectural constraint.
    if method != "dnet":
        candidates.update(
            {
                "outcome_min": outcome_lower,
                "outcome_max": outcome_upper,
                "outcome_lower": outcome_lower,
                "outcome_upper": outcome_upper,
            }
        )
    if method == "codite":
        candidates.update(
            {
                "ridge": 1e-3,
                "max_arm_samples": 192 if quick else None,
            }
        )
    elif method == "dnet":
        candidates.update(
            {
                "base_hidden_dims": (200, 100),
                "tower_hidden_dims": (100,),
                "num_quantiles": 50,
                "batch_size": 128,
                "training_steps": 100 if quick else 1_000,
            }
        )
    elif method == "causal_drf":
        full_trees = 2_500 if benchmark == "ihdp" else 4_000
        candidates.update(
            {
                "num_trees": 40 if quick else full_trees,
                "num_groups": 8 if quick else (50 if benchmark == "ihdp" else 40),
                # The manuscript does not report an empirical kappa. Keep one
                # fixed implementation default across both binary benchmarks.
                "min_leaf_per_arm": 4,
            }
        )
    elif method == "skcd":
        candidates.update(
            {
                "ridge": 1e-3,
                "propensity_backend": "lightgbm_optuna",
                "propensity_validation_fraction": 0.2,
                "propensity_optuna_trials": 4 if quick else 30,
                "propensity_early_stopping_rounds": 10,
                "propensity_max_estimators": 300 if quick else 2_000,
                "propensity_n_jobs": 1,
            }
        )
    return {name: value for name, value in candidates.items() if name in parameters}


def _fit_estimator(
    method: str,
    x: NDArray,
    treatment: NDArray,
    y: NDArray,
    *,
    x_dim: int,
    outcome_lower: float,
    outcome_upper: float,
    seed: int,
    quick: bool,
    benchmark: str,
    fit_seed: int | None = None,
    overrides: dict[str, object] | None = None,
    classes: dict[str, tuple[type, type]] | None = None,
) -> tuple[object, dict[str, object]]:
    available = _estimator_classes() if classes is None else classes
    config_class, estimator_class = available[method]
    config_seed = seed + FIT_SEED_OFFSETS[method] if fit_seed is None else fit_seed
    kwargs = _config_kwargs(
        config_class,
        method=method,
        x_dim=x_dim,
        outcome_lower=outcome_lower,
        outcome_upper=outcome_upper,
        seed=config_seed,
        quick=quick,
        benchmark=benchmark,
    )
    if overrides:
        unknown = set(overrides).difference(inspect.signature(config_class).parameters)
        if unknown:
            raise TypeError(f"{method} candidate overrides unknown config fields: {sorted(unknown)}")
        kwargs.update(overrides)
    config = config_class(**kwargs)
    estimator = estimator_class(config)
    estimator.fit(x, treatment, y)
    config_payload = asdict(cast(_DataclassConfig, config)) if is_dataclass(config) else dict(kwargs)
    return estimator, config_payload


def _select_validation_candidate(
    method: str,
    candidate_specs: list[dict[str, object]],
    scorer: Callable[[object], float],
    x: NDArray,
    treatment: NDArray,
    y: NDArray,
    *,
    x_dim: int,
    outcome_lower: float,
    outcome_upper: float,
    seed: int,
    quick: bool,
    benchmark: str,
    validation_metric: str,
    validation_seed: int,
    validation_sample_size: int,
    restart_seeds: bool,
    classes: dict[str, tuple[type, type]],
) -> tuple[object, dict[str, object]]:
    selected_estimator: object | None = None
    selected_index = -1
    selected_score = np.inf
    selected_config: dict[str, object] | None = None
    candidate_records: list[dict[str, object]] = []
    for candidate_index, candidate_spec in enumerate(candidate_specs):
        fit_seed = seed + FIT_SEED_OFFSETS[method]
        if restart_seeds:
            fit_seed += 100 * candidate_index
        estimator, config_payload = _fit_estimator(
            method,
            x,
            treatment,
            y,
            x_dim=x_dim,
            outcome_lower=outcome_lower,
            outcome_upper=outcome_upper,
            seed=seed,
            quick=quick,
            benchmark=benchmark,
            fit_seed=fit_seed,
            overrides=candidate_spec,
            classes=classes,
        )
        score = float(scorer(estimator))
        candidate_records.append(
            {
                "candidate_index": candidate_index,
                "fit_seed": fit_seed,
                "config": config_payload,
                "validation_score": score,
            }
        )
        if score < selected_score:
            selected_estimator = estimator
            selected_index = candidate_index
            selected_score = score
            selected_config = config_payload
    if selected_estimator is None or selected_config is None:
        raise RuntimeError(f"{method} candidate selection produced no fitted estimator")
    selection = {
        "strategy": "validation_selection",
        "validation_metric": validation_metric,
        "direction": "min",
        "validation_seed": validation_seed,
        "validation_sample_size": validation_sample_size,
        "candidate_scores": candidate_records,
        "selected_index": selected_index,
        "selected_score": selected_score,
        "selected_config": selected_config,
    }
    return selected_estimator, selection


def _codite_candidate_specs(quick: bool) -> list[dict[str, object]]:
    ridge_grid = (1e-3, 1e-2) if quick else (1e-4, 1e-3, 1e-2)
    return [{"ridge": ridge} for ridge in ridge_grid]


def _dnet_candidate_specs(quick: bool) -> list[dict[str, object]]:
    training_steps = (80, 120) if quick else (750, 1_000, 1_250)
    return [{"training_steps": steps} for steps in training_steps]


def _make_sampler(estimator: object) -> Sampler:
    sample_potential = getattr(estimator, "sample_potential")

    def sampler(x: NDArray, treatment: int, n_per_x: int, seed: int | None) -> NDArray:
        values = sample_potential(x, treatment, n_per_x=n_per_x, seed=seed)
        array = np.asarray(values, dtype=np.float32)
        expected = (np.asarray(x).reshape(len(np.asarray(x)), -1).shape[0], n_per_x, 1)
        if array.shape != expected:
            raise ValueError(f"sample_potential returned {array.shape}; expected {expected}")
        return array

    return sampler


def _run_ihdp_repetition(task: tuple[int, int, bool]) -> RepetitionResult:
    seed, replication, quick = task
    base_seed = seed + 1_000 + 10_000 * replication
    dgp = IHDPDistDGP(seed=base_seed + 1, split_seed=base_seed + 2)
    train = dgp.observed_split("train", seed=base_seed + 3)
    validation_x, _ = dgp.split("validation")
    test_x, _ = dgp.split("test")
    true_mu = dgp.potential_means(test_x).astype(np.float64)
    true_tau = true_mu[:, 1] - true_mu[:, 0]
    outcome_lower, outcome_upper = dgp.outcome_bounds(train["y"])
    eval_mc = 96 if quick else 192
    mean_mc = 96 if quick else 192
    qte_mc = 192 if quick else 512
    true_qte_mc = 1_024 if quick else 4_096
    validation_mc = 64 if quick else 96
    validation_seed = base_seed + 40_000
    classes = _estimator_classes()

    def validation_scorer(estimator: object) -> float:
        return core.ihdp_extended_w1(
            dgp,
            _make_sampler(estimator),
            validation_x,
            validation_mc,
            validation_seed,
        )

    codite, codite_selection = _select_validation_candidate(
        "codite",
        _codite_candidate_specs(quick),
        validation_scorer,
        train["x"],
        train["t"],
        train["y"],
        x_dim=dgp.feature_dim,
        outcome_lower=outcome_lower,
        outcome_upper=outcome_upper,
        seed=base_seed,
        quick=quick,
        benchmark="ihdp",
        validation_metric="validation_extended_w1",
        validation_seed=validation_seed,
        validation_sample_size=validation_mc,
        restart_seeds=False,
        classes=classes,
    )
    dnet, dnet_selection = _select_validation_candidate(
        "dnet",
        _dnet_candidate_specs(quick),
        validation_scorer,
        train["x"],
        train["t"],
        train["y"],
        x_dim=dgp.feature_dim,
        outcome_lower=outcome_lower,
        outcome_upper=outcome_upper,
        seed=base_seed,
        quick=quick,
        benchmark="ihdp",
        validation_metric="validation_extended_w1",
        validation_seed=validation_seed,
        validation_sample_size=validation_mc,
        restart_seeds=False,
        classes=classes,
    )
    estimators: dict[str, object] = {"codite": codite, "dnet": dnet}
    selection: dict[str, dict[str, object]] = {
        "codite": codite_selection,
        "dnet": dnet_selection,
    }
    for method in ("causal_drf", "skcd"):
        estimator, config_payload = _fit_estimator(
            method,
            train["x"],
            train["t"],
            train["y"],
            x_dim=dgp.feature_dim,
            outcome_lower=outcome_lower,
            outcome_upper=outcome_upper,
            seed=base_seed,
            quick=quick,
            benchmark="ihdp",
            classes=classes,
        )
        estimators[method] = estimator
        selection[method] = {
            "strategy": "single_fit",
            "fit_seed": base_seed + FIT_SEED_OFFSETS[method],
            "selected_config": config_payload,
        }
    samplers = {method: _make_sampler(estimator) for method, estimator in estimators.items()}

    results: dict[str, dict[str, float]] = {}
    for method, sampler in samplers.items():
        if method == "dnet":
            predicted_mu = np.asarray(
                getattr(estimators[method], "predict_potential_outcomes")(test_x),
                dtype=np.float64,
            )
        else:
            predicted_mu = core.estimate_binary_means_from_sampler(
                sampler,
                test_x,
                mean_mc,
                base_seed + 10_000,
            )
        predicted_tau = predicted_mu[:, 1] - predicted_mu[:, 0]
        method_metrics = core.ihdp_distribution_metrics(
            dgp,
            sampler,
            test_x,
            eval_mc,
            base_seed + 20_000,
        )
        pehe = float(np.mean((predicted_tau - true_tau) ** 2))
        method_metrics.update(
            {
                "pehe": pehe,
                "sqrt_pehe": float(np.sqrt(pehe)),
                "ate_abs_error": float(abs(np.mean(predicted_tau) - np.mean(true_tau))),
            }
        )
        results[method] = method_metrics

    quantiles = np.linspace(0.05, 0.95, 19, dtype=np.float64)
    true_curve = core.ihdp_true_qte_curve(
        dgp,
        test_x,
        quantiles,
        true_qte_mc,
        base_seed + 30_000,
    )
    qte_errors: dict[str, list[float]] = {}
    for method, sampler in samplers.items():
        curve = core.ihdp_qte_curve(
            dgp,
            sampler,
            test_x,
            quantiles,
            qte_mc,
            base_seed + IHDP_QTE_SEED_OFFSETS[method],
        )
        error = np.abs(curve - true_curve).astype(np.float64)
        qte_errors[method] = error.tolist()
        results[method]["qte_abs_error"] = float(error.mean())

    payload: RepetitionResult = {
        "replication": replication,
        "methods": results,
        "qte": {
            "quantiles": quantiles.tolist(),
            "errors": qte_errors,
        },
    }
    return payload


def _run_jobs_repetition(task: tuple[int, int, bool]) -> RepetitionResult:
    seed, replication, quick = task
    base_seed = seed + 1_500 + 10_000 * replication
    data = JobsLaLondeData(
        split_seed=base_seed + 1,
        include_nsw_control_in_observed=False,
    )
    train = data.observed_split("train")
    validation_rct = data.rct_split("validation")
    test_rct = data.rct_split("test")
    outcome_lower, outcome_upper = data.outcome_bounds(train["y"])
    eval_samples = 48 if quick else 96
    mean_mc = 48 if quick else 96
    validation_samples = 32 if quick else 64
    validation_seed = base_seed + 50_000
    validation_att = data.rct_att_earnings("validation")
    classes = _estimator_classes()

    def validation_scorer(estimator: object) -> float:
        metrics, _ = core.jobs_evaluate_sampler(
            _make_sampler(estimator),
            validation_rct,
            validation_att,
            n_per_x=validation_samples,
            mean_mc=validation_samples,
            seed=validation_seed,
            include_additional=False,
        )
        return float(metrics["rct_w1"])

    codite, codite_selection = _select_validation_candidate(
        "codite",
        _codite_candidate_specs(quick),
        validation_scorer,
        train["x"],
        train["t"],
        train["y"],
        x_dim=data.feature_dim,
        outcome_lower=outcome_lower,
        outcome_upper=outcome_upper,
        seed=base_seed,
        quick=quick,
        benchmark="jobs",
        validation_metric="validation_rct_w1",
        validation_seed=validation_seed,
        validation_sample_size=validation_samples,
        restart_seeds=False,
        classes=classes,
    )
    dnet, dnet_selection = _select_validation_candidate(
        "dnet",
        _dnet_candidate_specs(quick),
        validation_scorer,
        train["x"],
        train["t"],
        train["y"],
        x_dim=data.feature_dim,
        outcome_lower=outcome_lower,
        outcome_upper=outcome_upper,
        seed=base_seed,
        quick=quick,
        benchmark="jobs",
        validation_metric="validation_rct_w1",
        validation_seed=validation_seed,
        validation_sample_size=validation_samples,
        restart_seeds=False,
        classes=classes,
    )
    estimators: dict[str, object] = {"codite": codite, "dnet": dnet}
    selection: dict[str, dict[str, object]] = {
        "codite": codite_selection,
        "dnet": dnet_selection,
    }
    for method in ("causal_drf", "skcd"):
        estimator, config_payload = _fit_estimator(
            method,
            train["x"],
            train["t"],
            train["y"],
            x_dim=data.feature_dim,
            outcome_lower=outcome_lower,
            outcome_upper=outcome_upper,
            seed=base_seed,
            quick=quick,
            benchmark="jobs",
            classes=classes,
        )
        estimators[method] = estimator
        selection[method] = {
            "strategy": "single_fit",
            "fit_seed": base_seed + FIT_SEED_OFFSETS[method],
            "selected_config": config_payload,
        }
    samplers = {method: _make_sampler(estimator) for method, estimator in estimators.items()}
    true_att = data.rct_att_earnings("test")
    rates = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    results: dict[str, dict[str, float]] = {}
    policy_curves: dict[str, list[float]] = {}
    arrays: dict[str, NDArray] = {}
    for method, sampler in samplers.items():
        earnings_mean_predictor: Callable[[NDArray], NDArray] | None = None
        if method == "dnet":
            dnet_estimator = estimators[method]

            def predict_dnet_earnings_mean(
                values: NDArray,
                fitted: object = dnet_estimator,
            ) -> NDArray:
                quantile_grid = np.asarray(
                    getattr(fitted, "predict_quantiles")(values),
                    dtype=np.float64,
                )
                return np.mean(
                    core.model_scale_to_earnings(quantile_grid),
                    axis=2,
                )

            earnings_mean_predictor = predict_dnet_earnings_mean

        method_metrics, method_detail = core.jobs_evaluate_sampler(
            sampler,
            test_rct,
            true_att,
            n_per_x=eval_samples,
            mean_mc=mean_mc,
            seed=base_seed + JOBS_EVALUATION_SEED_OFFSETS[method],
            earnings_mean_predictor=earnings_mean_predictor,
        )
        results[method] = method_metrics
        effects = np.asarray(method_detail["effects"], dtype=np.float64)
        policy_curves[method] = core.jobs_policy_curve(
            test_rct["y_earnings"],
            test_rct["t"],
            effects,
            rates,
        ).tolist()
        for arm in (0, 1):
            arrays[f"{method}_arm_{arm}_earnings"] = np.asarray(
                method_detail["arm_samples_earnings"][arm],
                dtype=np.float32,
            )

    payload: RepetitionResult = {
        "replication": replication,
        "methods": results,
    }
    payload["arm_samples"] = arrays
    return payload


def _run_tasks(
    worker: Callable[[tuple], RepetitionResult],
    tasks: list[tuple],
    parallel: int,
    benchmark: str,
) -> list[RepetitionResult]:
    results = []
    if parallel <= 1 or len(tasks) <= 1:
        for task in tasks:
            payload = worker(task)
            results.append(payload)
            print(f"{benchmark}: finished repetition {int(payload['replication']) + 1}/{len(tasks)}", flush=True)
        return results
    with ProcessPoolExecutor(max_workers=min(parallel, len(tasks))) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for future in as_completed(futures):
            payload = future.result()
            results.append(payload)
            print(f"{benchmark}: finished repetition {int(payload['replication']) + 1}/{len(tasks)}", flush=True)
    return sorted(results, key=lambda payload: int(payload["replication"]))


def _merge_summary(
    existing: dict[str, dict[str, float]],
    additions: list[RepetitionResult],
) -> dict[str, dict[str, float]]:
    summary = {method: dict(metrics) for method, metrics in existing.items()}
    summary.update(core.summarize_metric_repetitions([payload["methods"] for payload in additions]))
    return summary


def _paper_label(label: str) -> str:
    return {
        "CoDiTE (RKHS adapter)": "CoDiTE",
        "DNet (quantile-grid adapter)": "DNet",
        "SKCD (CME adapter)": "SKCD (adapted)",
        "Diff-PO": "DiffPO",
    }.get(label, label)


def _write_ihdp_qte_outputs(
    output_dir: Path,
    old_csv_text: str,
    additions: list[RepetitionResult],
) -> None:
    """Merge pointwise QTE means and SEs without inventing repetition records."""
    replaced_labels = {core.LABELS[method] for method in ADDITIONAL_METHODS}
    rows = list(csv.DictReader(old_csv_text.splitlines()))
    for row in rows:
        row["method"] = _paper_label(row["method"])
    rows = [row for row in rows if row["method"] not in replaced_labels]
    quantiles = np.asarray(additions[0]["qte"]["quantiles"], dtype=np.float64)
    for method in ADDITIONAL_METHODS:
        stack = np.asarray([payload["qte"]["errors"][method] for payload in additions], dtype=np.float64)
        means = stack.mean(axis=0)
        ses = stack.std(axis=0, ddof=1) / np.sqrt(len(stack)) if len(stack) > 1 else np.zeros_like(means)
        rows.extend(
            {
                "quantile": f"{quantile:.6f}",
                "method": core.LABELS[method],
                "absolute_qte_error": f"{mean:.8f}",
                "absolute_qte_error_se": f"{se:.8f}",
            }
            for quantile, mean, se in zip(quantiles, means, ses, strict=True)
        )
    with (output_dir / "ihdp_qte_error.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["quantile", "method", "absolute_qte_error", "absolute_qte_error_se"]
        )
        writer.writeheader()
        writer.writerows(rows)
    plt.figure(figsize=(10.6, 6.2))
    for method in core.IHDP_QTE_METHODS:
        points = [row for row in rows if row["method"] == core.LABELS[method]]
        if not points:
            continue
        grid = np.asarray([float(row["quantile"]) for row in points])
        means = np.asarray([float(row["absolute_qte_error"]) for row in points])
        ses = np.asarray([float(row["absolute_qte_error_se"]) for row in points])
        plt.plot(
            grid,
            means,
            label=core.LABELS[method],
            color=core.COLORS[method],
            linewidth=3.0 if method == "ganice" else 2.0,
            linestyle="-" if method == "ganice" else "--",
        )
        plt.fill_between(
            grid, np.maximum(0.0, means - ses), means + ses, color=core.COLORS[method], alpha=0.16, linewidth=0
        )
    plt.xlabel("quantile level")
    plt.ylabel("|estimated QTE - true QTE|")
    plt.ylim(bottom=0.0)
    core.place_legend_outside()
    plt.tight_layout()
    plt.savefig(output_dir / "ihdp_qte_error.pdf")
    plt.close()


def _finalize_ihdp(
    existing_root: Path,
    output_root: Path,
    additions: list[RepetitionResult],
) -> dict[str, dict[str, float]]:
    source_dir = existing_root / "IHDP"
    output_dir = output_root / "IHDP"
    summary = _merge_summary(_read_summary(source_dir, "IHDP"), additions)
    old_qte_text = (source_dir / "ihdp_qte_error.csv").read_text(encoding="utf-8")
    _write_ihdp_qte_outputs(output_dir, old_qte_text, additions)
    core.write_ihdp_ablation_output(output_dir, summary)
    core.write_ihdp_additional_outputs(output_dir, summary)
    core.write_paper_metric_summary(output_dir, summary)
    return summary


def _cdf_rows(text: str) -> list[dict[str, str]]:
    rows = list(csv.DictReader(text.splitlines()))
    for row in rows:
        row["method"] = _paper_label(row["method"])
    return rows


def _merge_and_plot_jobs_cdfs(
    output_dir: Path,
    old_csv_text: str,
    additions: list[RepetitionResult],
) -> None:
    replaced_labels = {core.LABELS[method] for method in ADDITIONAL_METHODS}
    old_rows = [row for row in _cdf_rows(old_csv_text) if row["method"] not in replaced_labels]
    if not old_rows:
        raise ValueError("Jobs: existing jobs_rct_cdf_mean.csv is empty")
    fieldnames = ["arm", "method", "earnings_grid", "cdf", "cdf_se"]
    new_lines: list[str] = []
    details_cache = [payload["arm_samples"] for payload in additions]

    for arm in (1, 0):
        true_label = f"NSW RCT {'treated' if arm == 1 else 'control'}"
        grid = np.asarray(
            [float(row["earnings_grid"]) for row in old_rows if int(row["arm"]) == arm and row["method"] == true_label],
            dtype=np.float64,
        )
        if grid.size == 0:
            raise ValueError(f"Jobs: missing existing CDF grid for arm {arm}")
        for method in ADDITIONAL_METHODS:
            stack = np.stack(
                [core._cdf_on_grid(details[f"{method}_arm_{arm}_earnings"], grid) for details in details_cache],
                axis=0,
            )
            mean = stack.mean(axis=0)
            se = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0]) if stack.shape[0] > 1 else np.zeros_like(mean)
            for x_value, mean_value, se_value in zip(grid, mean, se, strict=True):
                new_lines.append(f"{arm},{METHOD_LABELS[method]},{x_value:.8f},{mean_value:.8f},{se_value:.8f}")

    old_lines = [",".join(fieldnames)] + [",".join(row[key] for key in fieldnames) for row in old_rows]
    combined_text = "\n".join(old_lines + new_lines) + "\n"
    (output_dir / "jobs_rct_cdf_mean.csv").write_text(combined_text, encoding="utf-8")
    combined_rows = _cdf_rows(combined_text)
    labels_by_method = {
        method: core.LABELS.get(method, METHOD_LABELS.get(method, method)) for method in ALL_BINARY_METHODS
    }

    for arm, arm_name in ((1, "treated"), (0, "control")):
        true_label = f"NSW RCT {arm_name}"
        plt.figure(figsize=(10.8, 6.3))
        ordered_labels = [true_label] + [labels_by_method[method] for method in ALL_BINARY_METHODS]
        for label in ordered_labels:
            rows = [row for row in combined_rows if int(row["arm"]) == arm and row["method"] == label]
            if not rows:
                continue
            grid = np.asarray([float(row["earnings_grid"]) for row in rows], dtype=np.float64)
            mean = np.asarray([float(row["cdf"]) for row in rows], dtype=np.float64)
            se = np.asarray([float(row["cdf_se"]) for row in rows], dtype=np.float64)
            if label == true_label:
                color = core.COLORS["true"]
                linewidth = 3.0
                linestyle = "-"
            else:
                method = next(name for name, method_label in labels_by_method.items() if method_label == label)
                color = core.COLORS.get(method, "#999999")
                linewidth = 3.0 if method == "ganice" else 1.9
                linestyle = "-" if method == "ganice" else "--"
            plt.plot(
                grid / 1000.0,
                mean,
                label=label,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
            )
            plt.fill_between(
                grid / 1000.0,
                np.clip(mean - se, 0.0, 1.0),
                np.clip(mean + se, 0.0, 1.0),
                color=color,
                alpha=0.16 if label == true_label else 0.06,
                linewidth=0,
            )
        arm_grid = np.asarray(
            [
                float(row["earnings_grid"])
                for row in combined_rows
                if int(row["arm"]) == arm and row["method"] == true_label
            ],
            dtype=np.float64,
        )
        plt.xlim(left=0.0, right=max(5.0, float(arm_grid.max()) / 1000.0))
        plt.ylim(0.0, 1.01)
        plt.xlabel("RE78 earnings (thousand USD)")
        plt.ylabel("empirical CDF")
        core.place_legend_outside()
        plt.tight_layout()
        plt.savefig(output_dir / f"jobs_rct_cdf_{arm_name}.pdf")
        if arm == 1:
            plt.savefig(output_dir / "jobs_rct_cdf.pdf")
        plt.close()


def _finalize_jobs(
    existing_root: Path,
    output_root: Path,
    additions: list[RepetitionResult],
    old_cdf_text: str,
) -> dict[str, dict[str, float]]:
    output_dir = output_root / "Jobs"
    summary = _merge_summary(_read_summary(existing_root / "Jobs", "Jobs"), additions)
    core.write_jobs_additional_outputs(output_dir, summary)
    _merge_and_plot_jobs_cdfs(output_dir, old_cdf_text, additions)
    core.write_paper_metric_summary(output_dir, summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run only CoDiTE, DNet, Causal-DRF, and SKCD on their supported binary benchmarks, "
            "then merge them with existing GANICE experiment outputs."
        )
    )
    parser.add_argument("--benchmark", choices=("all", "ihdp", "jobs", "tcga"), default="all")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--existing-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.parallel < 1:
        parser.error("--parallel must be positive")
    return args


def main() -> None:
    args = _parse_args()
    existing_root = args.existing_root.resolve()
    output_root = args.output_root.resolve()
    if output_root != existing_root and output_root.is_relative_to(existing_root):
        raise ValueError("--output-root cannot be nested within --existing-root")
    if not existing_root.is_dir():
        raise FileNotFoundError(f"--existing-root does not exist: {existing_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    selected = {
        "IHDP": args.benchmark in ("all", "ihdp"),
        "Jobs": args.benchmark in ("all", "jobs"),
        "TCGA": args.benchmark in ("all", "tcga"),
    }
    for benchmark, enabled in selected.items():
        if enabled:
            _copy_existing_benchmark(existing_root, output_root, benchmark)
    tasks = [(args.seed, replication, args.quick) for replication in range(args.repetitions)]
    if selected["IHDP"]:
        additions = _run_tasks(_run_ihdp_repetition, tasks, args.parallel, "IHDP")
        _finalize_ihdp(existing_root, output_root, additions)
    if selected["Jobs"]:
        old_cdf_text = (existing_root / "Jobs" / "jobs_rct_cdf_mean.csv").read_text(encoding="utf-8")
        additions = _run_tasks(_run_jobs_repetition, tasks, args.parallel, "Jobs")
        _finalize_jobs(existing_root, output_root, additions, old_cdf_text)
    if selected["TCGA"]:
        core.write_paper_metric_summary(output_root / "TCGA", _read_summary(existing_root / "TCGA", "TCGA"))
    if all((output_root / name / "metrics.csv").is_file() for name in ("IHDP", "Jobs", "TCGA")):
        from scripts.run_paper_experiments import write_main_results

        write_main_results(output_root)
    print(f"saved paper benchmark outputs to {output_root}")


if __name__ == "__main__":
    main()
