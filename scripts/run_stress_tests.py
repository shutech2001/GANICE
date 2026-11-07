"""Reproduce the finite-state GANICE sensitivity table in the paper.

Only the six reported one-factor comparisons are fitted. Hyperparameters,
random streams, statewise evaluation and target masses match the original
finite-state experiment; final artifacts are CSV numbers and a LaTeX table.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import os
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Callable, Iterable, Optional, TypedDict

import numpy as np
from numpy.typing import NDArray
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ganice-matplotlib-cache")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from estimators.mechanism_stress import MechanismStressConfig, MechanismStressDGP, TargetShiftSet  # noqa: E402
from estimators.metrics import sample_wasserstein_1_1d  # noqa: E402
from estimators.utils import ensure_dir  # noqa: E402
from scripts import run_ganice_experiment as core  # noqa: E402

NOMINAL_N = 2_048
NOMINAL_D = 4
NOMINAL_KAPPA = 1.0
NOMINAL_DELTA = 0.0
NOMINAL_H = 0.5
NOMINAL_RESOLUTION = 4
DEFAULT_DATA_SEED = 42
DEFAULT_TRAINING_REPETITIONS = 3
DEFAULT_EVALUATION_DRAWS = 1_024


class SensitivityGrids(TypedDict):
    weak_overlap: list[float]
    sparse_cells: list[int]
    target_design_shift: list[float]
    cell_resolution: list[int]
    conditioning_dimension: list[int]
    treatment_covariate_heterogeneity: list[float]


PREDECLARED_GRIDS: SensitivityGrids = {
    "weak_overlap": [1.0, 0.5, 0.25],
    "sparse_cells": [2_048, 1_024, 768, 512],
    "target_design_shift": [0.0, 0.45, 0.9],
    "cell_resolution": [0, 1, 2, 3, 4],
    "conditioning_dimension": [2, 3, 4, 5, 6],
    "treatment_covariate_heterogeneity": [0.0, 0.25, 0.5, 0.75, 1.0],
}
FACTOR_LABELS = {
    "weak_overlap": "Overlap",
    "sparse_cells": "Sample size",
    "target_design_shift": "Target-design shift",
    "cell_resolution": "Cell resolution",
    "conditioning_dimension": "Covariate dimension",
    "treatment_covariate_heterogeneity": "Heterogeneity",
}
FACTOR_PARAMETERS = {
    "weak_overlap": r"\kappa",
    "sparse_cells": "n",
    "target_design_shift": r"\delta",
    "cell_resolution": r"|\Pi|",
    "conditioning_dimension": "d",
    "treatment_covariate_heterogeneity": "h",
}
Sampler = Callable[[NDArray, int, int, Optional[int]], NDArray]


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    scenario_id: str
    n_train: int = NOMINAL_N
    conditioning_dim: int = NOMINAL_D
    overlap_kappa: float = NOMINAL_KAPPA
    target_shift: float = NOMINAL_DELTA
    heterogeneity: float = NOMINAL_H
    resolution_bits: int = NOMINAL_RESOLUTION
    target_shift_set: TargetShiftSet = "H"


@dataclass(frozen=True, slots=True)
class CurvePoint:
    axis: str
    order: int
    level: float
    level_label: str
    severity: float
    scenario_id: str


def _nominal_spec() -> ScenarioSpec:
    return ScenarioSpec(scenario_id="nominal")


def _float_id(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def build_suite() -> tuple[list[ScenarioSpec], dict[str, list[CurvePoint]]]:
    """Return the six paper comparisons, sharing each nominal fit."""

    nominal = _nominal_spec()
    specs: dict[str, ScenarioSpec] = {nominal.scenario_id: nominal}
    curves: dict[str, list[CurvePoint]] = {}

    def add_curve(
        axis: str,
        entries: Iterable[tuple[float, str, float, ScenarioSpec]],
    ) -> None:
        points: list[CurvePoint] = []
        for order, (level, label, severity, spec) in enumerate(entries):
            specs.setdefault(spec.scenario_id, spec)
            points.append(
                CurvePoint(
                    axis=axis,
                    order=order,
                    level=float(level),
                    level_label=label,
                    severity=float(severity),
                    scenario_id=spec.scenario_id,
                )
            )
        curves[axis] = points

    add_curve(
        "weak_overlap",
        (
            (
                kappa,
                f"{kappa:g}",
                1.0 - kappa,
                nominal
                if kappa == NOMINAL_KAPPA
                else replace(
                    nominal,
                    scenario_id=f"weak_overlap_kappa_{_float_id(kappa)}",
                    overlap_kappa=kappa,
                ),
            )
            for kappa in PREDECLARED_GRIDS["weak_overlap"]
        ),
    )
    add_curve(
        "sparse_cells",
        (
            (
                float(n_train),
                str(n_train),
                1.0 - n_train / NOMINAL_N,
                nominal
                if n_train == NOMINAL_N
                else replace(
                    nominal,
                    scenario_id=f"sparse_cells_n_{n_train}",
                    n_train=int(n_train),
                ),
            )
            for n_train in PREDECLARED_GRIDS["sparse_cells"]
        ),
    )
    add_curve(
        "target_design_shift",
        (
            (
                delta,
                f"{delta:g}",
                delta,
                nominal
                if delta == NOMINAL_DELTA
                else replace(
                    nominal,
                    scenario_id=f"target_shift_delta_{_float_id(delta)}",
                    target_shift=delta,
                ),
            )
            for delta in PREDECLARED_GRIDS["target_design_shift"]
        ),
    )
    add_curve(
        "cell_resolution",
        (
            (
                float(resolution),
                f"{resolution} ({2 ** (resolution + 1)} cells)",
                float(abs(NOMINAL_RESOLUTION - resolution)),
                nominal
                if resolution == NOMINAL_RESOLUTION
                else replace(
                    nominal,
                    scenario_id=f"cell_resolution_r_{resolution}",
                    resolution_bits=int(resolution),
                ),
            )
            for resolution in PREDECLARED_GRIDS["cell_resolution"]
        ),
    )
    add_curve(
        "conditioning_dimension",
        (
            (
                float(dimension),
                str(dimension),
                float(dimension),
                nominal
                if dimension == NOMINAL_D
                else replace(
                    nominal,
                    scenario_id=f"conditioning_dimension_d_{dimension}",
                    conditioning_dim=int(dimension),
                    resolution_bits=int(dimension),
                ),
            )
            for dimension in PREDECLARED_GRIDS["conditioning_dimension"]
        ),
    )
    add_curve(
        "treatment_covariate_heterogeneity",
        (
            (
                heterogeneity,
                f"{heterogeneity:g}",
                heterogeneity,
                nominal
                if heterogeneity == NOMINAL_H
                else replace(
                    nominal,
                    scenario_id=f"heterogeneity_h_{_float_id(heterogeneity)}",
                    heterogeneity=heterogeneity,
                ),
            )
            for heterogeneity in PREDECLARED_GRIDS["treatment_covariate_heterogeneity"]
        ),
    )

    return list(specs.values()), curves


def _make_dgp(spec: ScenarioSpec, data_seed: int) -> MechanismStressDGP:
    return MechanismStressDGP(
        MechanismStressConfig(
            conditioning_dim=spec.conditioning_dim,
            max_conditioning_dim=max(PREDECLARED_GRIDS["conditioning_dimension"]),
            master_observed_size=NOMINAL_N,
            overlap_kappa=spec.overlap_kappa,
            target_shift=spec.target_shift,
            heterogeneity=spec.heterogeneity,
            partition_bits=spec.resolution_bits,
            target_shift_set=spec.target_shift_set,
            seed=data_seed,
        )
    )


def _partition_transform(resolution_bits: int) -> Callable[[NDArray], NDArray[np.float32]]:
    """Map W to C_r=(Z_1,...,Z_r,T), with treatment always retained."""

    r = int(resolution_bits)

    def transform(w: NDArray) -> NDArray[np.float32]:
        w_array = np.asarray(w, dtype=np.float32)
        if w_array.ndim == 1:
            w_array = w_array[None, :]
        if r == 0:
            return w_array[:, -1:].astype(np.float32)
        return np.column_stack([w_array[:, :r], w_array[:, -1]]).astype(np.float32)

    return transform


def _ganice_sampler(model: object, dgp: MechanismStressDGP) -> Sampler:
    sample_conditional = getattr(model, "sample_conditional")

    def sampler(
        x: NDArray,
        treatment: int,
        n_per_x: int,
        seed: Optional[int],
    ) -> NDArray:
        x_array = np.asarray(x, dtype=np.float32)
        if x_array.ndim == 1:
            x_array = x_array[None, :]
        rows = []
        for row_index in range(x_array.shape[0]):
            w = dgp.encode_w(x_array[row_index:row_index + 1], treatment)
            rows.append(
                sample_conditional(
                    w,
                    n_per_x,
                    seed=None if seed is None else seed + row_index,
                ).reshape(1, n_per_x, 1)
            )
        return np.concatenate(rows, axis=0).astype(np.float32)

    return sampler


def _fit_ganice(
    dgp: MechanismStressDGP, train: dict[str, NDArray], *, fit_seed: int, quick: bool
) -> tuple[object, Sampler]:
    lower, upper, x_dim = dgp.outcome_lower, dgp.outcome_upper, dgp.feature_dim
    r = dgp.partition_bits
    cell_resolution = (1,) * (r + 1)
    config = core.ihdp_ganice_config(
        fit_seed,
        x_dim,
        lower,
        upper,
        quick=quick,
        cell_resolution=cell_resolution,
        min_cell_samples=6,
        shared_generator=True,
        factual_crps_weight=8.0,
        factual_mse_weight=2.0,
        generator_transport_weight=0.75,
    )
    config = replace(
        config,
        residual_quantile_calibration=False,
        target_mass_samples=12_000 if quick else 50_000,
    )
    model = core.fit_ganice(
        d_w=dgp.d_w,
        train_w=train["w"],
        train_y=train["y"],
        target_w_sampler=dgp.sample_target_w,
        config=config,
        seed=fit_seed + 1,
        d_cell_w=r + 1,
        cell_transform=_partition_transform(r),
    )
    return model, _ganice_sampler(model, dgp)


def _truth_samples(
    dgp: MechanismStressDGP,
    state_table: dict[str, NDArray],
    *,
    draws: int,
    seed: int,
) -> NDArray[np.float32]:
    values = np.empty((len(state_table["state_id"]), draws), dtype=np.float32)
    for state_index, state_id in enumerate(state_table["state_id"]):
        values[state_index] = dgp.sample_potential(
            state_table["z"][state_index:state_index + 1],
            int(state_table["t"][state_index]),
            n_per_x=draws,
            seed=seed + 1_009 * int(state_id),
        )[0, :, 0]
    return values


def _run_scenario_task(task: tuple[ScenarioSpec, int, int, bool, int]) -> tuple[str, int, float]:
    spec, data_seed, repetition, quick, draws = task
    torch.set_num_threads(1)
    dgp = _make_dgp(spec, data_seed)
    train = dgp.sample_observed(spec.n_train, seed=data_seed + 1_000)
    states = dgp.enumerate_target_states()
    truth = _truth_samples(dgp, states, draws=draws, seed=data_seed + 100_000)
    fit_seed = data_seed + 1_000_000 * (repetition + 1) + 101
    _, sampler = _fit_ganice(dgp, train, fit_seed=fit_seed, quick=quick)
    state_errors = np.empty(len(states["state_id"]), dtype=np.float64)
    for index, state_id in enumerate(states["state_id"]):
        predicted = (
            sampler(
                states["x"][index:index + 1],
                int(states["t"][index]),
                draws,
                data_seed + 200_000 + 1_009 * int(state_id),
            )[0]
            .reshape(-1)
            .astype(np.float64)
        )
        state_errors[index] = sample_wasserstein_1_1d(
            truth[index].astype(np.float64),
            predicted,
        )
    return spec.scenario_id, repetition, float(np.sum((states["q"] / states["q"].sum()) * state_errors))


def _mean_se(values: NDArray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0


def write_outputs(output_root: Path, results: dict[str, list[float]]) -> None:
    """Write the paper table and its full-precision numerical source."""
    output_root = ensure_dir(output_root)
    _, curves = build_suite()
    nominal = np.asarray(results["nominal"], dtype=np.float64)
    if nominal.size == 0 or np.any(~np.isfinite(nominal)) or np.any(nominal <= 0):
        raise ValueError("nominal errors must be finite, positive and nonempty")
    table_rows = []
    number_rows = []
    for axis, points in curves.items():
        settings, ratios = [], []
        for point in points:
            errors = np.asarray(results[point.scenario_id], dtype=np.float64)
            if errors.shape != nominal.shape or np.any(~np.isfinite(errors)):
                raise ValueError(f"incomplete paired seeds for {point.scenario_id}")
            mean, se = _mean_se(errors)
            # The table uses ratios of displayed error means. The paired-seed
            # delta method preserves the common nominal denominator covariance.
            relative = mean / float(nominal.mean())
            _, relative_se = _mean_se((errors - relative * nominal) / nominal.mean())
            setting = str(2 ** (int(point.level) + 1)) if axis == "cell_resolution" else f"{point.level:g}"
            settings.append(setting)
            ratios.append(f"{relative:.3f}")
            number_rows.append(
                {
                    "factor": FACTOR_LABELS[axis],
                    "setting": setting,
                    "scenario_id": point.scenario_id,
                    "repetitions": errors.size,
                    "extended_w1": mean,
                    "extended_w1_se": se,
                    "relative_error": relative,
                    "relative_error_se": relative_se,
                }
            )
        table_rows.append(
            {
                "factor": FACTOR_LABELS[axis],
                "parameter": FACTOR_PARAMETERS[axis],
                "settings": ",".join(settings),
                "relative_error": ",".join(ratios),
            }
        )
    with (output_root / "sensitivity_numbers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(number_rows[0]))
        writer.writeheader()
        writer.writerows(number_rows)
    with (output_root / "sensitivity_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    tex = [
        r"\begin{table}[tb]",
        r"\centering",
        r"\small",
        (
            r"\caption{Finite-state sensitivity analysis of GANICE. "
            r"Relative errors use the nominal configuration as the reference.}"
        ),
        r"\label{tab:stress-tests}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Factor & Settings & Relative error \\",
        r"\midrule",
    ]
    for row in table_rows:
        tex.append(f"{row['factor']} & ${row['parameter']}={row['settings']}$ & ${row['relative_error']}$ " + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output_root / "sensitivity_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/sensitivity"))
    parser.add_argument("--seed", type=int, default=DEFAULT_DATA_SEED)
    parser.add_argument("--training-repetitions", type=int, default=DEFAULT_TRAINING_REPETITIONS)
    parser.add_argument("--evaluation-draws", type=int, default=DEFAULT_EVALUATION_DRAWS)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.training_repetitions < 1 or args.evaluation_draws < 16 or args.parallel < 1:
        raise ValueError("repetitions and parallel must be positive; evaluation draws must be at least 16")
    specs, _ = build_suite()
    tasks = [
        (spec, args.seed, repetition, args.quick, args.evaluation_draws)
        for spec in specs
        for repetition in range(args.training_repetitions)
    ]
    results = {spec.scenario_id: [float("nan")] * args.training_repetitions for spec in specs}

    def record(result):
        scenario_id, repetition, error = result
        results[scenario_id][repetition] = error
        print(f"finished {scenario_id}, seed {repetition + 1}/{args.training_repetitions}", flush=True)

    if args.parallel == 1:
        for task in tasks:
            record(_run_scenario_task(task))
    else:
        with ProcessPoolExecutor(max_workers=min(args.parallel, len(tasks))) as executor:
            futures = [executor.submit(_run_scenario_task, task) for task in tasks]
            for future in as_completed(futures):
                record(future.result())
    write_outputs(args.output_root, results)
    print(f"saved sensitivity table to {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
