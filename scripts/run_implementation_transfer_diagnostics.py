"""Run the Appendix F.4 implementation-transfer diagnostics.

This runner deliberately leaves ``estimators/ganice.py`` unchanged.  It
refits only the shared GANICE models whose checkpoints were not retained, then
fits one independent hard-routed neural WGAN per target-relevant original cell.
Each routed expert receives only a constant dummy input, so its generated law
is of the one-state form G_C(U).

The primary plug-in diagnostic is computed on the original dyadic partition:

    epsilon_hat = sum_C q_hat_C W1_hat(nu_shared,C, nu_route,C).

For scalar outcomes and equal Monte Carlo sample sizes, W1_hat is the exact
order-statistic formula.  The reference uses the repository's WGAN-GP network
classes and fixed-step optimizer; it is therefore an executable neural proxy
for, not a certificate of, the abstract sieve ERM in Appendix F.2.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
import sys
from typing import Optional, TypedDict

import numpy as np
from numpy.typing import NDArray
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ganice-matplotlib-cache")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from estimators import GANICE, GANICEConfig, IHDPDistDGP, JobsLaLondeData, TCGADoseDGP  # noqa: E402
from estimators.ganice import _cell_midpoints  # noqa: E402
from estimators.neural import sample_latent  # noqa: E402
from estimators.utils import ensure_dir, ensure_row_matrix  # noqa: E402
from scripts import run_ganice_experiment as core  # noqa: E402


BENCHMARKS = ("IHDP", "Jobs", "TCGA")
DEFAULT_REPETITIONS = {"IHDP": 10, "Jobs": 10, "TCGA": 5}
DEFAULT_DIAGNOSTIC_DRAWS = 2_048
DEFAULT_DIAGNOSTIC_REPEATS = 5
DEFAULT_TARGET_PROBE_SIZE = 200_000


class _IHDPConfigWeights(TypedDict):
    generator_transport_weight: float
    factual_crps_weight: float
    factual_mse_weight: float


class _JobsConfigOverrides(TypedDict):
    num_steps: int
    generator_transport_weight: float
    factual_crps_weight: float


class _ImplementationTableRow(TypedDict):
    benchmark: str
    repetitions: int
    shared_error: float
    shared_error_se: float
    routed_error: float
    routed_error_se: float
    epsilon: float
    epsilon_se: float
    ratio: float


def exact_equal_mass_w1(sample_a: NDArray, sample_b: NDArray) -> float:
    """Exact one-dimensional empirical W1 for equally sized empirical laws."""

    a = np.asarray(sample_a, dtype=np.float64).reshape(-1)
    b = np.asarray(sample_b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        raise ValueError("W1 samples must be nonempty")
    if a.size != b.size:
        raise ValueError("exact_equal_mass_w1 requires equal sample sizes")
    return float(np.mean(np.abs(np.sort(a) - np.sort(b))))


def _mean_se(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    se = float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0
    return float(array.mean()), se


def _draw_raw_generator_once(model: GANICE, queries: NDArray, seed: int) -> NDArray[np.float64]:
    """Draw one raw neural-generator sample per query, without post-hoc calibration."""

    if model.generator is None:
        raise ValueError("GANICE model has not been fitted")
    query = ensure_row_matrix(np.asarray(queries, dtype=np.float32))
    cells = model._route_query_cells(query)
    outputs: list[NDArray[np.float32]] = []
    was_training = model.generator.training
    model.generator.eval()
    with torch.no_grad():
        for start in range(0, query.shape[0], int(model.config.max_predict_batch)):
            stop = min(query.shape[0], start + int(model.config.max_predict_batch))
            w_t = torch.as_tensor(query[start:stop], dtype=torch.float32, device=model.device)
            cells_t = torch.as_tensor(cells[start:stop], dtype=torch.long, device=model.device)
            output = model.generator(
                w_t,
                cells_t,
                sample_latent(
                    stop - start,
                    model.config.latent_dim,
                    model.device,
                    seed=seed + start,
                ),
            )
            outputs.append(output.cpu().numpy().astype(np.float32))
    model.generator.train(was_training)
    return np.concatenate(outputs, axis=0).reshape(-1).astype(np.float64)


@dataclass(slots=True)
class HardRoutedReference:
    """Hard wrapper around fitted one-state cell experts."""

    partition_model: GANICE
    experts: dict[int, GANICE]
    estimable_cells: NDArray[np.int64]

    def original_cells(self, queries: NDArray) -> NDArray[np.int64]:
        return self.partition_model._cell_ids(ensure_row_matrix(np.asarray(queries, dtype=np.float32)))

    def routed_cell(self, original_cell: int) -> int:
        cell = int(original_cell)
        if cell in self.experts:
            return cell
        alias = self.partition_model.cell_alias
        if alias is not None and int(alias[cell]) in self.experts:
            return int(alias[cell])
        expert_cells = np.asarray(sorted(self.experts), dtype=np.int64)
        if expert_cells.size == 0:
            raise ValueError("routed reference has no fitted expert")
        source_midpoint = _cell_midpoints(np.asarray([cell]), self.partition_model.levels)
        expert_midpoints = _cell_midpoints(expert_cells, self.partition_model.levels)
        nearest = int(np.argmin(np.sum((expert_midpoints - source_midpoint) ** 2, axis=1)))
        return int(expert_cells[nearest])

    def midpoint_queries(self, queries: NDArray) -> NDArray[np.float32]:
        cells = self.original_cells(queries)
        routed = np.asarray([self.routed_cell(int(cell)) for cell in cells], dtype=np.int64)
        return _cell_midpoints(routed, self.partition_model.levels).astype(np.float32)

    def sample_conditional(self, query: NDArray, n: int, seed: Optional[int] = None) -> NDArray[np.float32]:
        query_arr = ensure_row_matrix(np.asarray(query, dtype=np.float32))
        if query_arr.shape[0] != 1:
            raise ValueError("sample_conditional expects one conditioning point")
        cell = self.routed_cell(int(self.original_cells(query_arr)[0]))
        constant = np.full((1, 1), 0.5, dtype=np.float32)
        return self.experts[cell].sample_conditional(constant, n, seed=seed)

    def draw_raw_once(self, queries: NDArray, seed: int) -> NDArray[np.float64]:
        query_arr = ensure_row_matrix(np.asarray(queries, dtype=np.float32))
        cells = self.original_cells(query_arr)
        outputs = np.empty(query_arr.shape[0], dtype=np.float64)
        for cell in np.unique(cells).tolist():
            routed_cell = self.routed_cell(int(cell))
            positions = np.flatnonzero(cells == int(cell))
            constant = np.full((positions.size, 1), 0.5, dtype=np.float32)
            outputs[positions] = _draw_raw_generator_once(
                self.experts[routed_cell],
                constant,
                seed + 10_000 * routed_cell,
            )
        return outputs


def _reference_config(base: GANICEConfig, seed: int) -> GANICEConfig:
    """Make a stable one-state routed GANICE/WGAN proxy without changing GANICE."""

    return replace(
        base,
        resolution=(0,),
        min_cell_samples=1,
        cell_normalized=True,
        critic_uses_w=False,
        residual_quantile_calibration=False,
        shared_generator=True,
        seed=int(seed),
    )


def fit_hard_routed_reference(
    partition_model: GANICE,
    train_w: NDArray,
    train_y: NDArray,
    target_q: NDArray[np.float64],
    *,
    base_config: GANICEConfig,
    seed: int,
) -> tuple[HardRoutedReference, dict[str, float | int | list[int]]]:
    """Fit disjoint one-state experts on the original, unaliased cells."""

    w = ensure_row_matrix(np.asarray(train_w, dtype=np.float32))
    y = np.asarray(train_y, dtype=np.float32)
    original_cells = partition_model._cell_ids(w)
    observed_counts = np.bincount(original_cells, minlength=partition_model.num_cells).astype(np.int64)
    target_support = np.flatnonzero(np.asarray(target_q, dtype=np.float64) > 0.0).astype(np.int64)
    estimable_cells = target_support[observed_counts[target_support] > 0]
    missing_cells = target_support[observed_counts[target_support] == 0]
    if estimable_cells.size == 0:
        raise ValueError("no target-relevant original cell contains an observed outcome")

    experts: dict[int, GANICE] = {}

    def constant_target(n: int, seed: Optional[int] = None) -> NDArray[np.float32]:
        del seed
        return np.full((int(n), 1), 0.5, dtype=np.float32)

    for cell in estimable_cells.tolist():
        cell_mask = original_cells == int(cell)
        cell_y = y[cell_mask]
        cell_w = np.full((cell_y.shape[0], 1), 0.5, dtype=np.float32)
        cell_seed = int(seed + 10_000 * int(cell))
        config = _reference_config(base_config, cell_seed)
        expert = core.fit_ganice(
            d_w=1,
            train_w=cell_w,
            train_y=cell_y,
            target_w_sampler=constant_target,
            config=config,
            seed=cell_seed + 1,
        )
        if expert.active_cells is None or expert.active_cells.tolist() != [0]:
            raise RuntimeError(f"one-state expert for original cell {cell} did not retain its sole cell")
        experts[int(cell)] = expert

    reference = HardRoutedReference(
        partition_model=partition_model,
        experts=experts,
        estimable_cells=estimable_cells,
    )
    representative_config = _reference_config(base_config, seed)
    diagnostics: dict[str, float | int | list[int]] = {
        "num_original_cells": int(partition_model.num_cells),
        "num_target_support_cells": int(target_support.size),
        "num_estimable_cells": int(estimable_cells.size),
        "num_missing_cells": int(missing_cells.size),
        "num_independent_expert_fits": int(len(experts)),
        "estimable_cells": [int(cell) for cell in estimable_cells.tolist()],
        "missing_cells": [int(cell) for cell in missing_cells.tolist()],
        "minimum_estimable_cell_count": int(observed_counts[estimable_cells].min()),
        "maximum_estimable_cell_count": int(observed_counts[estimable_cells].max()),
        "num_steps_per_cell": int(representative_config.num_steps),
        "pretrain_steps_per_cell": int(representative_config.pretrain_steps),
        "critic_steps_per_cell": int(representative_config.critic_steps),
    }
    return reference, diagnostics


def estimate_transfer(
    shared: GANICE,
    routed: HardRoutedReference,
    target_probe_w: NDArray,
    *,
    draws_per_repeat: int,
    repeats: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, dict[str, object]]]:
    """Estimate original-cell transfer and its Monte Carlo self-distance."""

    probe = ensure_row_matrix(np.asarray(target_probe_w, dtype=np.float32))
    probe_cells = shared._cell_ids(probe)
    counts = np.bincount(probe_cells, minlength=shared.num_cells).astype(np.float64)
    q = counts / max(float(counts.sum()), 1.0)
    estimable = set(int(cell) for cell in routed.estimable_cells.tolist())
    target_support = np.flatnonzero(q > 0.0).astype(np.int64)
    missing_mass = float(sum(q[int(cell)] for cell in target_support if int(cell) not in estimable))

    alias = shared.cell_alias
    alias_mismatch_mass = 0.0
    if alias is not None:
        for cell in target_support.tolist():
            if int(alias[int(cell)]) != int(cell):
                alias_mismatch_mass += float(q[int(cell)])

    epsilon_repeats: list[float] = []
    route_floor_repeats: list[float] = []
    per_cell_values: dict[int, list[float]] = {int(cell): [] for cell in routed.estimable_cells.tolist()}
    per_cell_floor: dict[int, list[float]] = {int(cell): [] for cell in routed.estimable_cells.tolist()}
    for repeat in range(int(repeats)):
        rng = np.random.default_rng(seed + 100_000 * repeat)
        epsilon = 0.0
        route_floor = 0.0
        for cell in routed.estimable_cells.tolist():
            pool = np.flatnonzero(probe_cells == int(cell))
            if pool.size == 0:
                continue
            selected = rng.choice(pool, size=int(draws_per_repeat), replace=pool.size < int(draws_per_repeat))
            cell_w = probe[selected]
            shared_draws = _draw_raw_generator_once(
                shared,
                cell_w,
                seed + 100_000 * repeat + 1_000 * int(cell) + 11,
            )
            route_draws = routed.draw_raw_once(
                cell_w,
                seed + 100_000 * repeat + 1_000 * int(cell) + 23,
            )
            route_draws_2 = routed.draw_raw_once(
                cell_w,
                seed + 100_000 * repeat + 1_000 * int(cell) + 37,
            )
            cell_w1 = exact_equal_mass_w1(shared_draws, route_draws)
            cell_floor = exact_equal_mass_w1(route_draws, route_draws_2)
            epsilon += float(q[int(cell)]) * cell_w1
            route_floor += float(q[int(cell)]) * cell_floor
            per_cell_values[int(cell)].append(cell_w1)
            per_cell_floor[int(cell)].append(cell_floor)
        epsilon_repeats.append(float(epsilon))
        route_floor_repeats.append(float(route_floor))

    epsilon_mean, epsilon_mc_se = _mean_se(epsilon_repeats)
    floor_mean, floor_mc_se = _mean_se(route_floor_repeats)
    summary = {
        "epsilon_impl_raw_original_partition": epsilon_mean,
        "epsilon_impl_raw_mc_se": epsilon_mc_se,
        "route_self_mc_distance": floor_mean,
        "route_self_mc_se": floor_mc_se,
        "hard_reference_common_draw_epsilon_estimable_cells": 0.0,
        "hard_reference_gate_eta_estimable_cells": 0.0,
        "hard_reference_gate_transfer_bound_estimable_cells": 0.0,
        "shared_gate_eta_available": 0.0,
        "missing_target_mass": missing_mass,
        "estimable_target_mass": float(1.0 - missing_mass),
        "alias_mismatch_mass_original_partition": float(alias_mismatch_mass),
        "alias_indicator_l1_original_partition": float(2.0 * alias_mismatch_mass),
        "target_q_sum": float(q.sum()),
        "target_probe_size": float(probe.shape[0]),
        "diagnostic_draws_per_repeat": float(draws_per_repeat),
        "diagnostic_repeats": float(repeats),
    }
    cell_summary: dict[str, dict[str, object]] = {}
    for cell in target_support.tolist():
        values = per_cell_values.get(int(cell), [])
        floors = per_cell_floor.get(int(cell), [])
        value_mean, value_se = _mean_se(values)
        floor_cell_mean, floor_cell_se = _mean_se(floors)
        cell_summary[str(int(cell))] = {
            "q": float(q[int(cell)]),
            "target_probe_count": float(counts[int(cell)]),
            "observed_count": float(0.0 if shared.observed_counts is None else shared.observed_counts[int(cell)]),
            "estimable": float(int(cell) in estimable),
            "w1": value_mean if values else None,
            "w1_mc_se": value_se if values else None,
            "weighted_w1": float(q[int(cell)] * value_mean) if values else None,
            "route_self_mc_distance": floor_cell_mean if floors else None,
            "route_self_mc_se": floor_cell_se if floors else None,
            "alias": float(-1 if alias is None else alias[int(cell)]),
        }
    return summary, cell_summary


def _fit_ihdp_shared(seed: int, quick: bool):
    dgp = IHDPDistDGP(seed=seed + 1, split_seed=seed + 2)
    train = dgp.observed_split("train", seed=seed + 3)
    validation_x, _ = dgp.split("validation")
    test_x, _ = dgp.split("test")
    outcome_lower, outcome_upper = dgp.outcome_bounds(train["y"])
    cell_transform = core.fit_pca_treatment_cell_transform(train["x"], n_components=2)
    candidate_specs: list[_IHDPConfigWeights] = [
        {"generator_transport_weight": 0.75, "factual_crps_weight": 8.0, "factual_mse_weight": 2.0},
        {"generator_transport_weight": 1.25, "factual_crps_weight": 8.0, "factual_mse_weight": 2.0},
        {"generator_transport_weight": 0.75, "factual_crps_weight": 12.0, "factual_mse_weight": 1.0},
        {"generator_transport_weight": 1.25, "factual_crps_weight": 12.0, "factual_mse_weight": 1.0},
    ]
    candidates: list[tuple[float, GANICE, GANICEConfig, int]] = []
    for restart in range(2 if quick else 4):
        spec = candidate_specs[restart]
        config = core.ihdp_ganice_config(
            seed + 41 + 100 * restart,
            dgp.feature_dim,
            outcome_lower,
            outcome_upper,
            quick=quick,
            cell_resolution=(1, 1, 1),
            min_cell_samples=6,
            shared_generator=True,
            **spec,
        )
        model = core.fit_ganice(
            d_w=dgp.d_w,
            train_w=train["w"],
            train_y=train["y"],
            target_w_sampler=dgp.sample_target_w,
            config=config,
            seed=seed + 43 + 100 * restart,
            d_cell_w=3,
            cell_transform=cell_transform,
        )
        score = core.ihdp_extended_w1(
            dgp,
            core.make_ganice_binary_sampler(model, dgp),
            validation_x,
            64 if quick else 96,
            seed + 40_000 + 1_000 * restart,
        )
        candidates.append((score, model, config, restart))
    return dgp, train, test_x, min(candidates, key=lambda item: item[0])


def _fit_jobs_shared(seed: int, quick: bool):
    data = JobsLaLondeData(split_seed=seed + 1, include_nsw_control_in_observed=False)
    train = data.observed_split("train")
    validation_rct = data.rct_split("validation")
    test_rct = data.rct_split("test")
    outcome_lower, outcome_upper = data.outcome_bounds(train["y"])
    cell_transform = core.fit_jobs_cell_transform(train["x"])
    base = core.jobs_ganice_config(
        seed + 101,
        data.feature_dim,
        outcome_lower,
        outcome_upper,
        quick=quick,
    )
    specs: list[_JobsConfigOverrides]
    if quick:
        specs = [
            {"num_steps": 120, "generator_transport_weight": 4.0, "factual_crps_weight": 4.0},
            {"num_steps": 160, "generator_transport_weight": 6.0, "factual_crps_weight": 5.0},
            {"num_steps": 180, "generator_transport_weight": 4.0, "factual_crps_weight": 4.0},
        ]
    else:
        specs = [
            {"num_steps": 180, "generator_transport_weight": 4.0, "factual_crps_weight": 4.0},
            {"num_steps": 220, "generator_transport_weight": 6.0, "factual_crps_weight": 5.0},
            {"num_steps": 260, "generator_transport_weight": 4.0, "factual_crps_weight": 4.0},
        ]
    zero_masses = core.jobs_nsw_zero_masses(data, "train")
    validation_att = data.rct_att_earnings("validation")
    candidates: list[tuple[float, GANICE, GANICEConfig, int]] = []
    for restart, spec in enumerate(specs):
        config = replace(base, seed=seed + 101 + 100 * restart, **spec)
        model = core.fit_ganice(
            d_w=data.d_w,
            train_w=train["w"],
            train_y=train["y"],
            target_w_sampler=lambda n, seed=None, fitted_data=data: fitted_data.sample_target_w(
                n,
                seed=seed,
                split="test",
            ),
            config=config,
            seed=seed + 151 + 100 * restart,
            d_cell_w=3,
            cell_transform=cell_transform,
        )
        sampler = core.make_ganice_jobs_sampler(model, data, zero_masses=zero_masses)
        validation_metrics, _ = core.jobs_evaluate_sampler(
            sampler,
            validation_rct,
            validation_att,
            n_per_x=32 if quick else 64,
            mean_mc=32 if quick else 64,
            seed=seed + 50_000 + 1_000 * restart,
            include_additional=False,
        )
        candidates.append((validation_metrics["rct_w1"], model, config, restart))
    return data, train, validation_rct, test_rct, zero_masses, min(candidates, key=lambda item: item[0])


def _common_transfer_payload(
    shared: GANICE,
    routed: HardRoutedReference,
    target_probe: NDArray,
    reference_diagnostics: dict[str, float | int | list[int]],
    *,
    diagnostic_draws: int,
    diagnostic_repeats: int,
    seed: int,
) -> dict[str, object]:
    transfer, cells = estimate_transfer(
        shared,
        routed,
        target_probe,
        draws_per_repeat=diagnostic_draws,
        repeats=diagnostic_repeats,
        seed=seed,
    )
    numeric_reference = {
        key: float(value) for key, value in reference_diagnostics.items() if isinstance(value, (int, float))
    }
    transfer.update({f"reference_{key}": value for key, value in numeric_reference.items()})
    return {"metrics": transfer, "cells": cells}


def run_ihdp_repetition(
    repetition: int,
    *,
    base_seed: int,
    quick: bool,
    diagnostic_draws: int,
    diagnostic_repeats: int,
    target_probe_size: int,
    existing_root: Path,
) -> dict[str, object]:
    seed = int(base_seed + 1_000 + 10_000 * repetition)
    dgp, train, test_x, selected = _fit_ihdp_shared(seed, quick)
    validation_score, shared, selected_config, selected_restart = selected
    target_probe = dgp.sample_target_w(target_probe_size, seed=seed + 910_001)
    probe_cells = shared._cell_ids(target_probe)
    q = np.bincount(probe_cells, minlength=shared.num_cells).astype(np.float64)
    q /= max(float(q.sum()), 1.0)
    routed, reference_diagnostics = fit_hard_routed_reference(
        shared,
        train["w"],
        train["y"],
        q,
        base_config=selected_config,
        seed=seed + 920_001,
    )
    shared_ew = core.ihdp_extended_w1(
        dgp,
        core.make_ganice_binary_sampler(shared, dgp),
        test_x,
        96 if quick else 192,
        seed + 20_000,
    )

    def routed_sampler(x: NDArray, treatment: int, n: int, sample_seed: Optional[int]) -> NDArray:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
        outputs = []
        for row_idx in range(x_arr.shape[0]):
            outputs.append(
                routed.sample_conditional(
                    dgp.encode_w(x_arr[row_idx:row_idx + 1], treatment),
                    n,
                    seed=None if sample_seed is None else sample_seed + row_idx,
                ).reshape(1, n, 1)
            )
        return np.concatenate(outputs, axis=0).astype(np.float32)

    routed_ew = core.ihdp_extended_w1(
        dgp,
        routed_sampler,
        test_x,
        96 if quick else 192,
        seed + 20_000,
    )
    common = _common_transfer_payload(
        shared,
        routed,
        target_probe,
        reference_diagnostics,
        diagnostic_draws=diagnostic_draws,
        diagnostic_repeats=diagnostic_repeats,
        seed=seed + 930_001,
    )
    metrics = common["metrics"]
    assert isinstance(metrics, dict)
    metrics.update(
        {
            "shared_ew": float(shared_ew),
            "routed_ew": float(routed_ew),
            "hard_gated_eval_ew": float(routed_ew),
            "validation_shared_ew": float(validation_score),
            "selected_shared_restart": float(selected_restart),
        }
    )
    return {"benchmark": "IHDP", "repetition": repetition, **common}


def run_jobs_repetition(
    repetition: int,
    *,
    base_seed: int,
    quick: bool,
    diagnostic_draws: int,
    diagnostic_repeats: int,
    target_probe_size: int,
    existing_root: Path,
) -> dict[str, object]:
    seed = int(base_seed + 1_500 + 10_000 * repetition)
    data, train, validation_rct, test_rct, zero_masses, selected = _fit_jobs_shared(seed, quick)
    validation_score, shared, selected_config, selected_restart = selected
    target_probe = data.sample_target_w(target_probe_size, seed=seed + 910_001, split="test")
    probe_cells = shared._cell_ids(target_probe)
    q = np.bincount(probe_cells, minlength=shared.num_cells).astype(np.float64)
    q /= max(float(q.sum()), 1.0)
    routed, reference_diagnostics = fit_hard_routed_reference(
        shared,
        train["w"],
        train["y"],
        q,
        base_config=selected_config,
        seed=seed + 920_001,
    )
    shared_uncalibrated = core.make_ganice_jobs_sampler(shared, data, zero_masses=zero_masses)
    shared_calibrators = core.fit_jobs_arm_quantile_calibrators(
        shared_uncalibrated,
        validation_rct,
        n_per_x=64 if quick else 128,
        seed=seed + 55_000,
    )
    shared_reported = core.make_ganice_jobs_sampler(
        shared,
        data,
        zero_masses=zero_masses,
        arm_quantile_calibrators=shared_calibrators,
    )
    routed_uncalibrated = core.make_ganice_jobs_sampler(routed, data, zero_masses=zero_masses)
    routed_calibrators = core.fit_jobs_arm_quantile_calibrators(
        routed_uncalibrated,
        validation_rct,
        n_per_x=64 if quick else 128,
        seed=seed + 55_000,
    )
    routed_reported = core.make_ganice_jobs_sampler(
        routed,
        data,
        zero_masses=zero_masses,
        arm_quantile_calibrators=routed_calibrators,
    )
    routed_shared_calibration = core.make_ganice_jobs_sampler(
        routed,
        data,
        zero_masses=zero_masses,
        arm_quantile_calibrators=shared_calibrators,
    )
    true_att = data.rct_att_earnings("test")
    eval_samples = 48 if quick else 96
    mean_mc = 48 if quick else 96
    eval_seed = seed + 65_000
    shared_metrics, _ = core.jobs_evaluate_sampler(
        shared_reported,
        test_rct,
        true_att,
        n_per_x=eval_samples,
        mean_mc=mean_mc,
        seed=eval_seed,
        include_additional=False,
    )
    routed_metrics, _ = core.jobs_evaluate_sampler(
        routed_reported,
        test_rct,
        true_att,
        n_per_x=eval_samples,
        mean_mc=mean_mc,
        seed=eval_seed,
        include_additional=False,
    )
    routed_shared_calibration_metrics, _ = core.jobs_evaluate_sampler(
        routed_shared_calibration,
        test_rct,
        true_att,
        n_per_x=eval_samples,
        mean_mc=mean_mc,
        seed=eval_seed,
        include_additional=False,
    )
    shared_raw_metrics, _ = core.jobs_evaluate_sampler(
        core.make_ganice_jobs_sampler(shared, data),
        test_rct,
        true_att,
        n_per_x=eval_samples,
        mean_mc=mean_mc,
        seed=eval_seed,
        include_additional=False,
    )
    routed_raw_metrics, _ = core.jobs_evaluate_sampler(
        core.make_ganice_jobs_sampler(routed, data),
        test_rct,
        true_att,
        n_per_x=eval_samples,
        mean_mc=mean_mc,
        seed=eval_seed,
        include_additional=False,
    )
    common = _common_transfer_payload(
        shared,
        routed,
        target_probe,
        reference_diagnostics,
        diagnostic_draws=diagnostic_draws,
        diagnostic_repeats=diagnostic_repeats,
        seed=seed + 930_001,
    )
    metrics = common["metrics"]
    assert isinstance(metrics, dict)
    metrics.update(
        {
            "shared_ew": float(shared_metrics["rct_w1"]),
            "routed_ew": float(routed_metrics["rct_w1"]),
            "hard_gated_eval_ew": float(routed_metrics["rct_w1"]),
            "shared_ew_earnings": float(shared_metrics["rct_w1_earnings"]),
            "routed_ew_earnings": float(routed_metrics["rct_w1_earnings"]),
            "routed_ew_shared_calibration": float(routed_shared_calibration_metrics["rct_w1"]),
            "routed_ew_earnings_shared_calibration": float(routed_shared_calibration_metrics["rct_w1_earnings"]),
            "shared_raw_ew": float(shared_raw_metrics["rct_w1"]),
            "routed_raw_ew": float(routed_raw_metrics["rct_w1"]),
            "validation_shared_ew": float(validation_score),
            "selected_shared_restart": float(selected_restart),
            "shared_arm_calibrator_count": float(len(shared_calibrators)),
            "routed_arm_calibrator_count": float(len(routed_calibrators)),
        }
    )
    return {"benchmark": "Jobs", "repetition": repetition, **common}


def run_tcga_repetition(
    repetition: int,
    *,
    base_seed: int,
    quick: bool,
    diagnostic_draws: int,
    diagnostic_repeats: int,
    target_probe_size: int,
    existing_root: Path,
) -> dict[str, object]:
    seed = int(base_seed + 3_500 + 113 * repetition)
    dgp = TCGADoseDGP(data_dir=ROOT / "Data" / "TCGA", seed=seed)
    train = dgp.sample_observed("train", seed=seed + 1)
    test_x, test_z = dgp.split_arrays("test")
    if quick:
        train_n = min(2_400, train["x"].shape[0])
        train = {key: value[:train_n] for key, value in train.items()}
        test_x = test_x[:64]
        test_z = test_z[:64]
    outcome_lower = float(np.quantile(train["y"], 0.002) - 0.4)
    outcome_upper = float(np.quantile(train["y"], 0.998) + 0.4)
    config = core.tcga_ganice_config(seed + 102, dgp.pc_dim, outcome_lower, outcome_upper, quick=quick)
    train_w = core.encode_tcga_ganice_w(dgp, train["z"], train["t"], train["d"])

    def target_sampler(n: int, seed: Optional[int] = None) -> NDArray:
        return core.sample_tcga_ganice_target_w(dgp, n, seed=seed, split="test")

    shared = core.fit_ganice(
        d_w=dgp.pc_dim + 2,
        train_w=train_w,
        train_y=train["y"],
        target_w_sampler=target_sampler,
        config=config,
        seed=seed + 202,
    )
    target_probe = target_sampler(target_probe_size, seed=seed + 910_001)
    probe_cells = shared._cell_ids(target_probe)
    q = np.bincount(probe_cells, minlength=shared.num_cells).astype(np.float64)
    q /= max(float(q.sum()), 1.0)
    routed, reference_diagnostics = fit_hard_routed_reference(
        shared,
        train_w,
        train["y"],
        q,
        base_config=config,
        seed=seed + 920_001,
    )

    def shared_sampler(x, z, treatment, dosage, n, sample_seed):
        del x
        return shared.sample_conditional(
            core.encode_tcga_ganice_w(dgp, z, treatment, dosage),
            n,
            seed=sample_seed,
        )

    def routed_sampler(x, z, treatment, dosage, n, sample_seed):
        del x
        return routed.sample_conditional(
            core.encode_tcga_ganice_w(dgp, z, treatment, dosage),
            n,
            seed=sample_seed,
        )

    eval_n = min(40 if quick else 120, test_x.shape[0])
    n_per_state = 64 if quick else 128
    shared_ew = core.tcga_extended_w1(
        dgp,
        test_x[:eval_n],
        test_z[:eval_n],
        dgp.dosage_grid,
        shared_sampler,
        n_per_state=n_per_state,
        seed=seed + 80_000,
    )
    routed_ew = core.tcga_extended_w1(
        dgp,
        test_x[:eval_n],
        test_z[:eval_n],
        dgp.dosage_grid,
        routed_sampler,
        n_per_state=n_per_state,
        seed=seed + 80_000,
    )
    common = _common_transfer_payload(
        shared,
        routed,
        target_probe,
        reference_diagnostics,
        diagnostic_draws=diagnostic_draws,
        diagnostic_repeats=diagnostic_repeats,
        seed=seed + 930_001,
    )
    metrics = common["metrics"]
    assert isinstance(metrics, dict)
    metrics.update(
        {
            "shared_ew": float(shared_ew),
            "routed_ew": float(routed_ew),
            "hard_gated_eval_ew": float(routed_ew),
            "selected_shared_restart": 0.0,
        }
    )
    return {"benchmark": "TCGA", "repetition": repetition, **common}


def _run_task(task: tuple) -> dict[str, object]:
    (
        benchmark,
        repetition,
        base_seed,
        quick,
        diagnostic_draws,
        diagnostic_repeats,
        target_probe_size,
        existing_root,
    ) = task
    kwargs = {
        "base_seed": base_seed,
        "quick": quick,
        "diagnostic_draws": diagnostic_draws,
        "diagnostic_repeats": diagnostic_repeats,
        "target_probe_size": target_probe_size,
        "existing_root": Path(existing_root),
    }
    if benchmark == "IHDP":
        return run_ihdp_repetition(repetition, **kwargs)
    if benchmark == "Jobs":
        return run_jobs_repetition(repetition, **kwargs)
    if benchmark == "TCGA":
        return run_tcga_repetition(repetition, **kwargs)
    raise ValueError(f"unknown benchmark {benchmark}")


def _numeric_metrics(result: dict[str, object]) -> dict[str, float]:
    metrics = result["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError("result metrics must be a dictionary")
    return {
        str(key): float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }


def summarize(results: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for benchmark in BENCHMARKS:
        selected = [result for result in results if result["benchmark"] == benchmark]
        if not selected:
            continue
        keys = sorted(set.intersection(*(set(_numeric_metrics(result)) for result in selected)))
        row: dict[str, float] = {"n_repetitions": float(len(selected))}
        for key in keys:
            values = [value for result in selected if np.isfinite(value := _numeric_metrics(result)[key])]
            if not values:
                continue
            mean, se = _mean_se(values)
            row[f"{key}_mean"] = mean
            row[f"{key}_se"] = se
        metric_rows = [_numeric_metrics(result) for result in selected]
        paired_difference = [
            metrics["routed_ew"] - metrics["shared_ew"]
            for metrics in metric_rows
            if "routed_ew" in metrics and "shared_ew" in metrics
        ]
        if paired_difference:
            mean, se = _mean_se(paired_difference)
            row["paired_routed_minus_shared_ew_mean"] = mean
            row["paired_routed_minus_shared_ew_se"] = se
            row["shared_ew_better_fraction"] = float(np.mean(np.asarray(paired_difference, dtype=np.float64) > 0.0))
            row["routed_ew_better_fraction"] = float(np.mean(np.asarray(paired_difference, dtype=np.float64) < 0.0))
        transfer_excess = [
            metrics["epsilon_impl_raw_original_partition"] - metrics["route_self_mc_distance"]
            for metrics in metric_rows
            if "epsilon_impl_raw_original_partition" in metrics and "route_self_mc_distance" in metrics
        ]
        if transfer_excess:
            mean, se = _mean_se(transfer_excess)
            row["epsilon_minus_route_self_mc_distance_mean"] = mean
            row["epsilon_minus_route_self_mc_distance_se"] = se
        if benchmark == "Jobs":
            shared_calibration_difference = [
                metrics["routed_ew_shared_calibration"] - metrics["shared_ew"]
                for metrics in metric_rows
                if "routed_ew_shared_calibration" in metrics and "shared_ew" in metrics
            ]
            if shared_calibration_difference:
                mean, se = _mean_se(shared_calibration_difference)
                row["paired_routed_shared_calibration_minus_shared_ew_mean"] = mean
                row["paired_routed_shared_calibration_minus_shared_ew_se"] = se
        summary[benchmark] = row
    return summary


def write_outputs(
    output_dir: Path,
    results: list[dict[str, object]],
    summary: Optional[dict[str, dict[str, float]]] = None,
    manifest: Optional[dict[str, object]] = None,
) -> None:
    """Export the reported paired comparison and its numerical diagnostics."""
    del manifest  # Accepted for callers migrating historical in-memory results.
    output_dir = ensure_dir(output_dir)
    if summary is None:
        summary = summarize(results)
    rows: list[_ImplementationTableRow] = []
    for benchmark in BENCHMARKS:
        if benchmark not in summary:
            continue
        row = summary[benchmark]
        shared_error = row["shared_ew_mean"]
        rows.append(
            {
                "benchmark": benchmark,
                "repetitions": int(row["n_repetitions"]),
                "shared_error": shared_error,
                "shared_error_se": row["shared_ew_se"],
                "routed_error": row["routed_ew_mean"],
                "routed_error_se": row["routed_ew_se"],
                "epsilon": row["epsilon_impl_raw_original_partition_mean"],
                "epsilon_se": row["epsilon_impl_raw_original_partition_se"],
                "ratio": row["epsilon_impl_raw_original_partition_mean"] / shared_error
                if shared_error
                else float("nan"),
            }
        )
    if not rows:
        raise ValueError("no implementation diagnostic results to export")
    with (output_dir / "implementation_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tex = [
        r"\begin{table}[tb]",
        r"\centering",
        r"\small",
        (
            r"\caption{Paired implementation diagnostics. Standard errors are in parentheses; "
            r"repetition counts appear beside benchmark names.}"
        ),
        r"\label{tab:implementation-diagnostics}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Benchmark & Shared error & Routed error & $\hat{\varepsilon}$ & Ratio \\",
        r"\midrule",
    ]
    for table_row in rows:
        values = [
            rf"${table_row['shared_error']:.3f}\;({table_row['shared_error_se']:.3f})$",
            rf"${table_row['routed_error']:.3f}\;({table_row['routed_error_se']:.3f})$",
            rf"${table_row['epsilon']:.3f}\;({table_row['epsilon_se']:.3f})$",
        ]
        # The manuscript defines the displayed ratio using the displayed means.
        displayed_shared = round(table_row["shared_error"], 3)
        displayed_ratio = round(table_row["epsilon"], 3) / displayed_shared if displayed_shared else float("nan")
        tex.append(
            f"{table_row['benchmark']} ({table_row['repetitions']}) & "
            + " & ".join(values)
            + f" & ${displayed_ratio:.2f}$ "
            + r"\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output_dir / "implementation_diagnostics.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    metric_prefixes = (
        "route_self_mc_distance",
        "epsilon_minus_route_self_mc_distance",
        "hard_reference_common_draw_epsilon_estimable_cells",
        "hard_reference_gate_eta_estimable_cells",
        "hard_reference_gate_transfer_bound_estimable_cells",
        "missing_target_mass",
    )
    fields = ["benchmark"] + [f"{key}_{suffix}" for key in metric_prefixes for suffix in ("mean", "se")]
    with (output_dir / "implementation_numbers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for benchmark in BENCHMARKS:
            if benchmark in summary:
                writer.writerow(
                    {
                        "benchmark": benchmark,
                        **{key: summary[benchmark][key] for key in fields[1:] if key in summary[benchmark]},
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paper's paired implementation diagnostics.")
    parser.add_argument(
        "--output-dir", "--output-root", dest="output_dir", type=Path, default=Path("outputs/implementation")
    )
    parser.add_argument("--existing-root", type=Path, default=Path("outputs"), help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repetitions", type=int, help="Override paired repetition counts for every benchmark.")
    parser.add_argument("--ihdp-repetitions", type=int)
    parser.add_argument("--jobs-repetitions", type=int)
    parser.add_argument("--tcga-repetitions", type=int)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--diagnostic-draws", type=int, default=DEFAULT_DIAGNOSTIC_DRAWS)
    parser.add_argument("--diagnostic-repeats", type=int, default=DEFAULT_DIAGNOSTIC_REPEATS)
    parser.add_argument("--target-probe-size", type=int, default=DEFAULT_TARGET_PROBE_SIZE)
    parser.add_argument("--benchmark", choices=("all", "ihdp", "jobs", "tcga"), default="all")
    args = parser.parse_args()
    repetition_counts = {
        benchmark: (args.repetitions if args.repetitions is not None else count)
        for benchmark, count in DEFAULT_REPETITIONS.items()
    }
    for benchmark in BENCHMARKS:
        override = getattr(args, f"{benchmark.lower()}_repetitions")
        if override is not None:
            repetition_counts[benchmark] = override
    if any(count <= 0 for count in repetition_counts.values()) or args.parallel < 1:
        raise ValueError("repetition and parallel counts must be positive")
    if args.diagnostic_draws <= 1 or args.diagnostic_repeats <= 0 or args.target_probe_size <= 0:
        raise ValueError("diagnostic draws must exceed one; repeats and target probe size must be positive")
    benchmark_names = {"ihdp": "IHDP", "jobs": "Jobs", "tcga": "TCGA"}
    selected = list(BENCHMARKS) if args.benchmark == "all" else [benchmark_names[args.benchmark]]
    tasks = [
        (
            benchmark,
            repetition,
            args.seed,
            args.quick,
            args.diagnostic_draws,
            args.diagnostic_repeats,
            args.target_probe_size,
            str(args.existing_root),
        )
        for benchmark in selected
        for repetition in range(repetition_counts[benchmark])
    ]
    results = []

    def record(result):
        results.append(result)
        benchmark = result["benchmark"]
        print(
            f"finished {benchmark} repetition {int(result['repetition']) + 1}/{repetition_counts[benchmark]}",
            flush=True,
        )

    if args.parallel == 1:
        for task in tasks:
            record(_run_task(task))
    else:
        with ProcessPoolExecutor(max_workers=min(args.parallel, len(tasks))) as executor:
            futures = [executor.submit(_run_task, task) for task in tasks]
            for future in as_completed(futures):
                record(future.result())
    results.sort(key=lambda result: (BENCHMARKS.index(str(result["benchmark"])), int(result["repetition"])))
    write_outputs(args.output_dir, results)
    print(f"saved implementation diagnostics to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
