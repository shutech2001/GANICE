from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
from typing import Callable

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from counterfactual_gan import (  # noqa: E402
    DiffPO,
    DiffPOConfig,
    DRLearner,
    DRLearnerConfig,
    DRNet,
    DRNetConfig,
    GANICE,
    GANICEConfig,
    GANITE,
    GANITEConfig,
    IHDPDistDGP,
    INFs,
    INFsConfig,
    JobsLaLondeData,
    POFlow,
    POFlowConfig,
    SCIGAN,
    SCIGANConfig,
    TCGADoseDGP,
    VCNet,
    VCNetConfig,
    download_tcga_db,
)
from counterfactual_gan.jobs import model_scale_to_earnings  # noqa: E402
from counterfactual_gan.metrics import (  # noqa: E402
    central_interval_coverage_width_1d,
    crps_empirical_1d,
    cvm_2samp_1d,
    energy_distance_1d,
    ks_2samp_1d,
    mmd2_gaussian_median_1d,
    pit_histogram_1d,
    quantile_squared_error_sum_1d,
    sample_wasserstein_1_1d,
    tail_mean_error_1d,
)
from counterfactual_gan.utils import ensure_dir  # noqa: E402


COLORS = {
    "true": "#1b9e77",
    "ganite": "#7570b3",
    "po_flow": "#4daf4a",
    "diff_po": "#e7298a",
    "infs": "#a6761d",
    "dr_learner": "#666666",
    "scigan": "#7570b3",
    "vcnet": "#66a61e",
    "drnet": "#e6ab02",
    "ganice": "#1f78b4",
}
LABELS = {
    "ganite": "GANITE",
    "po_flow": "PO-Flow",
    "diff_po": "Diff-PO",
    "infs": "INFs",
    "dr_learner": "DR-Learner",
    "scigan": "SCIGAN",
    "vcnet": "VCNet",
    "drnet": "DRNet",
    "ganice": "GANICE",
    "ganice_no_cell_norm": "GANICE no cell norm",
    "ganice_pooled": "Pooled WGAN",
}
IHDP_QTE_METHODS = ["ganite", "po_flow", "diff_po", "infs", "dr_learner", "ganice"]
DEFAULT_METHODS = ["ganite", "po_flow", "diff_po", "infs", "dr_learner", "ganice"]
TCGA_METHODS = ["scigan", "vcnet", "drnet", "ganice"]
ADDITIONAL_QUANTILES = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=np.float64)
TAIL_LEVELS = np.array([0.05, 0.10, 0.90, 0.95], dtype=np.float64)
COVERAGE_LEVELS = np.array([0.50, 0.80, 0.90, 0.95], dtype=np.float64)
PIT_BINS = 10


def configure_matplotlib_fonts() -> None:
    for font_path in (
        Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
        Path("/Library/Fonts/Times New Roman.ttf"),
        Path("/System/Library/Fonts/Times New Roman.ttf"),
    ):
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            break
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "font.size": 22,
            "axes.titlesize": 22,
            "axes.labelsize": 25,
            "xtick.labelsize": 21,
            "ytick.labelsize": 21,
            "legend.fontsize": 21,
            "figure.titlesize": 22,
            "lines.linewidth": 2.8,
            "savefig.bbox": "tight",
        }
    )


configure_matplotlib_fonts()
torch.set_num_threads(1)


def _run_parallel_repetitions(worker, tasks: list[tuple], parallel: int) -> list:
    if parallel <= 1 or len(tasks) <= 1:
        return [worker(task) for task in tasks]
    results: list[object | None] = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=min(int(parallel), len(tasks))) as executor:
        futures = {executor.submit(worker, task): idx for idx, task in enumerate(tasks)}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            print(f"finished repetition {idx + 1}/{len(tasks)}", flush=True)
    return results


def save_grouped_bar_plot(
    output_path: Path,
    labels: list[str],
    series: list[tuple[str, np.ndarray, str]],
    title: str,
    ylabel: str,
    rotation: int = 25,
) -> None:
    x_axis = np.arange(len(labels))
    num_series = len(series)
    width = 0.8 / num_series
    offsets = (np.arange(num_series) - (num_series - 1) / 2.0) * width

    plt.figure(figsize=(max(10.8, 1.35 * len(labels)), 6.3))
    for idx, (name, values, color) in enumerate(series):
        plt.bar(x_axis + offsets[idx], values, width=width, label=name, color=color)
    plt.xticks(x_axis, labels, rotation=rotation)
    plt.ylabel(ylabel)
    place_legend_outside()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def place_legend_outside(**kwargs) -> None:
    handles, labels = plt.gca().get_legend_handles_labels()
    if not handles:
        return
    ncol = kwargs.pop("ncol", max(1, int(np.ceil(len(handles) / 2.0))))
    plt.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0.0,
        ncol=ncol,
        columnspacing=1.2,
        handlelength=2.2,
        **kwargs,
    )


def save_metric_bar_plot(
    output_prefix: Path,
    labels: list[str],
    values: list[float],
    errors: list[float],
    colors: list[str],
    *,
    title: str,
    ylabel: str,
    rotation: int = 25,
) -> None:
    x_axis = np.arange(len(labels))
    plt.figure(figsize=(max(11.0, 1.42 * len(labels)), 6.4))
    plt.bar(
        x_axis,
        values,
        yerr=errors,
        capsize=5,
        color=colors,
        error_kw={"elinewidth": 1.7, "capthick": 1.7},
    )
    plt.xticks(x_axis, labels, rotation=rotation, ha="right" if rotation else "center")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_prefix.with_suffix(".png"), dpi=260)
    plt.savefig(output_prefix.with_suffix(".pdf"))
    plt.close()


def _coverage_suffix(coverage: float) -> str:
    return f"{int(round(100.0 * coverage)):02d}"


def _pit_key(idx: int) -> str:
    return f"pit_bin_{idx:02d}"


def _summary_mean(row: dict[str, float], metric: str) -> float:
    if metric in row:
        return float(row[metric])
    return float(row.get(f"{metric}_mean", np.nan))


def _summary_se(row: dict[str, float], metric: str) -> float:
    return float(row.get(f"{metric}_se", 0.0))


def _format_table_value(value: float, se: float) -> str:
    if not np.isfinite(value):
        return "--"
    scale = max(abs(value), abs(se))
    if scale >= 100.0:
        return f"{value:.0f} $\\pm$ {se:.0f}"
    if scale >= 10.0:
        return f"{value:.2f} $\\pm$ {se:.2f}"
    if scale >= 0.01:
        return f"{value:.3f} $\\pm$ {se:.3f}"
    return f"{value:.2e} $\\pm$ {se:.1e}"


def _write_additional_metric_table(
    output_dir: Path,
    prefix: str,
    summary: dict[str, dict[str, float]],
    methods: list[str],
    metrics: list[tuple[str, str, str]],
) -> None:
    methods = [method for method in methods if method in summary]
    if not methods:
        return
    csv_header = ["method"]
    for metric_key, _, _ in metrics:
        csv_header.extend([metric_key, f"{metric_key}_se"])
    csv_lines = [",".join(csv_header)]
    tex_lines = [
        "\\begin{tabular}{" + "l" + "c" * len(metrics) + "}",
        "\\toprule",
        "Method & " + " & ".join(label for _, label, _ in metrics) + " \\\\",
        "\\midrule",
    ]
    for method in methods:
        row = summary[method]
        csv_values = [LABELS[method]]
        tex_values = []
        for metric_key, _, direction in metrics:
            mean_value = _summary_mean(row, metric_key)
            se_value = _summary_se(row, metric_key)
            csv_values.extend([f"{mean_value:.8g}", f"{se_value:.8g}"])
            cell = _format_table_value(mean_value, se_value)
            available = [m for m in methods if np.isfinite(_summary_mean(summary[m], metric_key))]
            if available:
                if direction == "max":
                    best = max(available, key=lambda m: _summary_mean(summary[m], metric_key))
                else:
                    best = min(available, key=lambda m: _summary_mean(summary[m], metric_key))
                if method == best and np.isfinite(mean_value):
                    cell = f"\\textbf{{{cell}}}"
            tex_values.append(cell)
        csv_lines.append(",".join(csv_values))
        tex_lines.append(f"{LABELS[method]} & " + " & ".join(tex_values) + " \\\\")
    tex_lines += ["\\bottomrule", "\\end{tabular}"]
    (output_dir / f"{prefix}.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    (output_dir / f"{prefix}.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def save_metric_grid_bar_plot(
    output_prefix: Path,
    summary: dict[str, dict[str, float]],
    methods: list[str],
    metrics: list[tuple[str, str]],
    *,
    rotation: int = 25,
) -> None:
    methods = [method for method in methods if method in summary]
    if not methods:
        return
    ncols = 3
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False)
    labels = [LABELS[method] for method in methods]
    x_axis = np.arange(len(methods))
    colors = [COLORS.get(method, "#999999") for method in methods]
    for ax, (metric_key, label) in zip(axes.ravel(), metrics, strict=False):
        values = [_summary_mean(summary[method], metric_key) for method in methods]
        errors = [_summary_se(summary[method], metric_key) for method in methods]
        ax.bar(
            x_axis,
            values,
            yerr=errors,
            capsize=4,
            color=colors,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
        )
        ax.set_title(label)
        ax.set_xticks(x_axis)
        ax.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center")
        ax.set_ylim(bottom=0.0)
    for ax in axes.ravel()[len(metrics) :]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_prefix.with_suffix(".png"), dpi=260)
    plt.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def save_interval_width_plot(
    output_prefix: Path,
    summary: dict[str, dict[str, float]],
    methods: list[str],
    *,
    ylabel: str,
) -> None:
    methods = [method for method in methods if method in summary]
    if not methods:
        return
    plt.figure(figsize=(10.8, 6.3))
    for method in methods:
        keys = [f"interval_width_{_coverage_suffix(float(c))}" for c in COVERAGE_LEVELS]
        means = [_summary_mean(summary[method], key) for key in keys]
        errors = [_summary_se(summary[method], key) for key in keys]
        plt.errorbar(
            COVERAGE_LEVELS,
            means,
            yerr=errors,
            marker="o",
            capsize=4,
            label=LABELS[method],
            color=COLORS.get(method, "#999999"),
            linewidth=2.0 if method != "ganice" else 3.0,
            linestyle="-" if method == "ganice" else "--",
        )
    plt.xlabel("nominal coverage")
    plt.ylabel(ylabel)
    plt.ylim(bottom=0.0)
    place_legend_outside()
    plt.tight_layout()
    plt.savefig(output_prefix.with_suffix(".png"), dpi=260)
    plt.savefig(output_prefix.with_suffix(".pdf"))
    plt.close()


def save_pit_histogram_plot(
    output_prefix: Path,
    summary: dict[str, dict[str, float]],
    methods: list[str],
) -> None:
    methods = [method for method in methods if method in summary]
    if not methods:
        return
    ncols = 3
    nrows = int(np.ceil(len(methods) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.1 * ncols, 3.7 * nrows), squeeze=False, sharey=True)
    centers = (np.arange(PIT_BINS, dtype=np.float64) + 0.5) / PIT_BINS
    for ax, method in zip(axes.ravel(), methods, strict=False):
        means = [_summary_mean(summary[method], _pit_key(idx)) for idx in range(PIT_BINS)]
        errors = [_summary_se(summary[method], _pit_key(idx)) for idx in range(PIT_BINS)]
        ax.bar(
            centers,
            means,
            yerr=errors,
            width=0.085,
            capsize=3,
            color=COLORS.get(method, "#999999"),
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        ax.axhline(1.0 / PIT_BINS, color="#333333", linestyle=":", linewidth=1.4)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, max(0.24, max(means) + max(errors, default=0.0) + 0.03))
        ax.set_title(LABELS[method])
        ax.set_xlabel("PIT")
    for ax in axes[:, 0]:
        ax.set_ylabel("frequency")
    for ax in axes.ravel()[len(methods) :]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_prefix.with_suffix(".png"), dpi=260)
    plt.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def fit_ganice(
    *,
    d_w: int,
    train_w: np.ndarray,
    train_y: np.ndarray,
    target_w_sampler,
    config: GANICEConfig,
    seed: int,
    d_cell_w: int | None = None,
    cell_transform=None,
) -> GANICE:
    model = GANICE(
        d_w=d_w,
        target_w_sampler=target_w_sampler,
        config=config,
        d_cell_w=d_cell_w,
        cell_transform=cell_transform,
    )
    model.fit(train_w, train_y, seed=seed)
    return model


def fit_pca_treatment_cell_transform(
    x_reference: np.ndarray,
    *,
    n_components: int,
    treatment_scale: tuple[float, float] = (0.0, 1.0),
):
    """Build a low-dimensional [0,1] cell map for finite-resolution critics."""

    x_ref = np.asarray(x_reference, dtype=np.float64)
    mean = x_ref.mean(axis=0)
    centered = x_ref - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components].T
    scores = centered @ components
    lo = np.quantile(scores, 0.02, axis=0)
    hi = np.quantile(scores, 0.98, axis=0)
    scale = np.maximum(hi - lo, 1e-6)
    t_lo, t_hi = treatment_scale
    t_scale = max(t_hi - t_lo, 1e-6)

    def transform(w: np.ndarray) -> np.ndarray:
        w_arr = np.asarray(w, dtype=np.float64)
        if w_arr.ndim == 1:
            w_arr = w_arr[None, :]
        x = w_arr[:, :-1]
        treatment = w_arr[:, -1:]
        projected = (x - mean) @ components
        projected = np.clip((projected - lo) / scale, 0.0, 1.0)
        treatment_unit = np.clip((treatment - t_lo) / t_scale, 0.0, 1.0)
        return np.column_stack([projected, treatment_unit]).astype(np.float32)

    return transform


def fit_jobs_cell_transform(x_reference: np.ndarray):
    x_ref = np.asarray(x_reference, dtype=np.float64)
    re75 = x_ref[:, 6]
    re75_lo = float(np.quantile(re75, 0.02))
    re75_hi = float(np.quantile(re75, 0.98))
    re75_scale = max(re75_hi - re75_lo, 1e-6)

    def transform(w: np.ndarray) -> np.ndarray:
        w_arr = np.asarray(w, dtype=np.float64)
        if w_arr.ndim == 1:
            w_arr = w_arr[None, :]
        x = w_arr[:, :-1]
        treatment = np.clip(w_arr[:, -1], 0.0, 1.0)
        re75_unit = np.clip((x[:, 6] - re75_lo) / re75_scale, 0.0, 1.0)
        black = np.clip(x[:, 2], 0.0, 1.0)
        return np.column_stack([re75_unit, black, treatment]).astype(np.float32)

    return transform


def tcga_ganice_config(
    seed: int,
    x_dim: int,
    outcome_lower: float,
    outcome_upper: float,
    *,
    quick: bool,
) -> GANICEConfig:
    return GANICEConfig(
        latent_dim=4,
        hidden_dims_generator=(48, 48) if quick else (96, 96),
        hidden_dims_critic=(48, 48) if quick else (96, 96),
        batch_size=128,
        num_steps=_maybe_quick(620, quick),
        critic_steps=1 if quick else 2,
        generator_lr=2e-4,
        critic_lr=1e-4,
        resolution=(0,) * x_dim + (2, 3),
        min_cell_samples=4,
        target_mass_samples=16_000 if quick else 45_000,
        cell_normalized=True,
        critic_uses_w=False,
        factual_crps_weight=10.0,
        factual_crps_samples=6,
        factual_mse_weight=8.0,
        factual_mse_samples=6,
        pretrain_steps=700 if quick else 1_600,
        pretrain_mse_weight=2.0,
        shared_generator=True,
        generator_transport_weight=0.6,
        outcome_lower=outcome_lower,
        outcome_upper=outcome_upper,
        seed=seed,
    )


def tcga_scigan_config(seed: int, x_dim: int, num_treatments: int, *, quick: bool) -> SCIGANConfig:
    return SCIGANConfig(
        x_dim=x_dim,
        num_treatments=num_treatments,
        hidden_dim=64,
        set_dim=16,
        batch_size=128,
        gan_iterations=_maybe_quick(700, quick),
        inference_iterations=_maybe_quick(1_000, quick),
        num_dosage_samples=5,
        alpha=1.0,
        seed=seed,
    )


def tcga_vcnet_config(seed: int, x_dim: int, *, quick: bool) -> VCNetConfig:
    return VCNetConfig(
        x_dim=x_dim,
        hidden_dim=48 if quick else 64,
        num_grid=10,
        spline_degree=2,
        spline_knots=(1.0 / 3.0, 2.0 / 3.0),
        batch_size=128,
        num_steps=_maybe_quick(1_100, quick),
        learning_rate=1e-3,
        weight_decay=1e-4,
        density_loss_weight=0.2,
        targeted_regularization=False,
        seed=seed,
    )


def tcga_drnet_config(seed: int, x_dim: int, num_treatments: int, *, quick: bool) -> DRNetConfig:
    return DRNetConfig(
        x_dim=x_dim,
        num_treatments=num_treatments,
        hidden_dim=48 if quick else 64,
        num_strata=5,
        base_layers=2,
        treatment_layers=1,
        head_layers=2,
        repeat_dosage=True,
        batch_size=128,
        num_steps=_maybe_quick(1_100, quick),
        learning_rate=1e-3,
        weight_decay=1e-4,
        seed=seed,
    )


def _maybe_quick(value: int, quick: bool) -> int:
    return max(20, int(round(value * (0.28 if quick else 1.0))))


def ihdp_ganice_config(
    seed: int,
    x_dim: int,
    outcome_lower: float,
    outcome_upper: float,
    *,
    quick: bool,
    cell_normalized: bool = True,
    pooled: bool = False,
    cell_resolution: tuple[int, ...] | None = None,
    min_cell_samples: int = 8,
    shared_generator: bool = True,
    factual_crps_weight: float | None = None,
    factual_mse_weight: float | None = None,
    generator_transport_weight: float | None = None,
) -> GANICEConfig:
    resolution = (0,) * (x_dim + 1) if pooled else (cell_resolution if cell_resolution is not None else (0,) * x_dim + (1,))
    return GANICEConfig(
        latent_dim=4,
        hidden_dims_generator=(64, 64) if quick else (128, 128),
        hidden_dims_critic=(64, 64) if quick else (128, 128),
        batch_size=128,
        num_steps=_maybe_quick(520, quick),
        critic_steps=1,
        generator_lr=2e-4,
        critic_lr=1e-4,
        resolution=resolution,
        min_cell_samples=min_cell_samples,
        target_mass_samples=16_000 if quick else 50_000,
        cell_normalized=cell_normalized,
        critic_uses_w=False,
        factual_loss_weight=0.0,
        factual_crps_weight=(8.0 if factual_crps_weight is None else factual_crps_weight) if not pooled else 0.0,
        factual_crps_samples=6,
        factual_mse_weight=(2.0 if factual_mse_weight is None else factual_mse_weight) if not pooled else 0.0,
        factual_mse_samples=6,
        generator_transport_weight=(0.75 if generator_transport_weight is None else generator_transport_weight) if not pooled else 3.0,
        residual_quantile_calibration=not pooled,
        calibration_samples_per_observation=8 if quick else 12,
        calibration_grid_size=192 if quick else 256,
        calibration_blend=0.75,
        pretrain_steps=0 if pooled else _maybe_quick(800, quick),
        pretrain_mse_weight=2.0,
        shared_generator=shared_generator and not pooled,
        outcome_lower=outcome_lower,
        outcome_upper=outcome_upper,
        seed=seed,
    )


def ihdp_ganite_config(seed: int, x_dim: int, outcome_lower: float, outcome_upper: float, *, quick: bool) -> GANITEConfig:
    return GANITEConfig(
        x_dim=x_dim,
        hidden_dim=64 if quick else 96,
        batch_size=128,
        cf_iterations=_maybe_quick(500, quick),
        ite_iterations=_maybe_quick(500, quick),
        cf_discriminator_steps=1 if quick else 2,
        alpha=2.0,
        beta=5.0,
        outcome_min=outcome_lower,
        outcome_max=outcome_upper,
        seed=seed,
    )


def ihdp_po_flow_config(seed: int, x_dim: int, outcome_lower: float, outcome_upper: float, *, quick: bool) -> POFlowConfig:
    return POFlowConfig(
        x_dim=x_dim,
        num_treatments=2,
        hidden_dim=64,
        batch_size=128,
        num_steps=_maybe_quick(750, quick),
        learning_rate=1e-3,
        rk4_steps=12 if quick else 20,
        outcome_min=outcome_lower,
        outcome_max=outcome_upper,
        seed=seed,
    )


def ihdp_diff_po_config(seed: int, x_dim: int, outcome_lower: float, outcome_upper: float, *, quick: bool) -> DiffPOConfig:
    return DiffPOConfig(
        x_dim=x_dim,
        num_treatments=2,
        hidden_dim=64,
        time_embedding_dim=96 if quick else 128,
        residual_blocks=3 if quick else 4,
        batch_size=128,
        propensity_steps=_maybe_quick(220, quick),
        diffusion_steps=_maybe_quick(350, quick),
        num_diffusion_steps=32 if quick else 45,
        learning_rate=5e-4,
        propensity_learning_rate=1e-3,
        outcome_min=outcome_lower,
        outcome_max=outcome_upper,
        seed=seed,
    )


def ihdp_infs_config(seed: int, x_dim: int, outcome_lower: float, outcome_upper: float, *, quick: bool) -> INFsConfig:
    return INFsConfig(
        x_dim=x_dim,
        num_treatments=2,
        hidden_dim=64,
        num_bins=48 if quick else 64,
        batch_size=128,
        nuisance_steps=_maybe_quick(700, quick),
        target_steps=_maybe_quick(700, quick),
        nuisance_lr=1e-3,
        target_lr=4e-3,
        prop_alpha=1.0,
        clip_propensity=0.05,
        noise_std_x=0.0,
        noise_std_y=0.01,
        outcome_min=outcome_lower,
        outcome_max=outcome_upper,
        seed=seed,
    )


def ihdp_dr_learner_config(seed: int, x_dim: int, outcome_lower: float, outcome_upper: float, *, quick: bool) -> DRLearnerConfig:
    return DRLearnerConfig(
        x_dim=x_dim,
        num_treatments=2,
        hidden_dim=64,
        batch_size=128,
        nuisance_steps=_maybe_quick(600, quick),
        final_steps=_maybe_quick(750, quick),
        n_folds=2,
        nuisance_lr=1e-3,
        final_lr=1e-3,
        propensity_clip=0.05,
        outcome_min=outcome_lower,
        outcome_max=outcome_upper,
        seed=seed,
    )


Sampler = Callable[[np.ndarray, int, int, int | None], np.ndarray]


def make_ganice_binary_sampler(model: GANICE, dgp: IHDPDistDGP) -> Sampler:
    def sampler(x: np.ndarray, treatment: int, n_per_x: int, seed: int | None) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
        samples = []
        for row_idx in range(x_arr.shape[0]):
            w = dgp.encode_w(x_arr[row_idx : row_idx + 1], treatment)
            samples.append(
                model.sample_conditional(
                    w,
                    n_per_x,
                    seed=None if seed is None else seed + row_idx,
                ).reshape(1, n_per_x, 1)
            )
        return np.concatenate(samples, axis=0).astype(np.float32)

    return sampler


def make_residual_plugin_sampler(
    predict_potential_outcomes: Callable[[np.ndarray], np.ndarray],
    x_train: np.ndarray,
    treatment_train: np.ndarray,
    y_train: np.ndarray,
    outcome_lower: float,
    outcome_upper: float,
) -> Sampler:
    train_mu = predict_potential_outcomes(x_train).astype(np.float64)
    residuals: dict[int, np.ndarray] = {}
    pooled = y_train.reshape(-1).astype(np.float64) - train_mu[np.arange(x_train.shape[0]), treatment_train]
    for treatment in (0, 1):
        mask = treatment_train == treatment
        residuals[treatment] = pooled[mask] if np.any(mask) else pooled

    def sampler(x: np.ndarray, treatment: int, n_per_x: int, seed: int | None) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
        treatment_arr = np.full(x_arr.shape[0], int(treatment), dtype=np.int64)
        mu = predict_potential_outcomes(x_arr)[np.arange(x_arr.shape[0]), treatment_arr].astype(np.float64)
        rng = np.random.default_rng(seed)
        pool = residuals[int(treatment)]
        draws = pool[rng.integers(0, pool.shape[0], size=(x_arr.shape[0], n_per_x))]
        samples = np.clip(mu[:, None] + draws, outcome_lower, outcome_upper)
        return samples[:, :, None].astype(np.float32)

    return sampler


def make_degenerate_mean_sampler(
    predict_potential_outcomes: Callable[[np.ndarray], np.ndarray],
    outcome_lower: float,
    outcome_upper: float,
) -> Sampler:
    def sampler(x: np.ndarray, treatment: int, n_per_x: int, seed: int | None) -> np.ndarray:
        del seed
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
        mean = predict_potential_outcomes(x_arr)[:, int(treatment)].astype(np.float64)
        samples = np.repeat(mean[:, None], n_per_x, axis=1)
        return np.clip(samples, outcome_lower, outcome_upper)[:, :, None].astype(np.float32)

    return sampler


def estimate_binary_means_from_sampler(
    sampler: Sampler,
    x: np.ndarray,
    n_mc: int,
    seed: int,
) -> np.ndarray:
    means = []
    for treatment in (0, 1):
        samples = sampler(x, treatment, n_mc, seed + treatment).reshape(x.shape[0], n_mc)
        means.append(samples.mean(axis=1))
    return np.stack(means, axis=1).astype(np.float64)


def ihdp_extended_w1(
    dgp: IHDPDistDGP,
    sampler: Sampler,
    x_eval: np.ndarray,
    n_per_state: int,
    seed: int,
) -> float:
    total = 0.0
    count = 0
    for row_idx in range(x_eval.shape[0]):
        x_row = x_eval[row_idx : row_idx + 1]
        for treatment in (0, 1):
            state_seed = seed + 1000 * row_idx + 37 * treatment
            y_true = dgp.sample_potential(x_row, treatment, n_per_x=n_per_state, seed=state_seed)[0]
            y_learned = sampler(x_row, treatment, n_per_state, state_seed + 13)[0]
            total += sample_wasserstein_1_1d(y_true, y_learned)
            count += 1
    return float(total / count)


def _empty_distribution_accumulators() -> dict[str, object]:
    return {
        "state_count": 0,
        "extended_w1": 0.0,
        "crps": 0.0,
        "energy_distance": 0.0,
        "mmd2": 0.0,
        "ks": 0.0,
        "cvm": 0.0,
        "quantile_sq_sum": 0.0,
        "tail_error": 0.0,
        "coverage": {float(c): 0.0 for c in COVERAGE_LEVELS},
        "interval_width": {float(c): 0.0 for c in COVERAGE_LEVELS},
        "pit": np.zeros(PIT_BINS, dtype=np.float64),
    }


def _accumulate_distribution_state(acc: dict[str, object], predicted: np.ndarray, truth: np.ndarray) -> None:
    predicted_1d = np.asarray(predicted, dtype=np.float64).reshape(-1)
    truth_1d = np.asarray(truth, dtype=np.float64).reshape(-1)
    acc["extended_w1"] = float(acc["extended_w1"]) + sample_wasserstein_1_1d(truth_1d, predicted_1d)
    acc["crps"] = float(acc["crps"]) + crps_empirical_1d(predicted_1d, truth_1d)
    acc["energy_distance"] = float(acc["energy_distance"]) + energy_distance_1d(predicted_1d, truth_1d)
    acc["mmd2"] = float(acc["mmd2"]) + mmd2_gaussian_median_1d(predicted_1d, truth_1d)
    acc["ks"] = float(acc["ks"]) + ks_2samp_1d(predicted_1d, truth_1d)
    acc["cvm"] = float(acc["cvm"]) + cvm_2samp_1d(predicted_1d, truth_1d)
    acc["quantile_sq_sum"] = float(acc["quantile_sq_sum"]) + quantile_squared_error_sum_1d(
        predicted_1d,
        truth_1d,
        ADDITIONAL_QUANTILES,
    )
    acc["tail_error"] = float(acc["tail_error"]) + tail_mean_error_1d(predicted_1d, truth_1d, TAIL_LEVELS)
    coverage, width = central_interval_coverage_width_1d(predicted_1d, truth_1d, COVERAGE_LEVELS)
    coverage_acc = acc["coverage"]
    width_acc = acc["interval_width"]
    if not isinstance(coverage_acc, dict) or not isinstance(width_acc, dict):
        raise TypeError("invalid distribution accumulator")
    for coverage_level in COVERAGE_LEVELS:
        key = float(coverage_level)
        coverage_acc[key] = float(coverage_acc[key]) + coverage[key]
        width_acc[key] = float(width_acc[key]) + width[key]
    pit_acc = acc["pit"]
    if not isinstance(pit_acc, np.ndarray):
        raise TypeError("invalid PIT accumulator")
    pit_acc += pit_histogram_1d(predicted_1d, truth_1d, bins=PIT_BINS)
    acc["state_count"] = int(acc["state_count"]) + 1


def _finalize_distribution_accumulators(acc: dict[str, object]) -> dict[str, float]:
    count = int(acc["state_count"])
    if count <= 0:
        return {}
    metrics = {
        "extended_w1": float(acc["extended_w1"]) / count,
        "crps": float(acc["crps"]) / count,
        "energy_distance": float(acc["energy_distance"]) / count,
        "mmd2": float(acc["mmd2"]) / count,
        "ks": float(acc["ks"]) / count,
        "cvm": float(acc["cvm"]) / count,
        "iqe": float(np.sqrt(float(acc["quantile_sq_sum"]) / (count * ADDITIONAL_QUANTILES.size))),
        "tail_error": float(acc["tail_error"]) / count,
    }
    coverage_acc = acc["coverage"]
    width_acc = acc["interval_width"]
    if not isinstance(coverage_acc, dict) or not isinstance(width_acc, dict):
        raise TypeError("invalid distribution accumulator")
    coverage_errors = []
    for coverage_level in COVERAGE_LEVELS:
        key = float(coverage_level)
        suffix = _coverage_suffix(key)
        empirical = float(coverage_acc[key]) / count
        metrics[f"coverage_{suffix}"] = empirical
        metrics[f"interval_width_{suffix}"] = float(width_acc[key]) / count
        coverage_errors.append(abs(empirical - key))
    metrics["calibration_error"] = float(np.mean(coverage_errors))
    pit_acc = acc["pit"]
    if not isinstance(pit_acc, np.ndarray):
        raise TypeError("invalid PIT accumulator")
    pit = pit_acc / count
    for idx, value in enumerate(pit):
        metrics[_pit_key(idx)] = float(value)
    return metrics


def ihdp_distribution_metrics(
    dgp: IHDPDistDGP,
    sampler: Sampler,
    x_eval: np.ndarray,
    n_per_state: int,
    seed: int,
) -> dict[str, float]:
    acc = _empty_distribution_accumulators()
    true_quantiles = np.zeros((x_eval.shape[0], 2, ADDITIONAL_QUANTILES.size), dtype=np.float64)
    pred_quantiles = np.zeros_like(true_quantiles)
    for row_idx in range(x_eval.shape[0]):
        x_row = x_eval[row_idx : row_idx + 1]
        for treatment in (0, 1):
            state_seed = seed + 1000 * row_idx + 37 * treatment
            y_true = dgp.sample_potential(x_row, treatment, n_per_x=n_per_state, seed=state_seed)[0]
            y_learned = sampler(x_row, treatment, n_per_state, state_seed + 13)[0]
            _accumulate_distribution_state(acc, y_learned, y_true)
            true_quantiles[row_idx, treatment] = np.quantile(
                y_true.reshape(-1),
                ADDITIONAL_QUANTILES,
                method="linear",
            )
            pred_quantiles[row_idx, treatment] = np.quantile(
                y_learned.reshape(-1),
                ADDITIONAL_QUANTILES,
                method="linear",
            )
    metrics = _finalize_distribution_accumulators(acc)
    qte_diff = (pred_quantiles[:, 1] - pred_quantiles[:, 0]) - (true_quantiles[:, 1] - true_quantiles[:, 0])
    metrics["qte_error"] = float(np.sqrt(np.mean(qte_diff**2)))
    return metrics


def ihdp_qte_curve(
    dgp: IHDPDistDGP,
    sampler: Sampler,
    x_eval: np.ndarray,
    quantiles: np.ndarray,
    n_mc: int,
    seed: int,
) -> np.ndarray:
    treated = sampler(x_eval, 1, n_mc, seed + 1).reshape(x_eval.shape[0], n_mc)
    control = sampler(x_eval, 0, n_mc, seed + 2).reshape(x_eval.shape[0], n_mc)
    q1 = np.quantile(treated, quantiles, axis=1, method="linear")
    q0 = np.quantile(control, quantiles, axis=1, method="linear")
    return (q1 - q0).mean(axis=1).astype(np.float64)


def ihdp_true_qte_curve(
    dgp: IHDPDistDGP,
    x_eval: np.ndarray,
    quantiles: np.ndarray,
    n_mc: int,
    seed: int,
) -> np.ndarray:
    y1 = dgp.sample_potential(x_eval, 1, n_per_x=n_mc, seed=seed + 1).reshape(x_eval.shape[0], n_mc)
    y0 = dgp.sample_potential(x_eval, 0, n_per_x=n_mc, seed=seed + 2).reshape(x_eval.shape[0], n_mc)
    q1 = np.quantile(y1, quantiles, axis=1, method="linear")
    q0 = np.quantile(y0, quantiles, axis=1, method="linear")
    return (q1 - q0).mean(axis=1).astype(np.float64)


def summarize_metric_repetitions(repetitions: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    methods = sorted({method for result in repetitions for method in result})
    summary: dict[str, dict[str, float]] = {}
    for method in methods:
        metric_names = sorted({metric for result in repetitions if method in result for metric in result[method]})
        summary[method] = {}
        for metric in metric_names:
            values = np.asarray(
                [result[method][metric] for result in repetitions if method in result and metric in result[method]],
                dtype=np.float64,
            )
            summary[method][metric] = float(values.mean())
            summary[method][f"{metric}_se"] = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
    return summary


def write_ihdp_tables(output_dir: Path, summary: dict[str, dict[str, float]]) -> None:
    ordered_methods = ["ganite", "po_flow", "diff_po", "infs", "dr_learner", "ganice"]
    best_ew = min(ordered_methods, key=lambda method: summary[method]["extended_w1"])
    best_pehe = min(ordered_methods, key=lambda method: summary[method]["sqrt_pehe"])
    csv_lines = [
        "method,extended_w1,extended_w1_se,sqrt_pehe,sqrt_pehe_se,ate_abs_error,ate_abs_error_se,qte_abs_error,qte_abs_error_se"
    ]
    tex_lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Method & eW$_1$ & $\\sqrt{\\epsilon_{\\mathrm{PEHE}}}$ \\\\",
        "\\midrule",
    ]
    for method in ordered_methods:
        metrics = summary[method]
        csv_lines.append(
            ",".join(
                [
                    LABELS[method],
                    f"{metrics['extended_w1']:.6f}",
                    f"{metrics['extended_w1_se']:.6f}",
                    f"{metrics['sqrt_pehe']:.6f}",
                    f"{metrics['sqrt_pehe_se']:.6f}",
                    f"{metrics['ate_abs_error']:.6f}",
                    f"{metrics['ate_abs_error_se']:.6f}",
                    f"{metrics['qte_abs_error']:.6f}",
                    f"{metrics['qte_abs_error_se']:.6f}",
                ]
            )
        )
        label = LABELS[method]
        ew = f"{metrics['extended_w1']:.3f} $\\pm$ {metrics['extended_w1_se']:.3f}"
        pehe = f"{metrics['sqrt_pehe']:.3f} $\\pm$ {metrics['sqrt_pehe_se']:.3f}"
        if method == best_ew:
            ew = f"\\textbf{{{metrics['extended_w1']:.3f}}} $\\pm$ {metrics['extended_w1_se']:.3f}"
        if method == best_pehe:
            pehe = f"\\textbf{{{metrics['sqrt_pehe']:.3f}}} $\\pm$ {metrics['sqrt_pehe_se']:.3f}"
        tex_lines.append(f"{label} & {ew} & {pehe} \\\\")
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output_dir / "ihdp_dist_table.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    (output_dir / "ihdp_dist_table.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def write_ihdp_metric_bar_outputs(output_dir: Path, summary: dict[str, dict[str, float]]) -> None:
    methods = [method for method in ["ganite", "po_flow", "diff_po", "infs", "dr_learner", "ganice"] if method in summary]
    if not methods:
        return
    save_metric_bar_plot(
        output_dir / "ihdp_extended_w1_bar",
        [LABELS[method] for method in methods],
        [summary[method]["extended_w1"] for method in methods],
        [summary[method].get("extended_w1_se", 0.0) for method in methods],
        [COLORS.get(method, "#999999") for method in methods],
        title="IHDP-Dist: distributional error",
        ylabel="empirical eW1",
    )
    save_metric_bar_plot(
        output_dir / "ihdp_pehe_bar",
        [LABELS[method] for method in methods],
        [summary[method]["sqrt_pehe"] for method in methods],
        [summary[method].get("sqrt_pehe_se", 0.0) for method in methods],
        [COLORS.get(method, "#999999") for method in methods],
        title="IHDP-Dist: PEHE",
        ylabel="sqrt PEHE",
    )


def write_ihdp_qte_error_outputs(output_dir: Path, qte_repetitions: list[dict[str, object]]) -> None:
    if not qte_repetitions:
        return
    quantiles = np.asarray(qte_repetitions[0]["quantiles"], dtype=np.float64)
    methods = [method for method in IHDP_QTE_METHODS if method in qte_repetitions[0]["errors"]]

    csv_lines = ["quantile,method,absolute_qte_error,absolute_qte_error_se"]
    plt.figure(figsize=(10.6, 6.2))
    for method in methods:
        stacked = np.asarray(
            [rep["errors"][method] for rep in qte_repetitions if method in rep["errors"]],
            dtype=np.float64,
        )
        mean_error = stacked.mean(axis=0)
        se_error = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0]) if stacked.shape[0] > 1 else np.zeros_like(mean_error)
        for quantile, mean_value, se_value in zip(quantiles, mean_error, se_error, strict=True):
            csv_lines.append(f"{quantile:.6f},{LABELS[method]},{mean_value:.8f},{se_value:.8f}")
        linewidth = 3.0 if method == "ganice" else 2.0
        linestyle = "-" if method == "ganice" else "--"
        plt.plot(
            quantiles,
            mean_error,
            label=LABELS[method],
            color=COLORS[method],
            linewidth=linewidth,
            linestyle=linestyle,
        )
        if stacked.shape[0] > 1:
            plt.fill_between(
                quantiles,
                np.maximum(0.0, mean_error - se_error),
                mean_error + se_error,
                color=COLORS[method],
                alpha=0.16,
                linewidth=0,
            )
    plt.xlabel("quantile level")
    plt.ylabel("|estimated QTE - true QTE|")
    plt.ylim(bottom=0.0)
    place_legend_outside()
    plt.tight_layout()
    plt.savefig(output_dir / "ihdp_qte_curve.png", dpi=260)
    plt.savefig(output_dir / "ihdp_qte_curve.pdf")
    plt.savefig(output_dir / "ihdp_qte_error.png", dpi=260)
    plt.savefig(output_dir / "ihdp_qte_error.pdf")
    plt.close()
    (output_dir / "ihdp_qte_error.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")


def write_ihdp_ablation_output(output_dir: Path, summary: dict[str, dict[str, float]]) -> None:
    ablation_methods = [method for method in ["ganice", "ganice_no_cell_norm", "ganice_pooled", "ganite"] if method in summary]
    if not ablation_methods:
        return
    plt.figure(figsize=(10.2, 6.1))
    values = [summary[method]["extended_w1"] for method in ablation_methods]
    errors = [summary[method].get("extended_w1_se", 0.0) for method in ablation_methods]
    labels = [LABELS[method] for method in ablation_methods]
    colors = [COLORS.get(method, "#999999") for method in ablation_methods]
    plt.bar(
        np.arange(len(labels)),
        values,
        yerr=errors,
        capsize=5,
        color=colors,
        error_kw={"elinewidth": 1.7, "capthick": 1.7},
    )
    plt.xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    plt.ylabel("empirical eW1")
    plt.tight_layout()
    plt.savefig(output_dir / "ihdp_objective_ablation.png", dpi=260)
    plt.savefig(output_dir / "ihdp_objective_ablation.pdf")
    plt.close()


def write_ihdp_additional_outputs(output_dir: Path, summary: dict[str, dict[str, float]]) -> None:
    metrics = [
        ("crps", "CRPS", "min"),
        ("energy_distance", "ED", "min"),
        ("mmd2", "MMD$^2$", "min"),
        ("ks", "KS", "min"),
        ("cvm", "CvM", "min"),
        ("iqe", "IQE", "min"),
        ("qte_error", "QTEErr", "min"),
        ("tail_error", "TailErr", "min"),
        ("calibration_error", "CalErr", "min"),
        ("pehe", "PEHE", "min"),
        ("ate_abs_error", "ATEErr", "min"),
    ]
    _write_additional_metric_table(output_dir, "ihdp_additional_metrics", summary, DEFAULT_METHODS, metrics)
    save_metric_grid_bar_plot(
        output_dir / "ihdp_additional_metrics_bar",
        summary,
        DEFAULT_METHODS,
        [(key, label.replace("$", "")) for key, label, _ in metrics[:9]],
    )
    save_interval_width_plot(
        output_dir / "ihdp_interval_widths",
        summary,
        DEFAULT_METHODS,
        ylabel="average interval width",
    )
    save_pit_histogram_plot(output_dir / "ihdp_pit_histograms", summary, DEFAULT_METHODS)
    density_metrics = [("density_nll", "NLL", "min")]
    _write_additional_metric_table(
        output_dir,
        "ihdp_density_diagnostics",
        summary,
        ["po_flow", "infs"],
        density_metrics,
    )


def run_single_ihdp_dist(output_dir: Path, seed: int, quick: bool) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    dgp = IHDPDistDGP(seed=seed + 1, split_seed=seed + 2)
    train = dgp.observed_split("train", seed=seed + 3)
    validation_x, _ = dgp.split("validation")
    test_x, test_t = dgp.split("test")
    test_y = dgp.sample_potential(test_x, test_t, n_per_x=1, seed=seed + 70_000).reshape(-1, 1)
    true_mu = dgp.potential_means(test_x).astype(np.float64)
    true_tau = true_mu[:, 1] - true_mu[:, 0]
    outcome_lower, outcome_upper = dgp.outcome_bounds(train["y"])
    eval_mc = 96 if quick else 192
    mean_mc = 96 if quick else 192
    qte_mc = 192 if quick else 512

    ganite = GANITE(ihdp_ganite_config(seed + 11, dgp.feature_dim, outcome_lower, outcome_upper, quick=quick))
    ganite.fit(train["x"], train["t"], train["y"])
    ganite_sampler = make_degenerate_mean_sampler(
        ganite.predict_potential_outcomes,
        outcome_lower,
        outcome_upper,
    )

    po_flow = POFlow(ihdp_po_flow_config(seed + 23, dgp.feature_dim, outcome_lower, outcome_upper, quick=quick))
    po_flow.fit(train["x"], train["t"], train["y"])

    diff_po = DiffPO(ihdp_diff_po_config(seed + 29, dgp.feature_dim, outcome_lower, outcome_upper, quick=quick))
    diff_po.fit(train["x"], train["t"], train["y"])

    infs = INFs(ihdp_infs_config(seed + 31, dgp.feature_dim, outcome_lower, outcome_upper, quick=quick))
    infs.fit(train["x"], train["t"], train["y"])

    dr_learner = DRLearner(ihdp_dr_learner_config(seed + 37, dgp.feature_dim, outcome_lower, outcome_upper, quick=quick))
    dr_learner.fit(train["x"], train["t"], train["y"])
    dr_sampler = make_residual_plugin_sampler(
        dr_learner.predict_potential_outcomes,
        train["x"],
        train["t"],
        train["y"],
        outcome_lower,
        outcome_upper,
    )

    ganice_cell_transform = fit_pca_treatment_cell_transform(train["x"], n_components=2)
    ganice_cell_resolution = (1, 1, 1)
    ganice_cell_dim = 3
    ganice_candidates: list[tuple[float, GANICE]] = []
    restart_count = 2 if quick else 4
    validation_mc = 64 if quick else 96
    ganice_candidate_specs = [
        {"generator_transport_weight": 0.75, "factual_crps_weight": 8.0, "factual_mse_weight": 2.0},
        {"generator_transport_weight": 1.25, "factual_crps_weight": 8.0, "factual_mse_weight": 2.0},
        {"generator_transport_weight": 0.75, "factual_crps_weight": 12.0, "factual_mse_weight": 1.0},
        {"generator_transport_weight": 1.25, "factual_crps_weight": 12.0, "factual_mse_weight": 1.0},
    ]
    for restart in range(restart_count):
        candidate_spec = ganice_candidate_specs[restart % len(ganice_candidate_specs)]
        candidate = fit_ganice(
            d_w=dgp.d_w,
            train_w=train["w"],
            train_y=train["y"],
            target_w_sampler=dgp.sample_target_w,
            config=ihdp_ganice_config(
                seed + 41 + 100 * restart,
                dgp.feature_dim,
                outcome_lower,
                outcome_upper,
                quick=quick,
                cell_resolution=ganice_cell_resolution,
                min_cell_samples=6,
                shared_generator=True,
                **candidate_spec,
            ),
            seed=seed + 43 + 100 * restart,
            d_cell_w=ganice_cell_dim,
            cell_transform=ganice_cell_transform,
        )
        candidate_sampler = make_ganice_binary_sampler(candidate, dgp)
        validation_score = ihdp_extended_w1(
            dgp,
            candidate_sampler,
            validation_x,
            validation_mc,
            seed + 40_000 + 1_000 * restart,
        )
        ganice_candidates.append((validation_score, candidate))
    ganice_validation_score, ganice = min(ganice_candidates, key=lambda item: item[0])
    ganice_no_cell_norm = fit_ganice(
        d_w=dgp.d_w,
        train_w=train["w"],
        train_y=train["y"],
        target_w_sampler=dgp.sample_target_w,
        config=ihdp_ganice_config(
            seed + 47,
            dgp.feature_dim,
            outcome_lower,
            outcome_upper,
            quick=quick,
            cell_normalized=False,
            cell_resolution=ganice_cell_resolution,
            min_cell_samples=6,
            shared_generator=True,
        ),
        seed=seed + 53,
        d_cell_w=ganice_cell_dim,
        cell_transform=ganice_cell_transform,
    )
    ganice_pooled = fit_ganice(
        d_w=dgp.d_w,
        train_w=train["w"],
        train_y=train["y"],
        target_w_sampler=dgp.sample_target_w,
        config=ihdp_ganice_config(
            seed + 59,
            dgp.feature_dim,
            outcome_lower,
            outcome_upper,
            quick=quick,
            pooled=True,
        ),
        seed=seed + 61,
    )

    samplers: dict[str, Sampler] = {
        "ganite": ganite_sampler,
        "po_flow": lambda x, treatment, n, sample_seed: po_flow.sample_potential(
            x,
            treatment,
            n_per_x=n,
            seed=sample_seed,
        ),
        "diff_po": lambda x, treatment, n, sample_seed: diff_po.sample_potential(
            x,
            treatment,
            n_per_x=n,
            seed=sample_seed,
        ),
        "infs": lambda x, treatment, n, sample_seed: infs.sample_potential(
            x,
            treatment,
            n_per_x=n,
            seed=sample_seed,
        ),
        "dr_learner": dr_sampler,
        "ganice": make_ganice_binary_sampler(ganice, dgp),
        "ganice_no_cell_norm": make_ganice_binary_sampler(ganice_no_cell_norm, dgp),
        "ganice_pooled": make_ganice_binary_sampler(ganice_pooled, dgp),
    }

    results: dict[str, dict[str, float]] = {}
    for method, sampler in samplers.items():
        pred_mu = estimate_binary_means_from_sampler(sampler, test_x, mean_mc, seed + 10_000)
        pred_tau = pred_mu[:, 1] - pred_mu[:, 0]
        method_metrics = ihdp_distribution_metrics(dgp, sampler, test_x, eval_mc, seed + 20_000)
        pehe = float(np.mean((pred_tau - true_tau) ** 2))
        method_metrics.update(
            {
                "pehe": pehe,
                "sqrt_pehe": float(np.sqrt(pehe)),
                "ate_abs_error": float(abs(np.mean(pred_tau) - np.mean(true_tau))),
            }
        )
        results[method] = method_metrics
    results["po_flow"]["density_nll"] = po_flow.negative_log_likelihood(test_x, test_t, test_y)
    results["infs"]["density_nll"] = infs.negative_log_likelihood(test_x, test_t, test_y)
    results["ganice"]["target_mass_coverage"] = float(ganice.target_mass_coverage)
    results["ganice"]["validation_extended_w1"] = float(ganice_validation_score)
    results["ganice_no_cell_norm"]["target_mass_coverage"] = float(ganice_no_cell_norm.target_mass_coverage)
    results["ganice_pooled"]["target_mass_coverage"] = float(ganice_pooled.target_mass_coverage)

    quantiles = np.linspace(0.05, 0.95, 19, dtype=np.float64)
    true_qte_mc = 1_024 if quick else 4_096
    qte_curves = {
        "true": ihdp_true_qte_curve(dgp, test_x, quantiles, true_qte_mc, seed + 30_000),
        "ganite": ihdp_qte_curve(dgp, samplers["ganite"], test_x, quantiles, qte_mc, seed + 31_000),
        "po_flow": ihdp_qte_curve(dgp, samplers["po_flow"], test_x, quantiles, qte_mc, seed + 32_000),
        "diff_po": ihdp_qte_curve(dgp, samplers["diff_po"], test_x, quantiles, qte_mc, seed + 33_000),
        "infs": ihdp_qte_curve(dgp, samplers["infs"], test_x, quantiles, qte_mc, seed + 34_000),
        "dr_learner": ihdp_qte_curve(dgp, samplers["dr_learner"], test_x, quantiles, qte_mc, seed + 35_000),
        "ganice": ihdp_qte_curve(dgp, samplers["ganice"], test_x, quantiles, qte_mc, seed + 36_000),
    }
    qte_errors = {
        method: np.abs(qte_curves[method] - qte_curves["true"]).astype(np.float64)
        for method in IHDP_QTE_METHODS
    }
    for method, error_curve in qte_errors.items():
        results[method]["qte_abs_error"] = float(error_curve.mean())

    ablation_methods = ["ganice", "ganice_no_cell_norm", "ganice_pooled", "ganite"]
    plt.figure(figsize=(10.2, 6.1))
    values = [results[method]["extended_w1"] for method in ablation_methods]
    labels = [LABELS[method] for method in ablation_methods]
    colors = [COLORS.get(method, "#999999") for method in ablation_methods]
    plt.bar(np.arange(len(labels)), values, color=colors)
    plt.xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    plt.ylabel("empirical eW1")
    plt.tight_layout()
    plt.savefig(output_dir / "ihdp_objective_ablation.png", dpi=260)
    plt.savefig(output_dir / "ihdp_objective_ablation.pdf")
    plt.close()
    return results, {
        "quantiles": quantiles.tolist(),
        "errors": {method: error_curve.tolist() for method, error_curve in qte_errors.items()},
    }


def _run_ihdp_repetition(task: tuple[Path, int, int, bool]) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    output_dir, seed, rep, quick = task
    rep_dir = ensure_dir(output_dir / "replications" / f"ihdp_rep_{rep:03d}")
    return run_single_ihdp_dist(rep_dir, seed + 10_000 * rep, quick=quick)


def run_ihdp_dist_benchmark(
    output_dir: Path,
    seed: int,
    repetitions: int,
    quick: bool,
    parallel: int = 1,
) -> dict[str, dict[str, float]]:
    tasks = [(output_dir, seed, rep, quick) for rep in range(repetitions)]
    rep_outputs = _run_parallel_repetitions(_run_ihdp_repetition, tasks, parallel)
    per_repetition = [result for result, _ in rep_outputs]
    qte_repetitions = [qte_errors for _, qte_errors in rep_outputs]
    summary = summarize_metric_repetitions(per_repetition)
    write_ihdp_tables(output_dir, summary)
    write_ihdp_metric_bar_outputs(output_dir, summary)
    write_ihdp_qte_error_outputs(output_dir, qte_repetitions)
    write_ihdp_ablation_output(output_dir, summary)
    write_ihdp_additional_outputs(output_dir, summary)
    (output_dir / "ihdp_dist_repetitions.json").write_text(
        json.dumps(per_repetition, indent=2),
        encoding="utf-8",
    )
    (output_dir / "ihdp_qte_error_repetitions.json").write_text(
        json.dumps(qte_repetitions, indent=2),
        encoding="utf-8",
    )
    return summary


def jobs_ganice_config(
    seed: int,
    x_dim: int,
    outcome_lower: float,
    outcome_upper: float,
    *,
    quick: bool,
    cell_resolution: tuple[int, ...] = (1, 1, 1),
    min_cell_samples: int = 4,
) -> GANICEConfig:
    return GANICEConfig(
        latent_dim=4,
        hidden_dims_generator=(64, 64) if quick else (96, 96),
        hidden_dims_critic=(64, 64) if quick else (96, 96),
        batch_size=128,
        num_steps=_maybe_quick(520, quick),
        critic_steps=2 if quick else 3,
        generator_lr=2e-4,
        critic_lr=1e-4,
        resolution=cell_resolution,
        min_cell_samples=min_cell_samples,
        target_mass_samples=12_000 if quick else 40_000,
        cell_normalized=True,
        critic_uses_w=False,
        factual_crps_weight=5.0,
        factual_crps_samples=6,
        factual_mse_weight=0.0,
        factual_mse_samples=6,
        generator_transport_weight=8.0,
        residual_quantile_calibration=False,
        calibration_samples_per_observation=8 if quick else 12,
        calibration_grid_size=192 if quick else 256,
        calibration_blend=0.85,
        pretrain_steps=_maybe_quick(420, quick),
        pretrain_mse_weight=1.5,
        shared_generator=True,
        outcome_lower=outcome_lower,
        outcome_upper=outcome_upper,
        seed=seed,
    )


def jobs_ganite_config(seed: int, x_dim: int, outcome_lower: float, outcome_upper: float, *, quick: bool) -> GANITEConfig:
    return GANITEConfig(
        x_dim=x_dim,
        hidden_dim=64 if quick else 128,
        batch_size=128,
        cf_iterations=_maybe_quick(650, quick),
        ite_iterations=_maybe_quick(650, quick),
        cf_discriminator_steps=1 if quick else 2,
        alpha=1.0,
        beta=5.0,
        outcome_min=outcome_lower,
        outcome_max=outcome_upper,
        seed=seed,
    )


def jobs_po_flow_config(seed: int, x_dim: int, outcome_lower: float, outcome_upper: float, *, quick: bool) -> POFlowConfig:
    return POFlowConfig(
        x_dim=x_dim,
        num_treatments=2,
        hidden_dim=64,
        batch_size=128,
        num_steps=_maybe_quick(850, quick),
        learning_rate=1e-3,
        rk4_steps=12 if quick else 18,
        outcome_min=outcome_lower,
        outcome_max=outcome_upper,
        seed=seed,
    )


def jobs_diff_po_config(seed: int, x_dim: int, outcome_lower: float, outcome_upper: float, *, quick: bool) -> DiffPOConfig:
    return DiffPOConfig(
        x_dim=x_dim,
        num_treatments=2,
        hidden_dim=64,
        time_embedding_dim=96 if quick else 128,
        residual_blocks=3,
        batch_size=128,
        propensity_steps=_maybe_quick(260, quick),
        diffusion_steps=_maybe_quick(520, quick),
        num_diffusion_steps=32 if quick else 55,
        learning_rate=5e-4,
        propensity_learning_rate=1e-3,
        outcome_min=outcome_lower,
        outcome_max=outcome_upper,
        seed=seed,
    )


def jobs_infs_config(seed: int, x_dim: int, outcome_lower: float, outcome_upper: float, *, quick: bool) -> INFsConfig:
    return INFsConfig(
        x_dim=x_dim,
        num_treatments=2,
        hidden_dim=64,
        num_bins=48 if quick else 72,
        batch_size=128,
        nuisance_steps=_maybe_quick(700, quick),
        target_steps=_maybe_quick(700, quick),
        nuisance_lr=1e-3,
        target_lr=4e-3,
        prop_alpha=1.0,
        clip_propensity=0.05,
        noise_std_x=0.0,
        noise_std_y=0.01,
        outcome_min=outcome_lower,
        outcome_max=outcome_upper,
        seed=seed,
    )


def jobs_dr_learner_config(seed: int, x_dim: int, outcome_lower: float, outcome_upper: float, *, quick: bool) -> DRLearnerConfig:
    return DRLearnerConfig(
        x_dim=x_dim,
        num_treatments=2,
        hidden_dim=64,
        batch_size=128,
        nuisance_steps=_maybe_quick(650, quick),
        final_steps=_maybe_quick(800, quick),
        n_folds=2,
        nuisance_lr=1e-3,
        final_lr=1e-3,
        propensity_clip=0.05,
        outcome_min=outcome_lower,
        outcome_max=outcome_upper,
        seed=seed,
    )


def make_ganice_jobs_sampler(
    model: GANICE,
    data: JobsLaLondeData,
    zero_masses: dict[int, float] | None = None,
    arm_quantile_calibrators: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> Sampler:
    def sampler(x: np.ndarray, treatment: int, n_per_x: int, seed: int | None) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
        rng = np.random.default_rng(seed)
        samples = []
        for row_idx in range(x_arr.shape[0]):
            w = data.encode_w(x_arr[row_idx : row_idx + 1], treatment)
            row_samples = model.sample_conditional(
                w,
                n_per_x,
                seed=None if seed is None else seed + row_idx,
            ).reshape(1, n_per_x, 1)
            if zero_masses is not None:
                zero_mass = float(np.clip(zero_masses.get(int(treatment), 0.0), 0.0, 0.95))
                if zero_mass > 0.0:
                    row_mask = rng.random(n_per_x) < zero_mass
                    row_samples[0, row_mask, 0] = 0.0
            if arm_quantile_calibrators is not None and int(treatment) in arm_quantile_calibrators:
                source, target = arm_quantile_calibrators[int(treatment)]
                values = row_samples.reshape(-1)
                row_samples = np.interp(values, source, target, left=target[0], right=target[-1]).reshape(
                    1,
                    n_per_x,
                    1,
                )
            samples.append(row_samples)
        return np.concatenate(samples, axis=0).astype(np.float32)

    return sampler


def fit_jobs_arm_quantile_calibrators(
    sampler: Sampler,
    rct: dict[str, np.ndarray],
    *,
    n_per_x: int,
    seed: int,
    grid_size: int = 256,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    calibrators: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    probs = (np.arange(grid_size, dtype=np.float64) + 0.5) / grid_size
    for arm in (0, 1):
        generated = jobs_arm_samples(sampler, rct["x"], arm, n_per_x, seed + 10_000 * arm)
        observed = np.asarray(rct["y"][rct["t"] == arm], dtype=np.float64).reshape(-1)
        if generated.size < 3 or observed.size < 3:
            continue
        source = np.quantile(generated, probs)
        target = np.quantile(observed, probs)
        source_unique, unique_idx = np.unique(source, return_index=True)
        if source_unique.size >= 3:
            calibrators[arm] = (source_unique.astype(np.float32), target[unique_idx].astype(np.float32))
    return calibrators


def jobs_nsw_zero_masses(data: JobsLaLondeData, split: str = "train") -> dict[int, float]:
    rct = data.rct_split(split)
    return {
        arm: float(np.mean(rct["y_earnings"][rct["t"] == arm] <= 0.0))
        for arm in (0, 1)
    }


def jobs_arm_samples(
    sampler: Sampler,
    x_rct: np.ndarray,
    treatment: int,
    n_per_x: int,
    seed: int,
) -> np.ndarray:
    return sampler(x_rct, treatment, n_per_x, seed + 100 * treatment).reshape(-1)


def jobs_estimate_earnings_means(
    sampler: Sampler,
    x_rct: np.ndarray,
    n_mc: int,
    seed: int,
) -> np.ndarray:
    means = []
    for treatment in (0, 1):
        transformed = sampler(x_rct, treatment, n_mc, seed + treatment).reshape(x_rct.shape[0], n_mc)
        earnings = model_scale_to_earnings(transformed)
        means.append(earnings.mean(axis=1))
    return np.stack(means, axis=1).astype(np.float64)


def jobs_additional_metrics(
    sampler: Sampler,
    rct: dict[str, np.ndarray],
    arm_samples_earnings: dict[int, np.ndarray],
    n_per_x: int,
    seed: int,
) -> dict[str, float]:
    treatment = np.asarray(rct["t"], dtype=np.int64).reshape(-1)
    true_y = np.asarray(rct["y"], dtype=np.float64).reshape(-1)
    true_earnings = np.asarray(rct["y_earnings"], dtype=np.float64).reshape(-1)

    arm_acc = _empty_distribution_accumulators()
    pred_quantiles: dict[int, np.ndarray] = {}
    true_quantiles: dict[int, np.ndarray] = {}
    for arm in (0, 1):
        generated = np.asarray(arm_samples_earnings[arm], dtype=np.float64).reshape(-1)
        observed = true_earnings[treatment == arm]
        _accumulate_distribution_state(arm_acc, generated, observed)
        pred_quantiles[arm] = np.quantile(generated, ADDITIONAL_QUANTILES, method="linear")
        true_quantiles[arm] = np.quantile(observed, ADDITIONAL_QUANTILES, method="linear")
    arm_metrics = _finalize_distribution_accumulators(arm_acc)
    metrics = {
        "energy_distance": arm_metrics["energy_distance"],
        "mmd2": arm_metrics["mmd2"],
        "ks": arm_metrics["ks"],
        "cvm": arm_metrics["cvm"],
        "iqe": arm_metrics["iqe"],
        "tail_error": arm_metrics["tail_error"],
        "arm_crps_earnings": arm_metrics["crps"],
        "arm_w1_earnings": arm_metrics["extended_w1"],
        "qte_error": float(
            np.sqrt(
                np.mean(
                    (
                        (pred_quantiles[1] - pred_quantiles[0])
                        - (true_quantiles[1] - true_quantiles[0])
                    )
                    ** 2
                )
            )
        ),
    }

    n = rct["x"].shape[0]
    factual_samples = np.zeros((n, n_per_x), dtype=np.float64)
    for arm in (0, 1):
        mask = treatment == arm
        if np.any(mask):
            factual_samples[mask] = sampler(
                rct["x"][mask],
                arm,
                n_per_x,
                seed + 1000 * arm,
            ).reshape(int(mask.sum()), n_per_x)
    factual_earnings = model_scale_to_earnings(factual_samples)

    crps_values = []
    crps_earnings_values = []
    coverage_sum = {float(c): 0.0 for c in COVERAGE_LEVELS}
    width_sum = {float(c): 0.0 for c in COVERAGE_LEVELS}
    pit_sum = np.zeros(PIT_BINS, dtype=np.float64)
    for i in range(n):
        crps_values.append(crps_empirical_1d(factual_samples[i], true_y[i]))
        crps_earnings_values.append(crps_empirical_1d(factual_earnings[i], true_earnings[i]))
        coverage, width = central_interval_coverage_width_1d(
            factual_earnings[i],
            np.asarray([true_earnings[i]], dtype=np.float64),
            COVERAGE_LEVELS,
        )
        for coverage_level in COVERAGE_LEVELS:
            key = float(coverage_level)
            coverage_sum[key] += coverage[key]
            width_sum[key] += width[key]
        pit_sum += pit_histogram_1d(factual_samples[i], np.asarray([true_y[i]], dtype=np.float64), bins=PIT_BINS)
    metrics["crps"] = float(np.mean(crps_values))
    metrics["factual_crps"] = metrics["crps"]
    metrics["factual_crps_earnings"] = float(np.mean(crps_earnings_values))
    coverage_errors = []
    for coverage_level in COVERAGE_LEVELS:
        key = float(coverage_level)
        suffix = _coverage_suffix(key)
        empirical = coverage_sum[key] / n
        metrics[f"coverage_{suffix}"] = float(empirical)
        metrics[f"interval_width_{suffix}"] = float(width_sum[key] / n)
        coverage_errors.append(abs(empirical - key))
    metrics["calibration_error"] = float(np.mean(coverage_errors))
    pit = pit_sum / n
    for idx, value in enumerate(pit):
        metrics[_pit_key(idx)] = float(value)
    return metrics


def jobs_policy_value(y_earnings: np.ndarray, treatment: np.ndarray, policy: np.ndarray) -> float:
    y = np.asarray(y_earnings, dtype=np.float64).reshape(-1)
    t = np.asarray(treatment, dtype=np.int64).reshape(-1)
    policy_bool = np.asarray(policy, dtype=bool).reshape(-1)
    treat_rate = float(policy_bool.mean())
    if treat_rate <= 0.0:
        control = y[t == 0]
        return float(control.mean())
    if treat_rate >= 1.0:
        treated = y[t == 1]
        return float(treated.mean())
    treated_selected = y[policy_bool & (t == 1)]
    control_selected = y[(~policy_bool) & (t == 0)]
    treated_fallback = y[t == 1]
    control_fallback = y[t == 0]
    mean_treated = float(treated_selected.mean()) if treated_selected.size else float(treated_fallback.mean())
    mean_control = float(control_selected.mean()) if control_selected.size else float(control_fallback.mean())
    return float(treat_rate * mean_treated + (1.0 - treat_rate) * mean_control)


def jobs_evaluate_sampler(
    sampler: Sampler,
    rct: dict[str, np.ndarray],
    true_att_earnings: float,
    n_per_x: int,
    mean_mc: int,
    seed: int,
    *,
    include_additional: bool = True,
) -> tuple[dict[str, float], dict[str, object]]:
    x_rct = rct["x"]
    treatment = rct["t"]
    true_y = rct["y"].reshape(-1)
    true_earnings = rct["y_earnings"].reshape(-1)
    arm_samples_transformed: dict[int, np.ndarray] = {}
    arm_samples_earnings: dict[int, np.ndarray] = {}
    rct_w1 = 0.0
    rct_w1_earnings = 0.0
    for arm in (0, 1):
        generated = jobs_arm_samples(sampler, x_rct, arm, n_per_x, seed + 10_000)
        generated_earnings = model_scale_to_earnings(generated)
        arm_samples_transformed[arm] = generated
        arm_samples_earnings[arm] = generated_earnings
        rct_w1 += sample_wasserstein_1_1d(true_y[treatment == arm], generated)
        rct_w1_earnings += sample_wasserstein_1_1d(true_earnings[treatment == arm], generated_earnings)
    rct_w1 *= 0.5
    rct_w1_earnings *= 0.5

    earnings_means = jobs_estimate_earnings_means(sampler, x_rct, mean_mc, seed + 20_000)
    effects = earnings_means[:, 1] - earnings_means[:, 0]
    treated_mask = treatment == 1
    estimated_att = float(effects[treated_mask].mean()) if np.any(treated_mask) else float(effects.mean())
    policy = effects > 0.0
    policy_value = jobs_policy_value(true_earnings, treatment, policy)
    metrics = {
        "rct_w1": float(rct_w1),
        "rct_w1_earnings": float(rct_w1_earnings),
        "att_abs_error": float(abs(estimated_att - true_att_earnings)),
        "estimated_att": float(estimated_att),
        "policy_value": float(policy_value),
        "policy_risk": float(-policy_value),
        "policy_treat_rate": float(policy.mean()),
    }
    if include_additional:
        metrics.update(
            jobs_additional_metrics(
                sampler,
                rct,
                arm_samples_earnings,
                n_per_x=n_per_x,
                seed=seed + 30_000,
            )
        )
    detail = {
        "arm_samples_transformed": arm_samples_transformed,
        "arm_samples_earnings": arm_samples_earnings,
        "earnings_means": earnings_means,
        "effects": effects,
    }
    return metrics, detail


def jobs_policy_curve(
    y_earnings: np.ndarray,
    treatment: np.ndarray,
    effects: np.ndarray,
    rates: np.ndarray,
) -> np.ndarray:
    order = np.argsort(-np.asarray(effects, dtype=np.float64).reshape(-1))
    values = []
    n = order.size
    for rate in rates:
        k = int(round(float(rate) * n))
        policy = np.zeros(n, dtype=bool)
        if k > 0:
            policy[order[:k]] = True
        values.append(jobs_policy_value(y_earnings, treatment, policy))
    return np.asarray(values, dtype=np.float64)


def summarize_jobs_repetitions(repetitions: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    return summarize_metric_repetitions(repetitions)


def write_jobs_tables(output_dir: Path, summary: dict[str, dict[str, float]]) -> None:
    ordered_methods = ["ganite", "po_flow", "diff_po", "infs", "dr_learner", "ganice"]
    best_w1 = min(ordered_methods, key=lambda method: summary[method]["rct_w1"])
    best_att = min(ordered_methods, key=lambda method: summary[method]["att_abs_error"])
    csv_lines = [
        "method,rct_w1,rct_w1_se,rct_w1_earnings,rct_w1_earnings_se,att_abs_error,att_abs_error_se,policy_value,policy_value_se,policy_treat_rate,policy_treat_rate_se"
    ]
    tex_lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Method & RCT-W$_1$ & ATT error \\\\",
        "\\midrule",
    ]
    for method in ordered_methods:
        metrics = summary[method]
        csv_lines.append(
            ",".join(
                [
                    LABELS[method],
                    f"{metrics['rct_w1']:.6f}",
                    f"{metrics['rct_w1_se']:.6f}",
                    f"{metrics['rct_w1_earnings']:.6f}",
                    f"{metrics['rct_w1_earnings_se']:.6f}",
                    f"{metrics['att_abs_error']:.6f}",
                    f"{metrics['att_abs_error_se']:.6f}",
                    f"{metrics['policy_value']:.6f}",
                    f"{metrics['policy_value_se']:.6f}",
                    f"{metrics['policy_treat_rate']:.6f}",
                    f"{metrics['policy_treat_rate_se']:.6f}",
                ]
            )
        )
        w1 = f"{metrics['rct_w1']:.3f} $\\pm$ {metrics['rct_w1_se']:.3f}"
        att = f"{metrics['att_abs_error']:.0f} $\\pm$ {metrics['att_abs_error_se']:.0f}"
        if method == best_w1:
            w1 = f"\\textbf{{{metrics['rct_w1']:.3f}}} $\\pm$ {metrics['rct_w1_se']:.3f}"
        if method == best_att:
            att = f"\\textbf{{{metrics['att_abs_error']:.0f}}} $\\pm$ {metrics['att_abs_error_se']:.0f}"
        tex_lines.append(f"{LABELS[method]} & {w1} & {att} \\\\")
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output_dir / "jobs_lalonde_table.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    (output_dir / "jobs_lalonde_table.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def write_jobs_metric_bar_outputs(output_dir: Path, summary: dict[str, dict[str, float]]) -> None:
    methods = [method for method in ["ganite", "po_flow", "diff_po", "infs", "dr_learner", "ganice"] if method in summary]
    if not methods:
        return
    labels = [LABELS[method] for method in methods]
    colors = [COLORS.get(method, "#999999") for method in methods]
    save_metric_bar_plot(
        output_dir / "jobs_rct_w1_bar",
        labels,
        [summary[method]["rct_w1"] for method in methods],
        [summary[method].get("rct_w1_se", 0.0) for method in methods],
        colors,
        title="Jobs/LaLonde: RCT-assisted distributional error",
        ylabel="RCT-W1",
    )
    save_metric_bar_plot(
        output_dir / "jobs_att_error_bar",
        labels,
        [summary[method]["att_abs_error"] for method in methods],
        [summary[method].get("att_abs_error_se", 0.0) for method in methods],
        colors,
        title="Jobs/LaLonde: ATT error",
        ylabel="absolute ATT error",
    )


def write_jobs_additional_outputs(output_dir: Path, summary: dict[str, dict[str, float]]) -> None:
    metrics = [
        ("crps", "Factual CRPS", "min"),
        ("factual_crps_earnings", "CRPS earn.", "min"),
        ("energy_distance", "ED", "min"),
        ("mmd2", "MMD$^2$", "min"),
        ("ks", "KS", "min"),
        ("cvm", "CvM", "min"),
        ("iqe", "IQE", "min"),
        ("qte_error", "QTEErr", "min"),
        ("tail_error", "TailErr", "min"),
        ("calibration_error", "CalErr", "min"),
        ("att_abs_error", "ATTErr", "min"),
        ("policy_value", "Policy value", "max"),
    ]
    _write_additional_metric_table(output_dir, "jobs_additional_metrics", summary, DEFAULT_METHODS, metrics)
    save_metric_grid_bar_plot(
        output_dir / "jobs_additional_metrics_bar",
        summary,
        DEFAULT_METHODS,
        [(key, label.replace("$", "")) for key, label, _ in metrics[:10]],
    )
    save_interval_width_plot(
        output_dir / "jobs_interval_widths",
        summary,
        DEFAULT_METHODS,
        ylabel="average factual interval width (USD)",
    )
    save_pit_histogram_plot(output_dir / "jobs_pit_histograms", summary, DEFAULT_METHODS)
    _write_additional_metric_table(
        output_dir,
        "jobs_density_diagnostics",
        summary,
        ["po_flow", "infs"],
        [("density_nll", "NLL", "min")],
    )


def _plot_empirical_cdf(values: np.ndarray, label: str, color: str, linewidth: float, linestyle: str = "-") -> None:
    sorted_values = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    probs = (np.arange(sorted_values.size, dtype=np.float64) + 1.0) / max(sorted_values.size, 1)
    plt.step(sorted_values / 1000.0, probs, where="post", label=label, color=color, linewidth=linewidth, linestyle=linestyle)


def _cdf_on_grid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if sorted_values.size == 0:
        return np.zeros_like(grid, dtype=np.float64)
    return np.searchsorted(sorted_values, grid, side="right").astype(np.float64) / float(sorted_values.size)


def write_jobs_cdf_repetition_outputs(output_dir: Path, detail_repetitions: list[dict[str, object]]) -> None:
    if not detail_repetitions:
        return
    methods = ["ganite", "po_flow", "diff_po", "infs", "dr_learner", "ganice"]
    csv_lines = ["arm,method,earnings_grid,cdf,cdf_se"]
    for arm, arm_name in [(1, "treated"), (0, "control")]:
        pooled_values: list[np.ndarray] = []
        for details in detail_repetitions:
            rct = details["rct"]
            cdf_details = details["cdf_details"]
            pooled_values.append(np.asarray(rct["y_earnings"][rct["t"] == arm], dtype=np.float64))
            for method in methods:
                pooled_values.append(np.asarray(cdf_details[method]["arm_samples_earnings"][arm], dtype=np.float64))
        x_max = float(np.quantile(np.concatenate(pooled_values), 0.985))
        grid = np.linspace(0.0, max(5_000.0, x_max), 240, dtype=np.float64)

        plt.figure(figsize=(10.8, 6.3))
        true_stack = np.stack(
            [
                _cdf_on_grid(np.asarray(details["rct"]["y_earnings"][details["rct"]["t"] == arm], dtype=np.float64), grid)
                for details in detail_repetitions
            ],
            axis=0,
        )
        true_mean = true_stack.mean(axis=0)
        true_se = true_stack.std(axis=0, ddof=1) / np.sqrt(true_stack.shape[0]) if true_stack.shape[0] > 1 else np.zeros_like(true_mean)
        plt.plot(grid / 1000.0, true_mean, label=f"NSW RCT {arm_name}", color=COLORS["true"], linewidth=3.0)
        plt.fill_between(
            grid / 1000.0,
            np.clip(true_mean - true_se, 0.0, 1.0),
            np.clip(true_mean + true_se, 0.0, 1.0),
            color=COLORS["true"],
            alpha=0.16,
            linewidth=0,
        )
        for x_value, y_value, se_value in zip(grid, true_mean, true_se, strict=True):
            csv_lines.append(f"{arm},NSW RCT {arm_name},{x_value:.8f},{y_value:.8f},{se_value:.8f}")

        for method in methods:
            method_stack = np.stack(
                [
                    _cdf_on_grid(
                        np.asarray(details["cdf_details"][method]["arm_samples_earnings"][arm], dtype=np.float64),
                        grid,
                    )
                    for details in detail_repetitions
                ],
                axis=0,
            )
            mean_curve = method_stack.mean(axis=0)
            se_curve = method_stack.std(axis=0, ddof=1) / np.sqrt(method_stack.shape[0]) if method_stack.shape[0] > 1 else np.zeros_like(mean_curve)
            linewidth = 3.0 if method == "ganice" else 1.9
            linestyle = "-" if method == "ganice" else "--"
            plt.plot(
                grid / 1000.0,
                mean_curve,
                label=LABELS[method],
                color=COLORS[method],
                linewidth=linewidth,
                linestyle=linestyle,
            )
            plt.fill_between(
                grid / 1000.0,
                np.clip(mean_curve - se_curve, 0.0, 1.0),
                np.clip(mean_curve + se_curve, 0.0, 1.0),
                color=COLORS[method],
                alpha=0.10 if method == "ganice" else 0.06,
                linewidth=0,
            )
            for x_value, y_value, se_value in zip(grid, mean_curve, se_curve, strict=True):
                csv_lines.append(f"{arm},{LABELS[method]},{x_value:.8f},{y_value:.8f},{se_value:.8f}")

        plt.xlim(left=0.0, right=max(5.0, x_max / 1000.0))
        plt.ylim(0.0, 1.01)
        plt.xlabel("RE78 earnings (thousand USD)")
        plt.ylabel("empirical CDF")
        place_legend_outside()
        plt.tight_layout()
        plt.savefig(output_dir / f"jobs_rct_cdf_{arm_name}.png", dpi=260)
        plt.savefig(output_dir / f"jobs_rct_cdf_{arm_name}.pdf")
        if arm == 1:
            plt.savefig(output_dir / "jobs_rct_cdf.png", dpi=260)
            plt.savefig(output_dir / "jobs_rct_cdf.pdf")
        plt.close()

    (output_dir / "jobs_rct_cdf_mean.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")


def write_jobs_cdf_outputs(output_dir: Path, rct: dict[str, np.ndarray], cdf_details: dict[str, dict[str, object]]) -> None:
    methods = ["ganite", "po_flow", "diff_po", "infs", "dr_learner", "ganice"]
    for arm, arm_name in [(1, "treated"), (0, "control")]:
        plt.figure(figsize=(10.8, 6.3))
        true_values = rct["y_earnings"][rct["t"] == arm]
        _plot_empirical_cdf(true_values, f"NSW RCT {arm_name}", COLORS["true"], 3.0)
        for method in methods:
            samples = cdf_details[method]["arm_samples_earnings"][arm]
            linewidth = 3.0 if method == "ganice" else 1.9
            linestyle = "-" if method == "ganice" else "--"
            _plot_empirical_cdf(samples, LABELS[method], COLORS[method], linewidth, linestyle)
        all_values = [rct["y_earnings"]]
        all_values.extend(cdf_details[method]["arm_samples_earnings"][arm] for method in methods)
        x_max = float(np.quantile(np.concatenate(all_values), 0.985) / 1000.0)
        plt.xlim(left=0.0, right=max(5.0, x_max))
        plt.ylim(0.0, 1.01)
        plt.xlabel("RE78 earnings (thousand USD)")
        plt.ylabel("empirical CDF")
        place_legend_outside()
        plt.tight_layout()
        plt.savefig(output_dir / f"jobs_rct_cdf_{arm_name}.png", dpi=260)
        plt.savefig(output_dir / f"jobs_rct_cdf_{arm_name}.pdf")
        if arm == 1:
            plt.savefig(output_dir / "jobs_rct_cdf.png", dpi=260)
            plt.savefig(output_dir / "jobs_rct_cdf.pdf")
        plt.close()


def write_jobs_policy_curve_output(output_dir: Path, rates: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    plt.figure(figsize=(10.6, 6.2))
    for method in ["ganite", "po_flow", "diff_po", "infs", "dr_learner", "ganice"]:
        linewidth = 3.0 if method == "ganice" else 2.0
        linestyle = "-" if method == "ganice" else "--"
        plt.plot(rates, curves[method] / 1000.0, label=LABELS[method], color=COLORS[method], linewidth=linewidth, linestyle=linestyle)
    plt.xlabel("treatment inclusion rate")
    plt.ylabel("RCT policy value (thousand USD)")
    place_legend_outside()
    plt.tight_layout()
    plt.savefig(output_dir / "jobs_policy_curve.png", dpi=260)
    plt.savefig(output_dir / "jobs_policy_curve.pdf")
    plt.close()


def run_single_jobs_lalonde(
    output_dir: Path,
    seed: int,
    quick: bool,
    include_nsw_control_in_observed: bool,
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    data = JobsLaLondeData(
        split_seed=seed + 1,
        include_nsw_control_in_observed=include_nsw_control_in_observed,
    )
    train = data.observed_split("train")
    validation_rct = data.rct_split("validation")
    test_rct = data.rct_split("test")
    outcome_lower, outcome_upper = data.outcome_bounds(train["y"])
    eval_samples = 48 if quick else 96
    mean_mc = 48 if quick else 96

    ganite = GANITE(jobs_ganite_config(seed + 11, data.feature_dim, outcome_lower, outcome_upper, quick=quick))
    ganite.fit(train["x"], train["t"], train["y"])
    ganite_sampler = make_degenerate_mean_sampler(
        ganite.predict_potential_outcomes,
        outcome_lower,
        outcome_upper,
    )

    po_flow = POFlow(jobs_po_flow_config(seed + 23, data.feature_dim, outcome_lower, outcome_upper, quick=quick))
    po_flow.fit(train["x"], train["t"], train["y"])

    diff_po = DiffPO(jobs_diff_po_config(seed + 29, data.feature_dim, outcome_lower, outcome_upper, quick=quick))
    diff_po.fit(train["x"], train["t"], train["y"])

    infs = INFs(jobs_infs_config(seed + 31, data.feature_dim, outcome_lower, outcome_upper, quick=quick))
    infs.fit(train["x"], train["t"], train["y"])

    dr_learner = DRLearner(jobs_dr_learner_config(seed + 37, data.feature_dim, outcome_lower, outcome_upper, quick=quick))
    dr_learner.fit(train["x"], train["t"], train["y"])
    dr_sampler = make_residual_plugin_sampler(
        dr_learner.predict_potential_outcomes,
        train["x"],
        train["t"],
        train["y"],
        outcome_lower,
        outcome_upper,
    )

    ganice_zero_masses = jobs_nsw_zero_masses(data, "train")
    ganice_cell_transform = fit_jobs_cell_transform(train["x"])
    ganice_cell_dim = 3
    ganice_candidates: list[tuple[float, GANICE, dict[str, float | int]]] = []
    base_ganice_config = jobs_ganice_config(
        seed + 101,
        data.feature_dim,
        outcome_lower,
        outcome_upper,
        quick=quick,
    )
    if quick:
        ganice_candidate_specs: list[dict[str, float | int]] = [
            {"num_steps": 120, "generator_transport_weight": 4.0, "factual_crps_weight": 4.0},
            {"num_steps": 160, "generator_transport_weight": 6.0, "factual_crps_weight": 5.0},
            {"num_steps": 180, "generator_transport_weight": 4.0, "factual_crps_weight": 4.0},
        ]
    else:
        ganice_candidate_specs = [
            {"num_steps": 180, "generator_transport_weight": 4.0, "factual_crps_weight": 4.0},
            {"num_steps": 220, "generator_transport_weight": 6.0, "factual_crps_weight": 5.0},
            {"num_steps": 260, "generator_transport_weight": 4.0, "factual_crps_weight": 4.0},
        ]
    validation_att = data.rct_att_earnings("validation")
    for restart, candidate_spec in enumerate(ganice_candidate_specs):
        candidate_config = replace(
            base_ganice_config,
            seed=seed + 101 + 100 * restart,
            **candidate_spec,
        )
        candidate = fit_ganice(
            d_w=data.d_w,
            train_w=train["w"],
            train_y=train["y"],
            target_w_sampler=lambda n, seed=None, fitted_data=data: fitted_data.sample_target_w(n, seed=seed, split="test"),
            config=candidate_config,
            seed=seed + 151 + 100 * restart,
            d_cell_w=ganice_cell_dim,
            cell_transform=ganice_cell_transform,
        )
        candidate_sampler = make_ganice_jobs_sampler(candidate, data, zero_masses=ganice_zero_masses)
        validation_metrics, _ = jobs_evaluate_sampler(
            candidate_sampler,
            validation_rct,
            validation_att,
            n_per_x=32 if quick else 64,
            mean_mc=32 if quick else 64,
            seed=seed + 50_000 + 1000 * restart,
            include_additional=False,
        )
        ganice_candidates.append((validation_metrics["rct_w1"], candidate, candidate_spec))
    ganice_validation_score, ganice, ganice_selected_spec = min(ganice_candidates, key=lambda item: item[0])
    ganice_uncalibrated_sampler = make_ganice_jobs_sampler(ganice, data, zero_masses=ganice_zero_masses)
    ganice_arm_calibrators = fit_jobs_arm_quantile_calibrators(
        ganice_uncalibrated_sampler,
        validation_rct,
        n_per_x=64 if quick else 128,
        seed=seed + 55_000,
    )

    samplers: dict[str, Sampler] = {
        "ganite": ganite_sampler,
        "po_flow": lambda x, treatment, n, sample_seed: po_flow.sample_potential(
            x,
            treatment,
            n_per_x=n,
            seed=sample_seed,
        ),
        "diff_po": lambda x, treatment, n, sample_seed: diff_po.sample_potential(
            x,
            treatment,
            n_per_x=n,
            seed=sample_seed,
        ),
        "infs": lambda x, treatment, n, sample_seed: infs.sample_potential(
            x,
            treatment,
            n_per_x=n,
            seed=sample_seed,
        ),
        "dr_learner": dr_sampler,
        "ganice": make_ganice_jobs_sampler(
            ganice,
            data,
            zero_masses=ganice_zero_masses,
            arm_quantile_calibrators=ganice_arm_calibrators,
        ),
    }

    true_att = data.rct_att_earnings("test")
    results: dict[str, dict[str, float]] = {}
    details: dict[str, dict[str, object]] = {}
    policy_curves: dict[str, np.ndarray] = {}
    rates = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    for method, sampler in samplers.items():
        method_metrics, method_detail = jobs_evaluate_sampler(
            sampler,
            test_rct,
            true_att,
            n_per_x=eval_samples,
            mean_mc=mean_mc,
            seed=seed + 60_000 + 1_000 * len(results),
        )
        results[method] = method_metrics
        details[method] = method_detail
        policy_curves[method] = jobs_policy_curve(
            test_rct["y_earnings"],
            test_rct["t"],
            np.asarray(method_detail["effects"], dtype=np.float64),
            rates,
        )
    results["po_flow"]["density_nll"] = po_flow.negative_log_likelihood(test_rct["x"], test_rct["t"], test_rct["y"])
    results["infs"]["density_nll"] = infs.negative_log_likelihood(test_rct["x"], test_rct["t"], test_rct["y"])
    results["ganice"]["target_mass_coverage"] = float(ganice.target_mass_coverage)
    results["ganice"]["validation_rct_w1"] = float(ganice_validation_score)
    results["ganice"]["arm_quantile_calibration"] = float(bool(ganice_arm_calibrators))
    for key, value in ganice_selected_spec.items():
        results["ganice"][f"selected_{key}"] = float(value)
    results["ganice"]["train_n"] = float(train["x"].shape[0])
    results["ganice"]["rct_test_n"] = float(test_rct["x"].shape[0])
    results["ganice"]["true_att"] = float(true_att)
    results["ganice"]["observed_includes_nsw_control"] = float(include_nsw_control_in_observed)

    return results, {
        "rct": test_rct,
        "cdf_details": details,
        "rates": rates.tolist(),
        "policy_curves": {method: curve.tolist() for method, curve in policy_curves.items()},
        "train_counts": data.source_counts("train"),
    }


def _run_jobs_repetition(task: tuple[Path, int, int, bool, bool]) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    output_dir, seed, rep, quick, include_nsw_control_in_observed = task
    rep_dir = ensure_dir(output_dir / "replications" / f"jobs_rep_{rep:03d}")
    return run_single_jobs_lalonde(
        rep_dir,
        seed + 10_000 * rep,
        quick=quick,
        include_nsw_control_in_observed=include_nsw_control_in_observed,
    )


def run_jobs_lalonde_benchmark(
    output_dir: Path,
    seed: int,
    repetitions: int,
    quick: bool,
    include_nsw_control_in_observed: bool,
    parallel: int = 1,
) -> dict[str, dict[str, float]]:
    tasks = [(output_dir, seed, rep, quick, include_nsw_control_in_observed) for rep in range(repetitions)]
    rep_outputs = _run_parallel_repetitions(_run_jobs_repetition, tasks, parallel)
    per_repetition = [result for result, _ in rep_outputs]
    last_details: dict[str, object] | None = rep_outputs[-1][1] if rep_outputs else None
    summary = summarize_jobs_repetitions(per_repetition)
    write_jobs_tables(output_dir, summary)
    write_jobs_metric_bar_outputs(output_dir, summary)
    write_jobs_additional_outputs(output_dir, summary)
    if last_details is not None:
        write_jobs_cdf_outputs(
            output_dir,
            last_details["rct"],
            last_details["cdf_details"],
        )
        write_jobs_policy_curve_output(
            output_dir,
            np.asarray(last_details["rates"], dtype=np.float64),
            {method: np.asarray(values, dtype=np.float64) for method, values in last_details["policy_curves"].items()},
        )
        (output_dir / "jobs_lalonde_last_split.json").write_text(
            json.dumps(
                {
                    "train_counts": last_details["train_counts"],
                    "rates": last_details["rates"],
                    "policy_curves": last_details["policy_curves"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    write_jobs_cdf_repetition_outputs(output_dir, [details for _, details in rep_outputs])
    (output_dir / "jobs_lalonde_repetitions.json").write_text(
        json.dumps(per_repetition, indent=2),
        encoding="utf-8",
    )
    return summary


def vcnet_features(x: np.ndarray, treatment: np.ndarray | int, num_treatments: int) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    treatment_arr = np.asarray(treatment, dtype=np.int64).reshape(-1)
    if treatment_arr.size == 1 and x_arr.shape[0] > 1:
        treatment_arr = np.full(x_arr.shape[0], int(treatment_arr.item()), dtype=np.int64)
    one_hot = np.eye(num_treatments, dtype=np.float32)[treatment_arr]
    return np.column_stack([x_arr, one_hot]).astype(np.float32)


def predict_scigan_factual(model: SCIGAN, x: np.ndarray, treatment: np.ndarray, dosage: np.ndarray) -> np.ndarray:
    pred = np.empty((x.shape[0], 1), dtype=np.float32)
    for treatment_value in np.unique(treatment).astype(np.int64):
        mask = treatment == int(treatment_value)
        pred[mask] = model.predict_response(x[mask], treatment=int(treatment_value), dosage=dosage[mask])
    return pred


def residual_pools_by_treatment_dose(
    treatment: np.ndarray,
    dosage: np.ndarray,
    residuals: np.ndarray,
    *,
    num_treatments: int,
    num_strata: int = 5,
) -> dict[tuple[int, int], np.ndarray]:
    treatment_arr = np.asarray(treatment, dtype=np.int64).reshape(-1)
    dosage_arr = np.asarray(dosage, dtype=np.float32).reshape(-1)
    residual_arr = np.asarray(residuals, dtype=np.float32).reshape(-1)
    strata = np.floor(np.clip(dosage_arr, 0.0, np.nextafter(1.0, 0.0)) * num_strata).astype(np.int64)
    pools: dict[tuple[int, int], np.ndarray] = {}
    global_pool = residual_arr
    for treatment_value in range(num_treatments):
        treatment_pool = residual_arr[treatment_arr == treatment_value]
        if treatment_pool.size == 0:
            treatment_pool = global_pool
        for stratum in range(num_strata):
            mask = (treatment_arr == treatment_value) & (strata == stratum)
            pool = residual_arr[mask]
            pools[(treatment_value, stratum)] = pool if pool.size > 0 else treatment_pool
    pools[(-1, -1)] = global_pool
    return pools


def sample_residual_plugin(
    mean: float,
    pools: dict[tuple[int, int], np.ndarray],
    treatment: int,
    dosage: float,
    n: int,
    seed: int,
    *,
    num_strata: int = 5,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    stratum = int(np.floor(np.clip(dosage, 0.0, np.nextafter(1.0, 0.0)) * num_strata))
    pool = pools.get((int(treatment), stratum), pools[(-1, -1)])
    residual = rng.choice(pool, size=n, replace=True)
    return (float(mean) + residual).reshape(-1, 1).astype(np.float32)


def build_tcga_true_grid(dgp: TCGADoseDGP, z_test: np.ndarray, dosage_grid: np.ndarray) -> np.ndarray:
    grid = np.zeros((z_test.shape[0], dgp.num_treatments, dosage_grid.size), dtype=np.float64)
    for treatment in range(dgp.num_treatments):
        for idx_d, dosage in enumerate(dosage_grid):
            grid[:, treatment, idx_d] = dgp.response_mean(z_test, treatment, dosage).astype(np.float64)
    return grid


def build_tcga_scigan_grid(model: SCIGAN, dgp: TCGADoseDGP, x_test: np.ndarray, dosage_grid: np.ndarray) -> np.ndarray:
    grid = np.zeros((x_test.shape[0], dgp.num_treatments, dosage_grid.size), dtype=np.float64)
    for treatment in range(dgp.num_treatments):
        for idx_d, dosage in enumerate(dosage_grid):
            grid[:, treatment, idx_d] = model.predict_response(
                x_test,
                treatment=treatment,
                dosage=np.full(x_test.shape[0], dosage, dtype=np.float32),
            ).reshape(-1).astype(np.float64)
    return grid


def build_tcga_vcnet_grid(model: VCNet, dgp: TCGADoseDGP, x_test: np.ndarray, dosage_grid: np.ndarray) -> np.ndarray:
    grid = np.zeros((x_test.shape[0], dgp.num_treatments, dosage_grid.size), dtype=np.float64)
    for treatment in range(dgp.num_treatments):
        features = vcnet_features(x_test, treatment, dgp.num_treatments)
        for idx_d, dosage in enumerate(dosage_grid):
            grid[:, treatment, idx_d] = model.predict_response(
                features,
                treatment=np.full(x_test.shape[0], dosage, dtype=np.float32),
            ).reshape(-1).astype(np.float64)
    return grid


def build_tcga_drnet_grid(model: DRNet, dgp: TCGADoseDGP, x_test: np.ndarray, dosage_grid: np.ndarray) -> np.ndarray:
    grid = np.zeros((x_test.shape[0], dgp.num_treatments, dosage_grid.size), dtype=np.float64)
    for treatment in range(dgp.num_treatments):
        for idx_d, dosage in enumerate(dosage_grid):
            grid[:, treatment, idx_d] = model.predict_response(
                x_test,
                treatment=treatment,
                dosage=np.full(x_test.shape[0], dosage, dtype=np.float32),
            ).reshape(-1).astype(np.float64)
    return grid


def build_tcga_ganice_grid(model: GANICE, dgp: TCGADoseDGP, x_test: np.ndarray, dosage_grid: np.ndarray, n_mc: int) -> np.ndarray:
    grid = np.zeros((x_test.shape[0], dgp.num_treatments, dosage_grid.size), dtype=np.float64)
    for treatment in range(dgp.num_treatments):
        for idx_d, dosage in enumerate(dosage_grid):
            query = dgp.encode_w(
                x_test,
                np.full(x_test.shape[0], treatment, dtype=np.int64),
                np.full(x_test.shape[0], dosage, dtype=np.float32),
            )
            grid[:, treatment, idx_d] = model.predict_mean(query, n_mc=n_mc).reshape(-1).astype(np.float64)
    return grid


def encode_tcga_ganice_w(dgp: TCGADoseDGP, z: np.ndarray, treatment: np.ndarray | int, dosage: np.ndarray | float) -> np.ndarray:
    z_arr = np.asarray(z, dtype=np.float32)
    if z_arr.ndim == 1:
        z_arr = z_arr[None, :]
    n = z_arr.shape[0]
    treatment_arr = np.asarray(treatment, dtype=np.int64).reshape(-1)
    dosage_arr = np.asarray(dosage, dtype=np.float32).reshape(-1)
    if treatment_arr.size == 1 and n > 1:
        treatment_arr = np.full(n, int(treatment_arr.item()), dtype=np.int64)
    if dosage_arr.size == 1 and n > 1:
        dosage_arr = np.full(n, float(dosage_arr.item()), dtype=np.float32)
    if treatment_arr.shape[0] != n or dosage_arr.shape[0] != n:
        raise ValueError("treatment and dosage must be scalar or align with z")
    return np.column_stack([z_arr, dgp.treatment_embedding(treatment_arr), dosage_arr]).astype(np.float32)


def sample_tcga_ganice_target_w(
    dgp: TCGADoseDGP,
    n: int,
    seed: int | None,
    split: str = "test",
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    _, z_pool = dgp.split_arrays(split)  # type: ignore[arg-type]
    draw = rng.integers(0, z_pool.shape[0], size=n)
    treatment = rng.integers(0, dgp.num_treatments, size=n)
    dosage = rng.choice(dgp.dosage_grid, size=n, replace=True)
    return encode_tcga_ganice_w(dgp, z_pool[draw], treatment, dosage)


def build_tcga_ganice_grid_from_z(
    model: GANICE,
    dgp: TCGADoseDGP,
    z_test: np.ndarray,
    dosage_grid: np.ndarray,
    n_mc: int,
) -> np.ndarray:
    grid = np.zeros((z_test.shape[0], dgp.num_treatments, dosage_grid.size), dtype=np.float64)
    for treatment in range(dgp.num_treatments):
        for idx_d, dosage in enumerate(dosage_grid):
            query = encode_tcga_ganice_w(
                dgp,
                z_test,
                np.full(z_test.shape[0], treatment, dtype=np.int64),
                np.full(z_test.shape[0], dosage, dtype=np.float32),
            )
            grid[:, treatment, idx_d] = model.predict_mean(query, n_mc=n_mc).reshape(-1).astype(np.float64)
    return grid


def tcga_response_metrics_from_grid(true_grid: np.ndarray, pred_grid: np.ndarray) -> dict[str, float]:
    mise = float(np.mean((pred_grid - true_grid) ** 2))
    dpe_terms: list[float] = []
    pe_terms: list[float] = []
    n, num_treatments, _ = true_grid.shape
    for i in range(n):
        true_best_values = []
        achieved_values = []
        for treatment in range(num_treatments):
            true_idx = int(np.argmax(true_grid[i, treatment]))
            pred_idx = int(np.argmax(pred_grid[i, treatment]))
            true_at_true = float(true_grid[i, treatment, true_idx])
            true_at_pred = float(true_grid[i, treatment, pred_idx])
            dpe_terms.append((true_at_true - true_at_pred) ** 2)
            true_best_values.append(true_at_true)
            achieved_values.append(true_at_pred)
        best_true = float(np.max(true_best_values))
        best_pred_t = int(np.argmax(np.max(pred_grid[i], axis=1)))
        best_pred_d = int(np.argmax(pred_grid[i, best_pred_t]))
        achieved = float(true_grid[i, best_pred_t, best_pred_d])
        pe_terms.append(best_true - achieved)
    return {
        "mise": mise,
        "dpe": float(np.mean(dpe_terms)),
        "policy_error": float(np.mean(pe_terms)),
    }


def tcga_extended_w1(
    dgp: TCGADoseDGP,
    x_eval: np.ndarray,
    z_eval: np.ndarray,
    dosage_grid: np.ndarray,
    learned_sampler,
    *,
    n_per_state: int,
    seed: int,
) -> float:
    total = 0.0
    count = 0
    for i in range(x_eval.shape[0]):
        for treatment in range(dgp.num_treatments):
            for idx_d, dosage in enumerate(dosage_grid):
                state_seed = seed + 10_000 * i + 200 * treatment + idx_d
                z_rep = np.repeat(z_eval[i : i + 1], n_per_state, axis=0)
                y_true = dgp.sample_potential(
                    z_rep,
                    np.full(n_per_state, treatment, dtype=np.int64),
                    np.full(n_per_state, float(dosage), dtype=np.float32),
                    seed=state_seed,
                )
                y_learned = learned_sampler(
                    x_eval[i : i + 1],
                    z_eval[i : i + 1],
                    treatment,
                    float(dosage),
                    n_per_state,
                    state_seed + 50_000,
                )
                total += sample_wasserstein_1_1d(y_true, y_learned)
                count += 1
    return float(total / count)


def tcga_distribution_metrics(
    dgp: TCGADoseDGP,
    x_eval: np.ndarray,
    z_eval: np.ndarray,
    dosage_grid: np.ndarray,
    learned_sampler,
    *,
    n_per_state: int,
    seed: int,
) -> dict[str, float]:
    acc = _empty_distribution_accumulators()
    for i in range(x_eval.shape[0]):
        for treatment in range(dgp.num_treatments):
            for idx_d, dosage in enumerate(dosage_grid):
                state_seed = seed + 10_000 * i + 200 * treatment + idx_d
                z_rep = np.repeat(z_eval[i : i + 1], n_per_state, axis=0)
                y_true = dgp.sample_potential(
                    z_rep,
                    np.full(n_per_state, treatment, dtype=np.int64),
                    np.full(n_per_state, float(dosage), dtype=np.float32),
                    seed=state_seed,
                )
                y_learned = learned_sampler(
                    x_eval[i : i + 1],
                    z_eval[i : i + 1],
                    treatment,
                    float(dosage),
                    n_per_state,
                    state_seed + 50_000,
                )
                _accumulate_distribution_state(acc, y_learned, y_true)
    metrics = _finalize_distribution_accumulators(acc)
    metrics["dose_quantile_error"] = metrics["iqe"]
    return metrics


def summarize_tcga_repetitions(repetitions: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    methods = sorted(repetitions[0].keys())
    summary: dict[str, dict[str, float]] = {}
    for method in methods:
        keys = sorted(repetitions[0][method].keys())
        summary[method] = {}
        for key in keys:
            values = np.asarray([rep[method][key] for rep in repetitions if key in rep[method]], dtype=np.float64)
            summary[method][f"{key}_mean"] = float(np.mean(values))
            summary[method][f"{key}_se"] = float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
    return summary


def write_tcga_tables(output_dir: Path, summary: dict[str, dict[str, float]]) -> None:
    order = ["scigan", "vcnet", "drnet", "ganice"]
    csv_lines = ["method,extended_w1_mean,extended_w1_se,policy_error_mean,policy_error_se,mise_mean,mise_se,dpe_mean,dpe_se"]
    tex_lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Method & eW$_1$ & PE \\\\",
        "\\midrule",
    ]
    best_w1 = min(summary[m]["extended_w1_mean"] for m in order if m in summary)
    for method in order:
        if method not in summary:
            continue
        row = summary[method]
        csv_lines.append(
            f"{LABELS[method]},{row['extended_w1_mean']:.6f},{row['extended_w1_se']:.6f},"
            f"{row['policy_error_mean']:.6f},{row['policy_error_se']:.6f},"
            f"{row['mise_mean']:.6f},{row['mise_se']:.6f},{row['dpe_mean']:.6f},{row['dpe_se']:.6f}"
        )
        w1 = f"{row['extended_w1_mean']:.3f} $\\pm$ {row['extended_w1_se']:.3f}"
        pe = f"{row['policy_error_mean']:.3f} $\\pm$ {row['policy_error_se']:.3f}"
        if abs(row["extended_w1_mean"] - best_w1) < 1e-12:
            w1 = f"\\textbf{{{w1}}}"
        tex_lines.append(f"{LABELS[method]} & {w1} & {pe} \\\\")
    tex_lines += ["\\bottomrule", "\\end{tabular}"]
    (output_dir / "tcga_dose_table.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    (output_dir / "tcga_dose_table.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def write_tcga_metric_bar_outputs(output_dir: Path, summary: dict[str, dict[str, float]]) -> None:
    methods = [method for method in ["scigan", "vcnet", "drnet", "ganice"] if method in summary]
    if not methods:
        return
    labels = [LABELS[method] for method in methods]
    colors = [COLORS.get(method, "#999999") for method in methods]
    save_metric_bar_plot(
        output_dir / "tcga_extended_w1_bar",
        labels,
        [summary[method]["extended_w1_mean"] for method in methods],
        [summary[method].get("extended_w1_se", 0.0) for method in methods],
        colors,
        title="TCGA-Dose: distributional error",
        ylabel="empirical eW1",
        rotation=15,
    )
    save_metric_bar_plot(
        output_dir / "tcga_policy_error_bar",
        labels,
        [summary[method]["policy_error_mean"] for method in methods],
        [summary[method].get("policy_error_se", 0.0) for method in methods],
        colors,
        title="TCGA-Dose: policy error",
        ylabel="policy error",
        rotation=15,
    )


def write_tcga_additional_outputs(output_dir: Path, summary: dict[str, dict[str, float]]) -> None:
    metrics = [
        ("crps", "CRPS", "min"),
        ("energy_distance", "ED", "min"),
        ("mmd2", "MMD$^2$", "min"),
        ("ks", "KS", "min"),
        ("cvm", "CvM", "min"),
        ("iqe", "IQE", "min"),
        ("dose_quantile_error", "DQErr", "min"),
        ("tail_error", "TailErr", "min"),
        ("calibration_error", "CalErr", "min"),
        ("mise", "MISE", "min"),
        ("dpe", "DPE", "min"),
        ("policy_error", "PE", "min"),
    ]
    _write_additional_metric_table(output_dir, "tcga_additional_metrics", summary, TCGA_METHODS, metrics)
    save_metric_grid_bar_plot(
        output_dir / "tcga_additional_metrics_bar",
        summary,
        TCGA_METHODS,
        [(key, label.replace("$", "")) for key, label, _ in metrics[:9]],
        rotation=15,
    )
    save_interval_width_plot(
        output_dir / "tcga_interval_widths",
        summary,
        TCGA_METHODS,
        ylabel="average interval width",
    )
    save_pit_histogram_plot(output_dir / "tcga_pit_histograms", summary, TCGA_METHODS)


def write_tcga_quantile_band(
    output_dir: Path,
    dgp: TCGADoseDGP,
    x_eval: np.ndarray,
    z_eval: np.ndarray,
    dosage_grid: np.ndarray,
    samplers: dict[str, object],
    *,
    treatment: int,
    n_per_x: int,
    seed: int,
) -> None:
    quantiles = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    curves: dict[str, np.ndarray] = {}
    for name, sampler in samplers.items():
        values = np.zeros((dosage_grid.size, quantiles.size), dtype=np.float64)
        for idx_d, dosage in enumerate(dosage_grid):
            per_x = []
            for i in range(x_eval.shape[0]):
                state_seed = seed + 20_000 * idx_d + i
                if name == "true":
                    z_rep = np.repeat(z_eval[i : i + 1], n_per_x, axis=0)
                    samples = dgp.sample_potential(
                        z_rep,
                        np.full(n_per_x, treatment, dtype=np.int64),
                        np.full(n_per_x, float(dosage), dtype=np.float32),
                        seed=state_seed,
                    )
                else:
                    samples = sampler(
                        x_eval[i : i + 1],
                        z_eval[i : i + 1],
                        treatment,
                        float(dosage),
                        n_per_x,
                        state_seed,
                    )
                per_x.append(np.quantile(samples.reshape(-1), quantiles))
            values[idx_d] = np.mean(np.stack(per_x, axis=0), axis=0)
        curves[name] = values

    csv_lines = ["method,dosage,q10,q50,q90"]
    for name, values in curves.items():
        for dosage, quantile_values in zip(dosage_grid, values, strict=True):
            csv_lines.append(
                f"{name},{float(dosage):.8f},{quantile_values[0]:.8f},{quantile_values[1]:.8f},{quantile_values[2]:.8f}"
            )
    (output_dir / "tcga_dose_quantile_band_curves.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    plt.figure(figsize=(10.6, 6.2))
    true_curve = curves["true"]
    plt.fill_between(dosage_grid, true_curve[:, 0], true_curve[:, 2], color=COLORS["true"], alpha=0.18)
    plt.plot(dosage_grid, true_curve[:, 1], color=COLORS["true"], linewidth=2.8, label="true median")
    if "ganice" in curves:
        ganice_curve = curves["ganice"]
        plt.fill_between(dosage_grid, ganice_curve[:, 0], ganice_curve[:, 2], color=COLORS["ganice"], alpha=0.18)
        plt.plot(dosage_grid, ganice_curve[:, 1], color=COLORS["ganice"], linewidth=2.8, label="GANICE median")
    for method in ("scigan", "vcnet", "drnet"):
        if method in curves:
            plt.plot(
                dosage_grid,
                curves[method][:, 1],
                color=COLORS[method],
                linewidth=1.9,
                linestyle="--",
                label=f"{LABELS[method]} median",
            )
    plt.xlabel("dosage")
    plt.ylabel("outcome")
    place_legend_outside()
    plt.tight_layout()
    plt.savefig(output_dir / "tcga_dose_quantile_band.png", dpi=260)
    plt.savefig(output_dir / "tcga_dose_quantile_band.pdf")
    plt.close()


def write_tcga_quantile_band_repetition_outputs(output_dir: Path) -> None:
    curve_files = sorted((output_dir / "replications").glob("tcga_rep_*/tcga_dose_quantile_band_curves.csv"))
    if not curve_files:
        return
    per_method: dict[str, list[np.ndarray]] = {}
    dosage_grid: np.ndarray | None = None
    for curve_file in curve_files:
        method_rows: dict[str, list[tuple[float, np.ndarray]]] = {}
        with curve_file.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                method_rows.setdefault(row["method"], []).append(
                    (
                        float(row["dosage"]),
                        np.asarray([float(row["q10"]), float(row["q50"]), float(row["q90"])], dtype=np.float64),
                    )
                )
        for method, entries in method_rows.items():
            entries.sort(key=lambda item: item[0])
            local_grid = np.asarray([item[0] for item in entries], dtype=np.float64)
            if dosage_grid is None:
                dosage_grid = local_grid
            values = np.stack([item[1] for item in entries], axis=0)
            per_method.setdefault(method, []).append(values)
    if dosage_grid is None:
        return

    csv_lines = ["method,dosage,q10,q10_se,q50,q50_se,q90,q90_se"]
    curves_mean: dict[str, np.ndarray] = {}
    curves_se: dict[str, np.ndarray] = {}
    for method, curves in per_method.items():
        stacked = np.stack(curves, axis=0)
        curves_mean[method] = stacked.mean(axis=0)
        curves_se[method] = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0]) if stacked.shape[0] > 1 else np.zeros_like(curves_mean[method])
        for dosage, mean_values, se_values in zip(dosage_grid, curves_mean[method], curves_se[method], strict=True):
            csv_lines.append(
                f"{method},{float(dosage):.8f},"
                f"{mean_values[0]:.8f},{se_values[0]:.8f},"
                f"{mean_values[1]:.8f},{se_values[1]:.8f},"
                f"{mean_values[2]:.8f},{se_values[2]:.8f}"
            )
    (output_dir / "tcga_dose_quantile_band_mean.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    plt.figure(figsize=(10.6, 6.2))
    if "true" in curves_mean:
        true_curve = curves_mean["true"]
        true_se = curves_se["true"]
        plt.fill_between(dosage_grid, true_curve[:, 0], true_curve[:, 2], color=COLORS["true"], alpha=0.18)
        plt.plot(dosage_grid, true_curve[:, 1], color=COLORS["true"], linewidth=2.8, label="true median")
        plt.fill_between(
            dosage_grid,
            true_curve[:, 1] - true_se[:, 1],
            true_curve[:, 1] + true_se[:, 1],
            color=COLORS["true"],
            alpha=0.12,
            linewidth=0,
        )
    if "ganice" in curves_mean:
        ganice_curve = curves_mean["ganice"]
        ganice_se = curves_se["ganice"]
        plt.fill_between(dosage_grid, ganice_curve[:, 0], ganice_curve[:, 2], color=COLORS["ganice"], alpha=0.18)
        plt.plot(dosage_grid, ganice_curve[:, 1], color=COLORS["ganice"], linewidth=2.8, label="GANICE median")
        plt.fill_between(
            dosage_grid,
            ganice_curve[:, 1] - ganice_se[:, 1],
            ganice_curve[:, 1] + ganice_se[:, 1],
            color=COLORS["ganice"],
            alpha=0.12,
            linewidth=0,
        )
    for method in ("scigan", "vcnet", "drnet"):
        if method in curves_mean:
            mean_curve = curves_mean[method]
            se_curve = curves_se[method]
            plt.plot(
                dosage_grid,
                mean_curve[:, 1],
                color=COLORS[method],
                linewidth=1.9,
                linestyle="--",
                label=f"{LABELS[method]} median",
            )
            plt.fill_between(
                dosage_grid,
                mean_curve[:, 1] - se_curve[:, 1],
                mean_curve[:, 1] + se_curve[:, 1],
                color=COLORS[method],
                alpha=0.08,
                linewidth=0,
            )
    plt.xlabel("dosage")
    plt.ylabel("outcome")
    place_legend_outside()
    plt.tight_layout()
    plt.savefig(output_dir / "tcga_dose_quantile_band.png", dpi=260)
    plt.savefig(output_dir / "tcga_dose_quantile_band.pdf")
    plt.close()


def run_single_tcga_dose(output_dir: Path, data_dir: Path, seed: int, quick: bool) -> dict[str, dict[str, float]]:
    dgp = TCGADoseDGP(data_dir=data_dir, seed=seed)
    train = dgp.sample_observed("train", seed=seed + 1)
    test_x, test_z = dgp.split_arrays("test")
    dosage_grid = dgp.dosage_grid
    if quick:
        train_n = min(2_400, train["x"].shape[0])
        train = {key: value[:train_n] for key, value in train.items()}
        test_x = test_x[:64]
        test_z = test_z[:64]
    outcome_lower = float(np.quantile(train["y"], 0.002) - 0.4)
    outcome_upper = float(np.quantile(train["y"], 0.998) + 0.4)

    scigan = SCIGAN(tcga_scigan_config(seed + 7, dgp.feature_dim, dgp.num_treatments, quick=quick))
    scigan.fit(train["x"], train["t"], train["d"], train["y"])
    scigan_grid = build_tcga_scigan_grid(scigan, dgp, test_x, dosage_grid)

    vcnet_train_x = vcnet_features(train["x"], train["t"], dgp.num_treatments)
    vcnet = VCNet(tcga_vcnet_config(seed + 57, vcnet_train_x.shape[1], quick=quick))
    vcnet.fit(vcnet_train_x, train["d"], train["y"])
    vcnet_grid = build_tcga_vcnet_grid(vcnet, dgp, test_x, dosage_grid)

    drnet = DRNet(tcga_drnet_config(seed + 63, dgp.feature_dim, dgp.num_treatments, quick=quick))
    drnet.fit(train["x"], train["t"], train["d"], train["y"])
    drnet_grid = build_tcga_drnet_grid(drnet, dgp, test_x, dosage_grid)

    ganice = fit_ganice(
        d_w=dgp.pc_dim + 2,
        train_w=encode_tcga_ganice_w(dgp, train["z"], train["t"], train["d"]),
        train_y=train["y"],
        target_w_sampler=lambda n, seed=None: sample_tcga_ganice_target_w(dgp, n, seed=seed, split="test"),
        config=tcga_ganice_config(seed + 102, dgp.pc_dim, outcome_lower, outcome_upper, quick=quick),
        seed=seed + 202,
    )
    ganice_grid = build_tcga_ganice_grid_from_z(ganice, dgp, test_z, dosage_grid, n_mc=96 if quick else 160)
    true_grid = build_tcga_true_grid(dgp, test_z, dosage_grid)

    scigan_residuals = train["y"] - predict_scigan_factual(scigan, train["x"], train["t"], train["d"])
    vcnet_residuals = train["y"] - vcnet.predict_response(vcnet_train_x, train["d"])
    drnet_residuals = train["y"] - drnet.predict_response(train["x"], train["t"], train["d"])
    scigan_pools = residual_pools_by_treatment_dose(train["t"], train["d"], scigan_residuals, num_treatments=dgp.num_treatments)
    vcnet_pools = residual_pools_by_treatment_dose(train["t"], train["d"], vcnet_residuals, num_treatments=dgp.num_treatments)
    drnet_pools = residual_pools_by_treatment_dose(train["t"], train["d"], drnet_residuals, num_treatments=dgp.num_treatments)

    def scigan_sampler(x, z, treatment, dosage, n, seed):
        del z
        mean = scigan.predict_response(x, treatment=treatment, dosage=np.array([dosage], dtype=np.float32))[0, 0]
        return sample_residual_plugin(mean, scigan_pools, treatment, dosage, n, seed)

    def vcnet_sampler(x, z, treatment, dosage, n, seed):
        del z
        mean = vcnet.predict_response(
            vcnet_features(x, treatment, dgp.num_treatments),
            treatment=np.array([dosage], dtype=np.float32),
        )[0, 0]
        return sample_residual_plugin(mean, vcnet_pools, treatment, dosage, n, seed)

    def drnet_sampler(x, z, treatment, dosage, n, seed):
        del z
        mean = drnet.predict_response(x, treatment=treatment, dosage=np.array([dosage], dtype=np.float32))[0, 0]
        return sample_residual_plugin(mean, drnet_pools, treatment, dosage, n, seed)

    def ganice_sampler(x, z, treatment, dosage, n, seed):
        del x
        return ganice.sample_conditional(encode_tcga_ganice_w(dgp, z, treatment, dosage), n, seed=seed)

    eval_n = min(40 if quick else 120, test_x.shape[0])
    n_per_state = 64 if quick else 128
    results = {
        "scigan": tcga_response_metrics_from_grid(true_grid, scigan_grid),
        "vcnet": tcga_response_metrics_from_grid(true_grid, vcnet_grid),
        "drnet": tcga_response_metrics_from_grid(true_grid, drnet_grid),
        "ganice": tcga_response_metrics_from_grid(true_grid, ganice_grid),
    }
    for name, sampler in {
        "scigan": scigan_sampler,
        "vcnet": vcnet_sampler,
        "drnet": drnet_sampler,
        "ganice": ganice_sampler,
    }.items():
        results[name].update(
            tcga_distribution_metrics(
                dgp,
                test_x[:eval_n],
                test_z[:eval_n],
                dosage_grid,
                sampler,
                n_per_state=n_per_state,
                seed=seed + 80_000,
            )
        )
    results["ganice"]["target_mass_coverage"] = float(ganice.target_mass_coverage)

    write_tcga_quantile_band(
        output_dir,
        dgp,
        test_x[: min(24 if quick else 64, test_x.shape[0])],
        test_z[: min(24 if quick else 64, test_z.shape[0])],
        dosage_grid,
        {
            "true": None,
            "scigan": scigan_sampler,
            "vcnet": vcnet_sampler,
            "drnet": drnet_sampler,
            "ganice": ganice_sampler,
        },
        treatment=0,
        n_per_x=48 if quick else 96,
        seed=seed + 90_000,
    )
    return results


def _run_tcga_repetition(task: tuple[Path, Path, int, int, bool]) -> dict[str, dict[str, float]]:
    output_dir, data_dir, seed, rep, quick = task
    rep_dir = ensure_dir(output_dir / "replications" / f"tcga_rep_{rep:03d}")
    return run_single_tcga_dose(rep_dir, data_dir, seed + 113 * rep, quick=quick)


def run_tcga_dose_benchmark(
    output_dir: Path,
    data_dir: Path,
    seed: int,
    *,
    repetitions: int,
    quick: bool,
    parallel: int = 1,
) -> dict[str, dict[str, float] | object]:
    tasks = [(output_dir, data_dir, seed, rep, quick) for rep in range(repetitions)]
    per_repetition = _run_parallel_repetitions(_run_tcga_repetition, tasks, parallel)
    summary = summarize_tcga_repetitions(per_repetition)
    write_tcga_tables(output_dir, summary)
    write_tcga_metric_bar_outputs(output_dir, summary)
    write_tcga_additional_outputs(output_dir, summary)
    if repetitions > 0:
        last_rep_dir = output_dir / "replications" / f"tcga_rep_{repetitions - 1:03d}"
        for suffix in ("png", "pdf"):
            source = last_rep_dir / f"tcga_dose_quantile_band.{suffix}"
            if source.exists():
                shutil.copy2(source, output_dir / source.name)
    write_tcga_quantile_band_repetition_outputs(output_dir)
    ranking = sorted(
        ((method, values["extended_w1_mean"]) for method, values in summary.items()),
        key=lambda item: item[1],
    )
    payload: dict[str, dict[str, float] | object] = {
        "summary": summary,
        "ranking_by_extended_w1": ranking,
        "ganice_best_extended_w1": bool(ranking and ranking[0][0] == "ganice"),
    }
    (output_dir / "tcga_dose_repetitions.json").write_text(
        json.dumps(per_repetition, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GANICE IHDP/Jobs/TCGA full-data experiments and baseline comparisons.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ganice_experiment"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--benchmark",
        choices=("all", "ihdp", "jobs", "tcga"),
        default="all",
        help="Select a benchmark; use jobs for the Jobs/LaLonde real-data experiment.",
    )
    parser.add_argument("--repetitions", type=int, default=1, help="Number of random splits/seeds for repeated benchmarks.")
    parser.add_argument("--parallel", type=int, default=1, help="Maximum number of repetitions to run concurrently.")
    parser.add_argument("--quick", action="store_true", help="Use shorter training and evaluation settings where available.")
    parser.add_argument(
        "--jobs-observed-controls",
        choices=("psid_nsw", "psid"),
        default="psid",
        help="Use PSID plus NSW train-split controls, or the stricter NSW-treated plus PSID-controls observational design.",
    )
    parser.add_argument("--tcga-data-dir", type=Path, default=Path("Data/TCGA"))
    parser.add_argument(
        "--tcga-download",
        action="store_true",
        help="Download drnet's tcga.db into --tcga-data-dir before running TCGA-Dose.",
    )
    parser.add_argument(
        "--tcga-force-download",
        action="store_true",
        help="Overwrite an existing TCGA SQLite database when --tcga-download is used.",
    )
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    metrics = {}
    if args.tcga_download:
        download_tcga_db(args.tcga_data_dir, force=args.tcga_force_download)
    if args.benchmark in ("all", "ihdp"):
        metrics["ihdp_dist"] = run_ihdp_dist_benchmark(
            output_dir,
            args.seed + 1_000,
            repetitions=args.repetitions,
            quick=args.quick,
            parallel=args.parallel,
        )
    if args.benchmark in ("all", "jobs"):
        metrics["jobs_lalonde"] = run_jobs_lalonde_benchmark(
            output_dir,
            args.seed + 1_500,
            repetitions=args.repetitions,
            quick=args.quick,
            include_nsw_control_in_observed=args.jobs_observed_controls == "psid_nsw",
            parallel=args.parallel,
        )
    if args.benchmark in ("all", "tcga"):
        metrics["tcga_dose"] = run_tcga_dose_benchmark(
            output_dir,
            args.tcga_data_dir,
            args.seed + 3_500,
            repetitions=args.repetitions,
            quick=args.quick,
            parallel=args.parallel,
        )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
