from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from counterfactual_gan import (
    ContinuousCausalDGP,
    ContinuousIGAN,
    ContinuousIGANConfig,
    FiniteStateCausalDGP,
    FiniteStateIGAN,
    FiniteStateIGANConfig,
    GANITE,
    GANITEBenchmarkDGP,
    GANITEConfig,
    SCIGAN,
    SCIGANBenchmarkDGP,
    SCIGANConfig,
)
from counterfactual_gan.utils import ensure_dir


IMPLEMENTATION_ORDER = ("voronoi", "anisotropic", "kernel")
IMPLEMENTATION_LABELS = {
    "voronoi": "IGAN-A",
    "anisotropic": "IGAN-B",
    "kernel": "IGAN-C",
}
IMPLEMENTATION_COLORS = {
    "voronoi": "#d95f02",
    "anisotropic": "#e7298a",
    "kernel": "#66a61e",
}


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

    plt.figure(figsize=(max(8, 1.15 * len(labels)), 4.6))
    for idx, (name, values, color) in enumerate(series):
        plt.bar(x_axis + offsets[idx], values, width=width, label=name, color=color)
    plt.xticks(x_axis, labels, rotation=rotation)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def fit_igan_variants(
    *,
    d_w: int,
    train_w: np.ndarray,
    train_y: np.ndarray,
    q_obs_density,
    q_target_density,
    target_w_sampler,
    kappa: float,
    config_map: dict[str, ContinuousIGANConfig],
    seed_offset: int,
) -> dict[str, ContinuousIGAN]:
    models: dict[str, ContinuousIGAN] = {}
    for index, implementation in enumerate(IMPLEMENTATION_ORDER):
        config = config_map[implementation]
        model = ContinuousIGAN(
            d_w=d_w,
            q_obs_density=q_obs_density,
            q_target_density=q_target_density,
            target_w_sampler=target_w_sampler,
            kappa=kappa,
            config=config,
        )
        model.fit(train_w, train_y, seed=seed_offset + 97 * (index + 1))
        models[implementation] = model
    return models


def continuous_variant_configs(outcome_lower: float, outcome_upper: float, seed: int) -> dict[str, ContinuousIGANConfig]:
    return {
        "voronoi": ContinuousIGANConfig(
            implementation="voronoi",
            latent_dim=2,
            hidden_dims_generator=(96, 96),
            hidden_dims_critic=(96, 96),
            batch_size=128,
            num_steps=280,
            critic_steps=4,
            resolution=(2, 2),
            num_experts=16,
            outcome_lower=outcome_lower,
            outcome_upper=outcome_upper,
            seed=seed + 1,
        ),
        "anisotropic": ContinuousIGANConfig(
            implementation="anisotropic",
            latent_dim=2,
            hidden_dims_generator=(96, 96),
            hidden_dims_critic=(96, 96),
            batch_size=128,
            num_steps=320,
            critic_steps=4,
            resolution=(2, 2),
            besov_weight=(0.6, 0.6),
            smoothness=(0.35, 0.35),
            outcome_lower=outcome_lower,
            outcome_upper=outcome_upper,
            seed=seed + 2,
        ),
        "kernel": ContinuousIGANConfig(
            implementation="kernel",
            latent_dim=2,
            hidden_dims_generator=(96, 96),
            hidden_dims_critic=(96, 96),
            batch_size=128,
            num_steps=320,
            critic_steps=4,
            resolution=(2, 2),
            besov_weight=(0.45, 0.45),
            smoothness=(0.35, 0.35),
            num_anchors=28,
            kernel="matern32",
            kernel_bandwidth=0.18,
            coefficient_l2_weight=5e-4,
            outcome_lower=outcome_lower,
            outcome_upper=outcome_upper,
            seed=seed + 3,
        ),
    }


def ganite_variant_configs(seed: int) -> dict[str, ContinuousIGANConfig]:
    return {
        "voronoi": ContinuousIGANConfig(
            implementation="voronoi",
            latent_dim=2,
            hidden_dims_generator=(80, 80),
            hidden_dims_critic=(80, 80),
            batch_size=128,
            num_steps=240,
            critic_steps=4,
            resolution=(2, 1),
            num_experts=8,
            outcome_lower=0.0,
            outcome_upper=1.0,
            seed=seed + 1,
        ),
        "anisotropic": ContinuousIGANConfig(
            implementation="anisotropic",
            latent_dim=2,
            hidden_dims_generator=(80, 80),
            hidden_dims_critic=(80, 80),
            batch_size=128,
            num_steps=280,
            critic_steps=4,
            resolution=(2, 1),
            besov_weight=(0.55, 0.02),
            smoothness=(0.35, 0.05),
            outcome_lower=0.0,
            outcome_upper=1.0,
            seed=seed + 2,
        ),
        "kernel": ContinuousIGANConfig(
            implementation="kernel",
            latent_dim=2,
            hidden_dims_generator=(80, 80),
            hidden_dims_critic=(80, 80),
            batch_size=128,
            num_steps=280,
            critic_steps=4,
            resolution=(2, 1),
            besov_weight=(0.35, 0.01),
            smoothness=(0.35, 0.05),
            num_anchors=20,
            kernel="matern32",
            kernel_bandwidth=0.16,
            coefficient_l2_weight=5e-4,
            outcome_lower=0.0,
            outcome_upper=1.0,
            seed=seed + 3,
        ),
    }


def scigan_variant_configs(seed: int) -> dict[str, ContinuousIGANConfig]:
    return {
        "voronoi": ContinuousIGANConfig(
            implementation="voronoi",
            latent_dim=2,
            hidden_dims_generator=(96, 96),
            hidden_dims_critic=(96, 96),
            batch_size=128,
            num_steps=260,
            critic_steps=4,
            resolution=(2, 1, 2),
            num_experts=16,
            outcome_lower=-0.5,
            outcome_upper=6.0,
            seed=1,
        ),
        "anisotropic": ContinuousIGANConfig(
            implementation="anisotropic",
            latent_dim=2,
            hidden_dims_generator=(96, 96),
            hidden_dims_critic=(96, 96),
            batch_size=128,
            num_steps=340,
            critic_steps=4,
            resolution=(2, 1, 2),
            besov_weight=(0.20, 0.0, 0.05),
            smoothness=(0.25, 0.01, 0.10),
            generator_transport_weight=1.5,
            outcome_lower=-0.5,
            outcome_upper=6.0,
            seed=2,
        ),
        "kernel": ContinuousIGANConfig(
            implementation="kernel",
            latent_dim=2,
            hidden_dims_generator=(96, 96),
            hidden_dims_critic=(96, 96),
            batch_size=128,
            num_steps=300,
            critic_steps=4,
            resolution=(2, 1, 2),
            besov_weight=(0.40, 0.01, 0.20),
            smoothness=(0.35, 0.05, 0.20),
            num_anchors=24,
            kernel="matern32",
            kernel_bandwidth=0.22,
            coefficient_l2_weight=7e-4,
            outcome_lower=-0.5,
            outcome_upper=6.0,
            seed=3,
        ),
    }


def run_finite_state(output_dir: Path, seed: int) -> dict[str, float]:
    dgp = FiniteStateCausalDGP()
    observed = dgp.sample_observed(n=2_400, seed=seed)
    model = FiniteStateIGAN(
        num_states=dgp.num_states,
        target_q=dgp.target_q,
        config=FiniteStateIGANConfig(
            latent_dim=dgp.latent_dim,
            hidden_dims_generator=(64, 64),
            hidden_dims_critic=(64, 64),
            batch_size=128,
            num_steps=320,
            critic_steps=4,
            min_state_samples=80,
            outcome_lower=-dgp.outcome_bound,
            outcome_upper=dgp.outcome_bound,
            seed=seed,
        ),
    )
    model.fit(observed["state"], observed["y"])

    true_means = np.array([dgp.true_state_mean(state) for state in range(dgp.num_states)], dtype=np.float64)
    est_means = model.estimated_state_means(n_mc=1_536)
    conditional_w1 = model.approximate_conditional_w1(
        true_sampler=lambda state, n: dgp.sample_state(state, n, seed=110_000 + state),
        n_per_state=768,
    )
    policy_value_true = float(np.dot(dgp.target_q, true_means))
    policy_value_est = float(np.dot(dgp.target_q, est_means))

    labels = [f"x={x}, t={t}" for x in dgp.x_support for t in dgp.treatments]
    save_grouped_bar_plot(
        output_dir / "finite_state_means.png",
        labels=labels,
        series=[
            ("true", true_means, "#1b9e77"),
            ("IGAN", est_means, IMPLEMENTATION_COLORS["voronoi"]),
        ],
        title="Finite-state IGAN: statewise interventional means",
        ylabel="conditional mean",
        rotation=0,
    )

    return {
        "conditional_w1": conditional_w1,
        "state_mean_l1": float(np.mean(np.abs(true_means - est_means))),
        "policy_value_true": policy_value_true,
        "policy_value_est": policy_value_est,
        "policy_value_abs_error": abs(policy_value_true - policy_value_est),
        "effective_sample_fraction": float(model.state_counts.min() / model.state_counts.sum()),
    }


def run_continuous_variants(output_dir: Path, seed: int) -> dict[str, dict[str, float]]:
    dgp = ContinuousCausalDGP()
    observed = dgp.sample_observed(n=2_600, seed=seed)
    models = fit_igan_variants(
        d_w=dgp.d_w,
        train_w=observed["w"],
        train_y=observed["y"],
        q_obs_density=dgp.q_obs_density,
        q_target_density=dgp.q_target_density,
        target_w_sampler=dgp.sample_target_w,
        kappa=dgp.kappa,
        config_map=continuous_variant_configs(-dgp.outcome_bound, dgp.outcome_bound, seed),
        seed_offset=seed + 1_000,
    )

    axes = [np.linspace(0.05, 0.95, 9, dtype=np.float32) for _ in range(dgp.d_w)]
    mesh = np.meshgrid(*axes, indexing="ij")
    grid = np.stack([axis.reshape(-1) for axis in mesh], axis=1)
    true_means = np.array(
        [dgp.sample_conditional(w, 1_024, seed=120_000 + idx).mean() for idx, w in enumerate(grid)],
        dtype=np.float64,
    )

    results: dict[str, dict[str, float]] = {}
    for idx, implementation in enumerate(IMPLEMENTATION_ORDER):
        model = models[implementation]
        predicted_means = model.predict_mean(grid, n_mc=384).reshape(-1).astype(np.float64)
        conditional_w1 = model.approximate_conditional_w1(
            true_sampler=lambda w, n: dgp.sample_conditional(w, n, seed=130_000 + int(1_000 * np.sum(w))),
            grid_size=7,
            n_per_w=384,
        )
        target_w = dgp.sample_target_w(2_048, seed=seed + 50 + idx)
        true_target_mean = np.array(
            [dgp.sample_conditional(w, 512, seed=140_000 + i).mean() for i, w in enumerate(target_w)],
            dtype=np.float64,
        )
        predicted_target_mean = model.predict_mean(target_w, n_mc=192).reshape(-1).astype(np.float64)
        results[implementation] = {
            "conditional_w1": conditional_w1,
            "grid_mean_l1": float(np.mean(np.abs(predicted_means - true_means))),
            "target_mean_l1": float(np.mean(np.abs(predicted_target_mean - true_target_mean))),
            "effective_sample_size": float(model.effective_sample_size or 0.0),
        }

    save_grouped_bar_plot(
        output_dir / "continuous_variants_metrics.png",
        labels=["conditional W1", "grid mean L1", "target mean L1"],
        series=[
            (
                IMPLEMENTATION_LABELS[implementation],
                np.asarray(
                    [
                        results[implementation]["conditional_w1"],
                        results[implementation]["grid_mean_l1"],
                        results[implementation]["target_mean_l1"],
                    ]
                ),
                IMPLEMENTATION_COLORS[implementation],
            )
            for implementation in IMPLEMENTATION_ORDER
        ],
        title="Continuous IGAN implementations: distributional recovery",
        ylabel="error",
        rotation=0,
    )
    return results


def igan_binary_potential_means(model: ContinuousIGAN, dgp: GANITEBenchmarkDGP, x: np.ndarray, n_mc: int = 256) -> np.ndarray:
    w0 = dgp.encode_w(x, np.zeros(x.shape[0], dtype=np.int64))
    w1 = dgp.encode_w(x, np.ones(x.shape[0], dtype=np.int64))
    mu0 = model.predict_mean(w0, n_mc=n_mc).reshape(-1)
    mu1 = model.predict_mean(w1, n_mc=n_mc).reshape(-1)
    return np.column_stack([mu0, mu1]).astype(np.float64)


def run_ganite_benchmark(output_dir: Path, seed: int) -> dict[str, dict[str, float]]:
    dgp = GANITEBenchmarkDGP()
    train = dgp.sample_observed(n=2_400, seed=seed)
    test_x = dgp.sample_x(256, seed=seed + 1)
    true_mu = dgp.potential_means(test_x).astype(np.float64)
    true_tau = true_mu[:, 1] - true_mu[:, 0]

    ganite = GANITE(
        GANITEConfig(
            x_dim=dgp.feature_dim,
            hidden_dim=64,
            batch_size=128,
            cf_iterations=350,
            ite_iterations=350,
            alpha=1.0,
            beta=2.0,
            outcome_min=0.0,
            outcome_max=1.0,
            seed=seed + 11,
        )
    )
    ganite.fit(train["x"], train["t"], train["y"])
    ganite_mu = ganite.predict_potential_outcomes(test_x).astype(np.float64)
    ganite_tau = ganite_mu[:, 1] - ganite_mu[:, 0]

    models = fit_igan_variants(
        d_w=dgp.d_w,
        train_w=train["w"],
        train_y=train["y"],
        q_obs_density=dgp.q_obs_density,
        q_target_density=dgp.q_target_density,
        target_w_sampler=dgp.sample_target_w,
        kappa=dgp.kappa,
        config_map=ganite_variant_configs(seed + 100),
        seed_offset=seed + 2_000,
    )

    results: dict[str, dict[str, float]] = {
        "ganite": {
            "pehe": float(np.mean((ganite_tau - true_tau) ** 2)),
            "ate_abs_error": float(abs(np.mean(ganite_tau) - np.mean(true_tau))),
            "mean_l1": float(np.mean(np.abs(ganite_mu - true_mu))),
        }
    }

    x_axis = test_x[:, 0]
    order = np.argsort(x_axis)
    plt.figure(figsize=(8.4, 4.4))
    plt.plot(x_axis[order], true_tau[order], label="true ITE", color="#1b9e77", linewidth=2.2)
    plt.plot(x_axis[order], ganite_tau[order], label="GANITE", color="#7570b3", linewidth=1.8)

    anchor_x = np.array([[0.15], [0.45], [0.75]], dtype=np.float32)
    anchor_true = dgp.potential_means(anchor_x).astype(np.float64)
    anchor_series: list[tuple[str, np.ndarray, str]] = [("true", anchor_true.reshape(-1), "#1b9e77")]
    anchor_series.append(("GANITE", ganite.predict_potential_outcomes(anchor_x).reshape(-1).astype(np.float64), "#7570b3"))

    for implementation in IMPLEMENTATION_ORDER:
        igan_mu = igan_binary_potential_means(models[implementation], dgp, test_x, n_mc=224)
        igan_tau = igan_mu[:, 1] - igan_mu[:, 0]
        results[implementation] = {
            "pehe": float(np.mean((igan_tau - true_tau) ** 2)),
            "ate_abs_error": float(abs(np.mean(igan_tau) - np.mean(true_tau))),
            "mean_l1": float(np.mean(np.abs(igan_mu - true_mu))),
        }
        plt.plot(
            x_axis[order],
            igan_tau[order],
            label=IMPLEMENTATION_LABELS[implementation],
            color=IMPLEMENTATION_COLORS[implementation],
            linewidth=1.8,
        )
        anchor_mu = igan_binary_potential_means(models[implementation], dgp, anchor_x, n_mc=320)
        anchor_series.append(
            (
                IMPLEMENTATION_LABELS[implementation],
                anchor_mu.reshape(-1),
                IMPLEMENTATION_COLORS[implementation],
            )
        )

    plt.xlabel("x")
    plt.ylabel("treatment effect")
    plt.title("GANITE benchmark: ITE recovery")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ganite_benchmark_ite_curve.png", dpi=180)
    plt.close()

    labels = [f"x={float(anchor_x[i, 0]):.2f}\nt={treatment}" for i in range(anchor_x.shape[0]) for treatment in range(2)]
    save_grouped_bar_plot(
        output_dir / "ganite_benchmark_means.png",
        labels=labels,
        series=anchor_series,
        title="GANITE benchmark: representative conditional means",
        ylabel="mean outcome",
        rotation=0,
    )
    return results


def scigan_response_metrics_from_grid(
    dgp: SCIGANBenchmarkDGP,
    x_test: np.ndarray,
    dosage_grid: np.ndarray,
    pred_grid: np.ndarray,
) -> dict[str, float]:
    n = x_test.shape[0]
    num_treatments = dgp.num_treatments
    true_grid = np.zeros_like(pred_grid)
    for treatment in range(num_treatments):
        for idx_d, dosage in enumerate(dosage_grid):
            true_grid[:, treatment, idx_d] = dgp.response_mean(x_test, treatment, dosage)

    mise = float(np.mean((pred_grid - true_grid) ** 2))
    dpe_terms: list[float] = []
    pe_terms: list[float] = []

    for i in range(n):
        true_best_values = []
        pred_best_values = []
        pred_best_dosages = []
        for treatment in range(num_treatments):
            pred_idx = int(np.argmax(pred_grid[i, treatment]))
            pred_best_d = float(dosage_grid[pred_idx])
            pred_best_dosages.append(pred_best_d)
            true_opt_d = float(dgp.optimal_dosage(x_test[i : i + 1], treatment)[0])
            true_at_true = float(dgp.response_mean(x_test[i : i + 1], treatment, true_opt_d)[0])
            true_at_pred = float(dgp.response_mean(x_test[i : i + 1], treatment, pred_best_d)[0])
            dpe_terms.append((true_at_true - true_at_pred) ** 2)
            true_best_values.append(true_at_true)
            pred_best_values.append(float(true_at_pred))
        best_true_t = int(np.argmax(true_best_values))
        best_pred_t = int(np.argmax(pred_best_values))
        best_pred_d = pred_best_dosages[best_pred_t]
        true_best = float(true_best_values[best_true_t])
        achieved = float(dgp.response_mean(x_test[i : i + 1], best_pred_t, best_pred_d)[0])
        pe_terms.append((true_best - achieved) ** 2)

    return {"mise": mise, "dpe": float(np.mean(dpe_terms)), "pe": float(np.mean(pe_terms))}


def build_igan_response_grid(
    model: ContinuousIGAN,
    dgp: SCIGANBenchmarkDGP,
    x_test: np.ndarray,
    dosage_grid: np.ndarray,
    n_mc: int = 160,
) -> np.ndarray:
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


def build_scigan_response_grid(
    model: SCIGAN,
    dgp: SCIGANBenchmarkDGP,
    x_test: np.ndarray,
    dosage_grid: np.ndarray,
) -> np.ndarray:
    grid = np.zeros((x_test.shape[0], dgp.num_treatments, dosage_grid.size), dtype=np.float64)
    for treatment in range(dgp.num_treatments):
        for idx_d, dosage in enumerate(dosage_grid):
            pred = model.predict_response(
                x_test,
                treatment=treatment,
                dosage=np.full(x_test.shape[0], dosage, dtype=np.float32),
            )
            grid[:, treatment, idx_d] = pred.reshape(-1).astype(np.float64)
    return grid


def run_scigan_benchmark(output_dir: Path, seed: int) -> dict[str, dict[str, float]]:
    dgp = SCIGANBenchmarkDGP(seed=seed)
    train = dgp.sample_observed(n=2_600, seed=seed + 1)
    test_x = dgp.sample_x(96, seed=seed + 2)
    dosage_grid = np.linspace(0.0, 1.0, 41, dtype=np.float32)

    scigan = SCIGAN(
        SCIGANConfig(
            x_dim=dgp.feature_dim,
            num_treatments=dgp.num_treatments,
            hidden_dim=64,
            set_dim=32,
            batch_size=128,
            gan_iterations=500,
            inference_iterations=750,
            num_dosage_samples=5,
            alpha=1.0,
            seed=seed + 7,
        )
    )
    scigan.fit(train["x"], train["t"], train["d"], train["y"])
    scigan_grid = build_scigan_response_grid(scigan, dgp, test_x, dosage_grid)

    models = fit_igan_variants(
        d_w=dgp.d_w,
        train_w=train["w"],
        train_y=train["y"],
        q_obs_density=dgp.q_obs_density,
        q_target_density=dgp.q_target_density,
        target_w_sampler=dgp.sample_target_w,
        kappa=dgp.kappa,
        config_map=scigan_variant_configs(seed + 100),
        seed_offset=500,
    )

    results: dict[str, dict[str, float]] = {"scigan": scigan_response_metrics_from_grid(dgp, test_x, dosage_grid, scigan_grid)}
    igan_grids: dict[str, np.ndarray] = {}
    for implementation in IMPLEMENTATION_ORDER:
        grid = build_igan_response_grid(models[implementation], dgp, test_x, dosage_grid, n_mc=128)
        igan_grids[implementation] = grid
        results[implementation] = scigan_response_metrics_from_grid(dgp, test_x, dosage_grid, grid)

    representative_x = np.array([[0.62]], dtype=np.float32)
    plt.figure(figsize=(9.2, 4.6))
    treatment_colors = ["#1b9e77", "#d95f02"]
    for treatment in range(dgp.num_treatments):
        true_curve = np.array(
            [dgp.response_mean(representative_x, treatment, dosage)[0] for dosage in dosage_grid],
            dtype=np.float64,
        )
        scigan_curve = np.array(
            [scigan.predict_response(representative_x, treatment=treatment, dosage=np.array([dosage], dtype=np.float32))[0, 0] for dosage in dosage_grid],
            dtype=np.float64,
        )
        plt.plot(dosage_grid, true_curve, color=treatment_colors[treatment], linewidth=2.2, label=f"true t={treatment}")
        plt.plot(dosage_grid, scigan_curve, color=treatment_colors[treatment], linestyle="--", linewidth=1.8, label=f"SCIGAN t={treatment}")
        for implementation in IMPLEMENTATION_ORDER:
            query = dgp.encode_w(
                np.repeat(representative_x, dosage_grid.size, axis=0),
                np.full(dosage_grid.size, treatment, dtype=np.int64),
                dosage_grid,
            )
            curve = models[implementation].predict_mean(query, n_mc=128).reshape(-1).astype(np.float64)
            plt.plot(
                dosage_grid,
                curve,
                color=IMPLEMENTATION_COLORS[implementation],
                linestyle={0: "-", 1: ":"}[treatment],
                linewidth=1.6,
                alpha=0.9,
                label=f"{IMPLEMENTATION_LABELS[implementation]} t={treatment}",
            )
    plt.xlabel("dosage")
    plt.ylabel("mean outcome")
    plt.title("SCIGAN benchmark: dose-response curves at x=0.62")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "scigan_benchmark_curves.png", dpi=180)
    plt.close()

    anchor_x = np.array([[0.15], [0.45], [0.75]], dtype=np.float32)
    labels: list[str] = []
    true_values: list[float] = []
    scigan_values: list[float] = []
    igan_values: dict[str, list[float]] = {implementation: [] for implementation in IMPLEMENTATION_ORDER}
    for treatment in range(dgp.num_treatments):
        opt_d = dgp.optimal_dosage(anchor_x, treatment).astype(np.float32)
        labels.extend(
            [f"x={float(anchor_x[i, 0]):.2f}\nt={treatment}, d*={float(opt_d[i]):.2f}" for i in range(anchor_x.shape[0])]
        )
        true_vals = dgp.response_mean(anchor_x, treatment, opt_d).astype(np.float64)
        scigan_vals = scigan.predict_response(anchor_x, treatment=treatment, dosage=opt_d).reshape(-1).astype(np.float64)
        query = dgp.encode_w(anchor_x, np.full(anchor_x.shape[0], treatment, dtype=np.int64), opt_d)
        true_values.extend(true_vals.tolist())
        scigan_values.extend(scigan_vals.tolist())
        for implementation in IMPLEMENTATION_ORDER:
            igan_pred = models[implementation].predict_mean(query, n_mc=192).reshape(-1).astype(np.float64)
            igan_values[implementation].extend(igan_pred.tolist())

    series = [
        ("true", np.asarray(true_values), "#1b9e77"),
        ("SCIGAN", np.asarray(scigan_values), "#7570b3"),
    ]
    series.extend(
        (
            IMPLEMENTATION_LABELS[implementation],
            np.asarray(igan_values[implementation]),
            IMPLEMENTATION_COLORS[implementation],
        )
        for implementation in IMPLEMENTATION_ORDER
    )
    save_grouped_bar_plot(
        output_dir / "scigan_benchmark_means.png",
        labels=labels,
        series=series,
        title="SCIGAN benchmark: means at true optimal dosages",
        ylabel="mean outcome",
        rotation=0,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IGAN sanity experiments and baseline comparisons.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/igan_experiment"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    metrics = {
        "finite_state": run_finite_state(output_dir, args.seed),
        "continuous_variants": run_continuous_variants(output_dir, args.seed + 1_000),
        "ganite_benchmark": run_ganite_benchmark(output_dir, args.seed + 2_000),
        "scigan_benchmark": run_scigan_benchmark(output_dir, args.seed + 3_000),
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
