from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from .igan_core import (
    AnchoredConditionalCritic,
    ConditionalGenerator,
    IGANDiagnostics,
    KernelOutcomeCritic,
    VoronoiCritic,
    VoronoiGenerator,
    coefficient_besov_penalty,
    finite_difference_besov_penalty,
    outcome_gradient_penalty,
    sample_latent,
)
from .metrics import continuous_conditional_w1_grid
from .utils import ensure_2d, ensure_row_matrix


ImplementationName = Literal["voronoi", "anisotropic", "kernel"]


def _expand_scalar_or_tuple(value: float | int | tuple[float, ...], dim: int, name: str) -> np.ndarray:
    if isinstance(value, tuple):
        if len(value) != dim:
            raise ValueError(f"{name} must have length {dim}")
        array = np.asarray(value, dtype=np.float32)
    else:
        array = np.full(dim, float(value), dtype=np.float32)
    return array


def _resolution_to_taus(resolution: int | tuple[int, ...], dim: int) -> np.ndarray:
    if isinstance(resolution, tuple):
        if len(resolution) != dim:
            raise ValueError(f"resolution must have length {dim}")
        levels = np.asarray(resolution, dtype=np.int64)
    else:
        levels = np.full(dim, int(resolution), dtype=np.int64)
    if np.any(levels < 0):
        raise ValueError("resolution levels must be nonnegative")
    return (2.0 ** (-levels)).astype(np.float32)


def _resolution_levels(resolution: int | tuple[int, ...], dim: int) -> np.ndarray:
    if isinstance(resolution, tuple):
        if len(resolution) != dim:
            raise ValueError(f"resolution must have length {dim}")
        levels = np.asarray(resolution, dtype=np.int64)
    else:
        levels = np.full(dim, int(resolution), dtype=np.int64)
    if np.any(levels < 0):
        raise ValueError("resolution levels must be nonnegative")
    return levels


def _dyadic_cell_transport_loss(
    *,
    w: torch.Tensor,
    real_y: torch.Tensor,
    fake_y: torch.Tensor,
    resolution_levels: np.ndarray,
) -> torch.Tensor:
    side_lengths = torch.as_tensor(2**resolution_levels, dtype=torch.long, device=w.device)
    clipped = torch.clamp(w, 0.0, torch.nextafter(torch.tensor(1.0, device=w.device), torch.tensor(0.0, device=w.device)))
    grid = torch.floor(clipped * side_lengths.to(dtype=w.dtype)).to(dtype=torch.long)
    multipliers = torch.ones_like(side_lengths)
    for coord in range(1, side_lengths.shape[0]):
        multipliers[coord] = multipliers[coord - 1] * side_lengths[coord - 1]
    cells = torch.sum(grid * multipliers.unsqueeze(0), dim=1)

    losses: list[torch.Tensor] = []
    total = float(w.shape[0])
    for cell in cells.unique(sorted=True).tolist():
        mask = cells == cell
        count = int(mask.sum().item())
        if count < 2:
            continue
        real_state = torch.sort(real_y[mask].reshape(-1))[0]
        fake_state = torch.sort(fake_y[mask].reshape(-1))[0]
        losses.append((count / total) * torch.mean(torch.abs(real_state - fake_state)))
    if not losses:
        return torch.zeros((), device=w.device, dtype=real_y.dtype)
    return torch.stack(losses).sum()


@dataclass(slots=True)
class ContinuousIGANConfig:
    implementation: ImplementationName = "voronoi"
    latent_dim: int = 2
    hidden_dims_generator: tuple[int, ...] = (96, 96)
    hidden_dims_critic: tuple[int, ...] = (96, 96)
    batch_size: int = 128
    num_steps: int = 420
    critic_steps: int = 4
    generator_lr: float = 2e-4
    critic_lr: float = 1e-4
    betas: tuple[float, float] = (0.0, 0.9)
    gradient_penalty_weight: float = 10.0
    generator_transport_weight: float = 2.5
    besov_weight: float | tuple[float, ...] = 1.0
    smoothness: float | tuple[float, ...] = 0.2
    besov_p: float = 2.0
    resolution: int | tuple[int, ...] = 1
    num_experts: int | None = None
    num_anchors: int = 24
    kernel: Literal["laplace", "matern12", "matern32"] = "matern32"
    kernel_bandwidth: float = 0.2
    coefficient_l2_weight: float = 1e-3
    outcome_lower: float = -3.0
    outcome_upper: float = 3.0
    critic_anchor: float = 0.0
    activation: str = "elu"
    device: str = "cpu"
    seed: int = 123


class ContinuousIGAN:
    def __init__(
        self,
        d_w: int,
        q_obs_density,
        q_target_density,
        target_w_sampler,
        kappa: float,
        config: ContinuousIGANConfig,
    ) -> None:
        self.d_w = int(d_w)
        self.q_obs_density = q_obs_density
        self.q_target_density = q_target_density
        self.target_w_sampler = target_w_sampler
        self.kappa = float(kappa)
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        torch.set_num_threads(1)
        self.taus = _resolution_to_taus(config.resolution, self.d_w)
        self.levels = _resolution_levels(config.resolution, self.d_w)
        self.besov_weights = _expand_scalar_or_tuple(config.besov_weight, self.d_w, "besov_weight")
        self.smoothness = _expand_scalar_or_tuple(config.smoothness, self.d_w, "smoothness")
        self.generator = None
        self.critic = None
        self.generator_optimizer = None
        self.critic_optimizer = None
        self.weights: np.ndarray | None = None
        self.y_dim: int | None = None
        self.diagnostics = IGANDiagnostics()
        self.effective_sample_size: float | None = None
        self.retained_count: int = 0

    def _num_experts(self) -> int:
        if self.config.num_experts is not None:
            return int(self.config.num_experts)
        levels = np.rint(-np.log2(self.taus)).astype(np.int64)
        return int(2 ** int(levels.sum()))

    def _build_networks(self, y_dim: int) -> None:
        anchor = np.full((y_dim,), self.config.critic_anchor, dtype=np.float32)
        if self.config.implementation == "voronoi":
            self.generator = VoronoiGenerator(
                w_dim=self.d_w,
                latent_dim=self.config.latent_dim,
                y_dim=y_dim,
                num_experts=self._num_experts(),
                hidden_dims=self.config.hidden_dims_generator,
                lower=self.config.outcome_lower,
                upper=self.config.outcome_upper,
                activation=self.config.activation,
            ).to(self.device)
            self.critic = VoronoiCritic(
                w_dim=self.d_w,
                y_dim=y_dim,
                num_experts=self._num_experts(),
                hidden_dims=self.config.hidden_dims_critic,
                anchor=anchor,
                activation=self.config.activation,
            ).to(self.device)
        elif self.config.implementation == "anisotropic":
            self.generator = ConditionalGenerator(
                w_dim=self.d_w,
                latent_dim=self.config.latent_dim,
                y_dim=y_dim,
                hidden_dims=self.config.hidden_dims_generator,
                lower=self.config.outcome_lower,
                upper=self.config.outcome_upper,
                activation=self.config.activation,
            ).to(self.device)
            self.critic = AnchoredConditionalCritic(
                w_dim=self.d_w,
                y_dim=y_dim,
                hidden_dims=self.config.hidden_dims_critic,
                anchor=anchor,
                activation=self.config.activation,
            ).to(self.device)
        elif self.config.implementation == "kernel":
            self.generator = ConditionalGenerator(
                w_dim=self.d_w,
                latent_dim=self.config.latent_dim,
                y_dim=y_dim,
                hidden_dims=self.config.hidden_dims_generator,
                lower=self.config.outcome_lower,
                upper=self.config.outcome_upper,
                activation=self.config.activation,
            ).to(self.device)
            self.critic = KernelOutcomeCritic(
                w_dim=self.d_w,
                y_dim=y_dim,
                num_anchors=self.config.num_anchors,
                hidden_dims=self.config.hidden_dims_critic,
                lower=self.config.outcome_lower,
                upper=self.config.outcome_upper,
                anchor=anchor,
                activation=self.config.activation,
                kernel=self.config.kernel,
                bandwidth=self.config.kernel_bandwidth,
            ).to(self.device)
        else:
            raise ValueError(f"unknown implementation {self.config.implementation}")

        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.generator_lr,
            betas=self.config.betas,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config.critic_lr,
            betas=self.config.betas,
        )

    def _sample_target_w_tensor(self, batch_size: int, seed: int | None = None) -> torch.Tensor:
        draws = self.target_w_sampler(batch_size, seed=seed)
        return torch.as_tensor(ensure_row_matrix(draws), dtype=torch.float32, device=self.device)

    def fit(self, w_obs: np.ndarray, outcomes: np.ndarray, seed: int | None = None) -> "ContinuousIGAN":
        rng = np.random.default_rng(seed)
        w_arr = ensure_row_matrix(w_obs)
        y_arr = ensure_2d(outcomes)
        if w_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("w_obs and outcomes must have the same number of rows")
        weights = self.q_target_density(w_arr) / self.q_obs_density(w_arr)
        if np.any(self.kappa * weights > 1.0 + 1e-6):
            raise ValueError("kappa is too large for the requested density ratio")

        self.weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        retained = rng.binomial(1, np.clip(self.kappa * self.weights, 0.0, 1.0), size=w_arr.shape[0]).astype(bool)
        self.retained_count = int(retained.sum())
        if self.retained_count < max(32, self.config.batch_size // 2):
            raise ValueError("too few retained observations after thinning")
        self.effective_sample_size = float(self.retained_count)
        self.y_dim = y_arr.shape[1]
        self._build_networks(self.y_dim)

        w_t = torch.as_tensor(w_arr[retained], dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y_arr[retained], dtype=torch.float32, device=self.device)
        taus_t = torch.as_tensor(self.taus, dtype=torch.float32, device=self.device)
        smoothness_t = torch.as_tensor(self.smoothness, dtype=torch.float32, device=self.device)
        besov_weights_t = torch.as_tensor(self.besov_weights, dtype=torch.float32, device=self.device)
        batch_size = min(self.config.batch_size, w_t.shape[0])
        self.diagnostics = IGANDiagnostics()

        for step in range(self.config.num_steps):
            critic_loss_value = 0.0
            objective_gap = 0.0
            for _ in range(self.config.critic_steps):
                idx = torch.randint(0, w_t.shape[0], size=(batch_size,), device=self.device)
                w_batch = w_t[idx]
                y_batch = y_t[idx]

                w_target = self._sample_target_w_tensor(
                    batch_size=batch_size,
                    seed=int(rng.integers(0, 2**31 - 1)),
                )
                fake_target = self.generator(
                    w_target,
                    sample_latent(batch_size, self.config.latent_dim, self.device),
                ).detach()
                fake_penalty = self.generator(
                    w_batch,
                    sample_latent(batch_size, self.config.latent_dim, self.device),
                ).detach()

                real_score = self.critic(w_batch, y_batch)
                fake_score = self.critic(w_target, fake_target)
                objective = real_score.mean() - fake_score.mean()

                critic_loss = -objective
                if self.config.implementation in {"voronoi", "anisotropic"}:
                    gp = outcome_gradient_penalty(self.critic, w_batch, y_batch, fake_penalty)
                    critic_loss = critic_loss + self.config.gradient_penalty_weight * gp
                if self.config.implementation == "anisotropic":
                    penalty_w = torch.cat([w_batch, w_target], dim=0)
                    penalty_y = torch.cat([y_batch, fake_target], dim=0)
                    coord_penalty = finite_difference_besov_penalty(
                        self.critic,
                        penalty_w,
                        penalty_y,
                        taus=taus_t,
                        smoothness=smoothness_t,
                        p=self.config.besov_p,
                    )
                    critic_loss = critic_loss + torch.dot(besov_weights_t, coord_penalty)
                if self.config.implementation == "kernel":
                    coeff_penalty = coefficient_besov_penalty(
                        self.critic.coefficient_map,
                        torch.cat([w_batch, w_target], dim=0),
                        taus=taus_t,
                        smoothness=smoothness_t,
                        p=self.config.besov_p,
                    )
                    coeff_norm = self.critic.coefficient_map(w_target).pow(2).mean()
                    critic_loss = critic_loss + torch.dot(besov_weights_t, coeff_penalty)
                    critic_loss = critic_loss + self.config.coefficient_l2_weight * coeff_norm

                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                self.critic_optimizer.step()

                critic_loss_value = float(critic_loss.detach().cpu().item())
                objective_gap = float(objective.detach().cpu().item())

            w_target = self._sample_target_w_tensor(
                batch_size=batch_size,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            fake = self.generator(
                w_target,
                sample_latent(batch_size, self.config.latent_dim, self.device),
            )
            generator_loss = -self.critic(w_target, fake).mean()
            if self.y_dim == 1 and self.config.generator_transport_weight > 0.0:
                idx = torch.randint(0, w_t.shape[0], size=(batch_size,), device=self.device)
                w_same = w_t[idx]
                real_same = y_t[idx]
                fake_same = self.generator(
                    w_same,
                    sample_latent(batch_size, self.config.latent_dim, self.device),
                )
                generator_loss = generator_loss + self.config.generator_transport_weight * _dyadic_cell_transport_loss(
                    w=w_same,
                    real_y=real_same,
                    fake_y=fake_same,
                    resolution_levels=self.levels,
                )

            self.generator_optimizer.zero_grad(set_to_none=True)
            generator_loss.backward()
            self.generator_optimizer.step()

            if step % 10 == 0 or step == self.config.num_steps - 1:
                self.diagnostics.critic_losses.append(critic_loss_value)
                self.diagnostics.generator_losses.append(float(generator_loss.detach().cpu().item()))
                self.diagnostics.objective_gaps.append(objective_gap)
        return self

    def sample_conditional(self, w: np.ndarray, n: int, seed: int | None = None) -> np.ndarray:
        if self.generator is None:
            raise ValueError("model has not been fitted")
        query = ensure_row_matrix(w)
        if query.shape[0] != 1:
            raise ValueError("sample_conditional expects a single conditioning point")
        repeated = np.repeat(query, n, axis=0)
        w_t = torch.as_tensor(repeated, dtype=torch.float32, device=self.device)
        self.generator.eval()
        with torch.no_grad():
            samples = self.generator(
                w_t,
                sample_latent(n, self.config.latent_dim, self.device, seed=seed),
            ).cpu().numpy()
        self.generator.train()
        return samples.astype(np.float32)

    def predict_mean(self, queries: np.ndarray, n_mc: int = 1024) -> np.ndarray:
        query_arr = ensure_row_matrix(queries)
        if self.generator is None:
            raise ValueError("model has not been fitted")
        repeated = np.repeat(query_arr, n_mc, axis=0)
        w_t = torch.as_tensor(repeated, dtype=torch.float32, device=self.device)
        self.generator.eval()
        with torch.no_grad():
            samples = self.generator(
                w_t,
                sample_latent(w_t.shape[0], self.config.latent_dim, self.device, seed=70_000),
            )
            samples = samples.reshape(query_arr.shape[0], n_mc, self.y_dim or 1).mean(dim=1).cpu().numpy()
        self.generator.train()
        return samples.astype(np.float32)

    def approximate_conditional_w1(
        self,
        true_sampler,
        grid_size: int = 7,
        n_per_w: int = 512,
    ) -> float:
        axes = [np.linspace(0.05, 0.95, grid_size, dtype=np.float32) for _ in range(self.d_w)]
        mesh = np.meshgrid(*axes, indexing="ij")
        w_grid = np.stack([axis.reshape(-1) for axis in mesh], axis=1)
        return continuous_conditional_w1_grid(
            w_grid=w_grid,
            true_sampler=true_sampler,
            learned_sampler=lambda w, n: self.sample_conditional(w, n, seed=90_000 + int(1_000 * np.sum(w))),
            n_per_w=n_per_w,
        )
