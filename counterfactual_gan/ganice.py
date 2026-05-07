from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn

from .metrics import continuous_conditional_w1_grid
from .neural import ActivationName, AdversarialDiagnostics, AnchoredOutcomeCritic, BoundedMLP, build_mlp, sample_latent
from .utils import ensure_2d, ensure_row_matrix


def _resolution_levels(resolution: int | Tuple[int, ...], dim: int) -> NDArray[np.int64]:
    if isinstance(resolution, tuple):
        if len(resolution) != dim:
            raise ValueError("resolution must have length d_cell_w")
        levels = np.asarray(resolution, dtype=np.int64)
    else:
        levels = np.full(dim, int(resolution), dtype=np.int64)
    if np.any(levels < 0):
        raise ValueError("resolution levels must be nonnegative")
    return levels


def _num_cells(levels: NDArray[np.int64]) -> int:
    return int(np.prod(2**levels))


def _cell_indices(w: NDArray[np.float32], levels: NDArray[np.int64]) -> NDArray[np.int64]:
    w_arr = ensure_row_matrix(w)
    sides = 2**levels
    clipped = np.minimum(np.maximum(w_arr, 0.0), np.nextafter(1.0, 0.0))
    grid = np.floor(clipped * sides.reshape(1, -1)).astype(np.int64)
    multipliers = np.ones(levels.shape[0], dtype=np.int64)
    for coord in range(1, levels.shape[0]):
        multipliers[coord] = multipliers[coord - 1] * sides[coord - 1]
    return (grid * multipliers.reshape(1, -1)).sum(axis=1).astype(np.int64)


def _cell_midpoints(cell_ids: NDArray[np.int64], levels: NDArray[np.int64]) -> NDArray[np.float32]:
    ids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
    sides = 2**levels
    coords = np.zeros((ids.shape[0], levels.shape[0]), dtype=np.float32)
    residual = ids.copy()
    for coord, side in enumerate(sides):
        coords[:, coord] = (residual % side).astype(np.float32)
        residual //= side
    return (coords + 0.5) / sides.reshape(1, -1)


def _cell_transport_loss(cells: torch.Tensor, real_y: torch.Tensor, fake_y: torch.Tensor) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    total = float(cells.shape[0])
    for cell in cells.unique(sorted=True).tolist():
        mask = cells == cell
        if int(mask.sum().item()) < 2:
            continue
        real_state = torch.sort(real_y[mask].reshape(-1))[0]
        fake_state = torch.sort(fake_y[mask].reshape(-1))[0]
        losses.append((float(mask.sum().item()) / total) * torch.mean(torch.abs(real_state - fake_state)))
    if not losses:
        return torch.zeros((), device=real_y.device, dtype=real_y.dtype)
    return torch.stack(losses).sum()


class _CellConditionalGenerator(nn.Module):
    def __init__(
        self,
        num_cells: int,
        w_dim: int,
        latent_dim: int,
        y_dim: int,
        hidden_dims: Tuple[int, ...],
        outcome_lower: float,
        outcome_upper: float,
        activation: ActivationName,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.y_dim = int(y_dim)
        self.experts = nn.ModuleList(
            [
                BoundedMLP(
                    input_dim=w_dim + latent_dim,
                    hidden_dims=hidden_dims,
                    output_dim=y_dim,
                    lower=outcome_lower,
                    upper=outcome_upper,
                    activation=activation,
                )
                for _ in range(num_cells)
            ]
        )

    def forward(self, w: torch.Tensor, cells: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        outputs = torch.zeros(w.shape[0], self.y_dim, device=w.device, dtype=w.dtype)
        features = torch.cat([w, z], dim=1)
        for cell in cells.unique(sorted=True).tolist():
            mask = cells == cell
            outputs[mask] = self.experts[int(cell)](features[mask])
        return outputs


class _SharedConditionalGenerator(nn.Module):
    def __init__(
        self,
        w_dim: int,
        latent_dim: int,
        y_dim: int,
        hidden_dims: Tuple[int, ...],
        outcome_lower: float,
        outcome_upper: float,
        activation: ActivationName,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.y_dim = int(y_dim)
        self.network = BoundedMLP(
            input_dim=w_dim + latent_dim,
            hidden_dims=hidden_dims,
            output_dim=y_dim,
            lower=outcome_lower,
            upper=outcome_upper,
            activation=activation,
        )

    def forward(self, w: torch.Tensor, cells: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        del cells
        return self.network(torch.cat([w, z], dim=1))


class _CellOutcomeCritic(nn.Module):
    def __init__(
        self,
        num_cells: int,
        y_dim: int,
        hidden_dims: Tuple[int, ...],
        anchor: NDArray[np.float32],
        activation: ActivationName,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [
                AnchoredOutcomeCritic(
                    y_dim=y_dim,
                    hidden_dims=hidden_dims,
                    anchor=anchor,
                    activation=activation,
                )
                for _ in range(num_cells)
            ]
        )

    def forward(self, cells: torch.Tensor, y: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        del w
        outputs = torch.zeros(y.shape[0], 1, device=y.device, dtype=y.dtype)
        for cell in cells.unique(sorted=True).tolist():
            mask = cells == cell
            outputs[mask] = self.experts[int(cell)](y[mask])
        return outputs


class _CellContextOutcomeCritic(nn.Module):
    def __init__(
        self,
        num_cells: int,
        w_dim: int,
        y_dim: int,
        hidden_dims: Tuple[int, ...],
        anchor: NDArray[np.float32],
        activation: ActivationName,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [build_mlp(w_dim + y_dim, hidden_dims, 1, activation=activation) for _ in range(num_cells)]
        )
        self.register_buffer("anchor", torch.as_tensor(anchor.reshape(1, -1), dtype=torch.float32))

    def forward(self, cells: torch.Tensor, y: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        if w is None:
            raise ValueError("context critic requires w")
        outputs = torch.zeros(y.shape[0], 1, device=y.device, dtype=y.dtype)
        anchor_y = self.anchor.expand(y.shape[0], -1)
        for cell in cells.unique(sorted=True).tolist():
            mask = cells == cell
            features = torch.cat([w[mask], y[mask]], dim=1)
            anchor_features = torch.cat([w[mask], anchor_y[mask]], dim=1)
            expert = self.experts[int(cell)]
            outputs[mask] = expert(features) - expert(anchor_features)
        return outputs


def _cell_gradient_penalty(
    critic: _CellOutcomeCritic | _CellContextOutcomeCritic,
    cells: torch.Tensor,
    real_y: torch.Tensor,
    fake_y: torch.Tensor,
    w: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    alpha = torch.rand(real_y.shape[0], 1, device=real_y.device, dtype=real_y.dtype)
    alpha = alpha.expand_as(real_y)
    interpolated = alpha * real_y + (1.0 - alpha) * fake_y
    interpolated.requires_grad_(True)
    scores = critic(cells, interpolated, w)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.shape[0], -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


@dataclass(slots=True)
class GANICEConfig:
    latent_dim: int = 2
    hidden_dims_generator: Tuple[int, ...] = (64, 64)
    hidden_dims_critic: Tuple[int, ...] = (64, 64)
    batch_size: int = 128
    num_steps: int = 420
    critic_steps: int = 4
    generator_lr: float = 2e-4
    critic_lr: float = 1e-4
    betas: Tuple[float, float] = (0.0, 0.9)
    gradient_penalty_weight: float = 10.0
    generator_transport_weight: float = 4.0
    resolution: int | Tuple[int, ...] = 2
    min_cell_samples: int = 8
    target_mass_samples: int = 20_000
    cell_normalized: bool = True
    critic_uses_w: bool = False
    factual_loss_weight: float = 0.0
    factual_crps_weight: float = 0.0
    factual_crps_samples: int = 4
    factual_mse_weight: float = 0.0
    factual_mse_samples: int = 4
    residual_quantile_calibration: bool = False
    calibration_samples_per_observation: int = 8
    calibration_grid_size: int = 256
    calibration_blend: float = 1.0
    pretrain_steps: int = 0
    pretrain_mse_weight: float = 1.0
    shared_generator: bool = False
    outcome_lower: float = -3.0
    outcome_upper: float = 3.0
    critic_anchor: float = 0.0
    activation: ActivationName = "elu"
    max_predict_batch: int = 65_536
    device: str = "cpu"
    seed: int = 123


class GANICE:
    """Finite-resolution, cell-normalized conditional WGAN for GANICE.

    The training minibatch samples cells according to the estimated target cell
    masses q_C, then samples observed outcomes uniformly inside the chosen cell.
    This implements the stratified objective from the GANICE draft without
    estimating or plugging in w_rho.
    """

    def __init__(
        self,
        d_w: int,
        target_w_sampler,
        config: GANICEConfig,
        *,
        d_cell_w: Optional[int] = None,
        cell_transform=None,
    ) -> None:
        self.d_w = int(d_w)
        self.d_cell_w = int(self.d_w if d_cell_w is None else d_cell_w)
        self.target_w_sampler = target_w_sampler
        self.cell_transform = cell_transform
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        torch.set_num_threads(1)
        self.levels = _resolution_levels(config.resolution, self.d_cell_w)
        self.num_cells = _num_cells(self.levels)
        self.generator: _CellConditionalGenerator | _SharedConditionalGenerator | None = None
        self.critic: _CellOutcomeCritic | _CellContextOutcomeCritic | None = None
        self.generator_optimizer: Optional[torch.optim.Optimizer] = None
        self.critic_optimizer: Optional[torch.optim.Optimizer] = None
        self.y_dim: Optional[int] = None
        self.target_q: Optional[NDArray[np.float64]] = None
        self.training_target_q: Optional[NDArray[np.float64]] = None
        self.observed_counts: Optional[NDArray[np.int64]] = None
        self.active_cells: Optional[NDArray[np.int64]] = None
        self.cell_alias: Optional[NDArray[np.int64]] = None
        self.target_mass_coverage: float = 0.0
        self.residual_calibration: Dict[int, Tuple[NDArray[np.float32], NDArray[np.float32]]] = {}
        self.diagnostics = AdversarialDiagnostics()

    def _build_networks(self, y_dim: int) -> None:
        anchor = np.full((y_dim,), self.config.critic_anchor, dtype=np.float32)
        if self.config.shared_generator:
            self.generator = _SharedConditionalGenerator(
                w_dim=self.d_w,
                latent_dim=self.config.latent_dim,
                y_dim=y_dim,
                hidden_dims=self.config.hidden_dims_generator,
                outcome_lower=self.config.outcome_lower,
                outcome_upper=self.config.outcome_upper,
                activation=self.config.activation,
            ).to(self.device)
        else:
            self.generator = _CellConditionalGenerator(
                num_cells=self.num_cells,
                w_dim=self.d_w,
                latent_dim=self.config.latent_dim,
                y_dim=y_dim,
                hidden_dims=self.config.hidden_dims_generator,
                outcome_lower=self.config.outcome_lower,
                outcome_upper=self.config.outcome_upper,
                activation=self.config.activation,
            ).to(self.device)
        if self.config.critic_uses_w:
            self.critic = _CellContextOutcomeCritic(
                num_cells=self.num_cells,
                w_dim=self.d_w,
                y_dim=y_dim,
                hidden_dims=self.config.hidden_dims_critic,
                anchor=anchor,
                activation=self.config.activation,
            ).to(self.device)
        else:
            self.critic = _CellOutcomeCritic(
                num_cells=self.num_cells,
                y_dim=y_dim,
                hidden_dims=self.config.hidden_dims_critic,
                anchor=anchor,
                activation=self.config.activation,
            ).to(self.device)
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

    def _cell_coordinates(self, w: NDArray[np.float32]) -> NDArray[np.float32]:
        w_arr = ensure_row_matrix(w)
        if self.cell_transform is None:
            cell_w = w_arr
        else:
            cell_w = ensure_row_matrix(self.cell_transform(w_arr))
        if cell_w.shape[0] != w_arr.shape[0]:
            raise ValueError("cell_transform must preserve the number of rows")
        if cell_w.shape[1] != self.d_cell_w:
            raise ValueError(f"cell_transform returned {cell_w.shape[1]} columns, expected {self.d_cell_w}")
        return cell_w.astype(np.float32)

    def _cell_ids(self, w: NDArray[np.float32]) -> NDArray[np.int64]:
        return _cell_indices(self._cell_coordinates(w), self.levels)

    def _target_cell_masses(
        self, seed: Optional[int] = None
    ) -> Tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.int64]]:
        target_w = ensure_row_matrix(self.target_w_sampler(self.config.target_mass_samples, seed=seed))
        target_cells = self._cell_ids(target_w)
        counts = np.bincount(target_cells, minlength=self.num_cells).astype(np.float64)
        q = counts / np.maximum(counts.sum(), 1.0)
        return q, target_w.astype(np.float32), target_cells.astype(np.int64)

    def _alias_target_cells(
        self, target_q: NDArray[np.float64], observed_counts: NDArray[np.int64]
    ) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Route target-relevant sparse cells to nearby observed cells.

        The paper objective aggregates all target cell masses q_C.  In finite
        samples, some high-resolution cells can have too few observed outcomes
        for stable adversarial training.  Instead of dropping that target mass
        and renormalizing the estimand, we merge each sparse target cell into
        the nearest target-relevant cell with enough observations.
        """

        target_support = np.flatnonzero(target_q > 0.0).astype(np.int64)
        active_cells = target_support[observed_counts[target_support] >= self.config.min_cell_samples]
        if active_cells.size == 0:
            fallback = target_support[observed_counts[target_support] > 0]
            if fallback.size == 0:
                raise ValueError("no dyadic cell has both target mass and observed outcomes")
            active_cells = fallback.astype(np.int64)

        alias = np.full(self.num_cells, -1, dtype=np.int64)
        alias[active_cells] = active_cells
        inactive = target_support[alias[target_support] < 0]
        if inactive.size > 0:
            active_midpoints = _cell_midpoints(active_cells, self.levels)
            inactive_midpoints = _cell_midpoints(inactive, self.levels)
            for idx, cell in enumerate(inactive.tolist()):
                nearest = int(np.argmin(np.sum((active_midpoints - inactive_midpoints[idx: idx + 1]) ** 2, axis=1)))
                alias[int(cell)] = int(active_cells[nearest])

        routed_q = np.zeros(self.num_cells, dtype=np.float64)
        for cell in target_support.tolist():
            routed = int(alias[int(cell)])
            if routed >= 0:
                routed_q[routed] += float(target_q[int(cell)])
        active_cells = np.flatnonzero(routed_q > 0.0).astype(np.int64)
        return alias, active_cells

    def _sample_cells(self, q_active: torch.Tensor, active_cells: torch.Tensor, batch_size: int) -> torch.Tensor:
        draws = torch.multinomial(q_active, batch_size, replacement=True)
        return active_cells[draws]

    def _sample_from_cell_lists(
        self,
        cell_batch: torch.Tensor,
        index_by_cell: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        sampled = torch.empty(cell_batch.shape[0], dtype=torch.long, device=self.device)
        for cell in cell_batch.unique(sorted=True).tolist():
            positions = torch.nonzero(cell_batch == cell, as_tuple=False).reshape(-1)
            candidates = index_by_cell[int(cell)]
            draws = torch.randint(0, candidates.shape[0], size=(positions.shape[0],), device=self.device)
            sampled[positions] = candidates[draws]
        return sampled

    def _fit_residual_quantile_calibration(
        self,
        w_t: torch.Tensor,
        y_t: torch.Tensor,
        obs_index_by_cell: Dict[int, torch.Tensor],
        active_cells: NDArray[np.int64],
        seed: int,
    ) -> None:
        self.residual_calibration = {}
        if self.generator is None or self.y_dim != 1 or not self.config.residual_quantile_calibration:
            return
        k = max(2, int(self.config.calibration_samples_per_observation))
        grid_size = max(16, int(self.config.calibration_grid_size))
        self.generator.eval()
        with torch.no_grad():
            for cell in active_cells.tolist():
                idx = obs_index_by_cell[int(cell)]
                if idx.numel() < 3:
                    continue
                w_cell = w_t[idx]
                n_cell = int(w_cell.shape[0])
                repeated_w = w_cell.repeat_interleave(k, dim=0)
                repeated_cells = torch.full((n_cell * k,), int(cell), dtype=torch.long, device=self.device)
                fake = (
                    self.generator(
                        repeated_w,
                        repeated_cells,
                        sample_latent(n_cell * k, self.config.latent_dim, self.device, seed=seed + int(cell)),
                    )
                    .reshape(n_cell, k)
                    .cpu()
                    .numpy()
                )
                centers = fake.mean(axis=1)
                fake_residual = (fake - centers[:, None]).reshape(-1)
                real_residual = y_t[idx].reshape(-1).cpu().numpy() - centers
                if np.std(fake_residual) < 1e-8 or np.std(real_residual) < 1e-8:
                    continue
                probs = (np.arange(grid_size, dtype=np.float64) + 0.5) / grid_size
                source = np.quantile(fake_residual, probs)
                target = np.quantile(real_residual, probs)
                source_unique, unique_idx = np.unique(source, return_index=True)
                if source_unique.size < 3:
                    continue
                self.residual_calibration[int(cell)] = (
                    source_unique.astype(np.float32),
                    target[unique_idx].astype(np.float32),
                )
        self.generator.train()

    def _apply_residual_quantile_calibration(self, samples: NDArray[np.float32], cell: int) -> NDArray[np.float32]:
        if self.y_dim != 1 or not self.residual_calibration:
            return samples
        calibration = self.residual_calibration.get(int(cell))
        if calibration is None:
            return samples
        source, target = calibration
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        center = float(values.mean())
        mapped_residual = np.interp(values - center, source, target, left=target[0], right=target[-1])
        calibrated = center + mapped_residual
        blend = float(np.clip(self.config.calibration_blend, 0.0, 1.0))
        values = (1.0 - blend) * values + blend * calibrated.astype(np.float32)
        return values.reshape(samples.shape).astype(np.float32)

    def fit(self, w_obs: NDArray[np.float32], outcomes: NDArray[np.float32], seed: Optional[int] = None) -> "GANICE":
        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        w_arr = ensure_row_matrix(w_obs)
        y_arr = ensure_2d(outcomes)
        if w_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("w_obs and outcomes must have the same number of rows")

        obs_cells = self._cell_ids(w_arr)
        observed_counts = np.bincount(obs_cells, minlength=self.num_cells).astype(np.int64)
        target_q, target_pool_w, target_pool_cells = self._target_cell_masses(seed=int(rng.integers(0, 2**31 - 1)))
        cell_alias, active_cells = self._alias_target_cells(target_q, observed_counts)
        routed_target_pool_cells = target_pool_cells.copy()
        target_pool_mask = target_pool_cells >= 0
        routed_target_pool_cells[target_pool_mask] = cell_alias[target_pool_cells[target_pool_mask]]
        routed_counts = np.bincount(
            routed_target_pool_cells[routed_target_pool_cells >= 0],
            minlength=self.num_cells,
        ).astype(np.float64)
        routed_q = routed_counts / np.maximum(routed_counts.sum(), 1.0)
        self.target_mass_coverage = float(target_q[cell_alias >= 0].sum())
        target_q_active = routed_q[active_cells] / np.maximum(routed_q[active_cells].sum(), 1e-12)
        self.target_q = target_q
        self.training_target_q = routed_q
        self.observed_counts = observed_counts
        self.active_cells = active_cells
        self.cell_alias = cell_alias
        self.y_dim = y_arr.shape[1]
        self._build_networks(self.y_dim)
        if self.generator is None or self.critic is None:
            raise RuntimeError("networks were not initialized")
        if self.generator_optimizer is None or self.critic_optimizer is None:
            raise RuntimeError("optimizers were not initialized")

        w_t = torch.as_tensor(w_arr, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y_arr, dtype=torch.float32, device=self.device)
        target_w_t = torch.as_tensor(target_pool_w, dtype=torch.float32, device=self.device)
        obs_cells_t = torch.as_tensor(obs_cells, dtype=torch.long, device=self.device)
        target_pool_cells_t = torch.as_tensor(routed_target_pool_cells, dtype=torch.long, device=self.device)
        active_cells_t = torch.as_tensor(active_cells, dtype=torch.long, device=self.device)
        q_active_t = torch.as_tensor(target_q_active, dtype=torch.float32, device=self.device)

        obs_index_by_cell = {
            int(cell): torch.nonzero(obs_cells_t == int(cell), as_tuple=False).reshape(-1)
            for cell in active_cells.tolist()
        }
        target_index_by_cell = {
            int(cell): torch.nonzero(target_pool_cells_t == int(cell), as_tuple=False).reshape(-1)
            for cell in active_cells.tolist()
        }
        for cell, target_indices in list(target_index_by_cell.items()):
            if target_indices.numel() == 0:
                target_index_by_cell[cell] = obs_index_by_cell[cell]
        active_obs_indices = torch.cat([obs_index_by_cell[int(cell)] for cell in active_cells.tolist()])
        active_target_indices = torch.cat([target_index_by_cell[int(cell)] for cell in active_cells.tolist()])

        batch_size = min(self.config.batch_size, int(sum(observed_counts[active_cells])))
        self.diagnostics = AdversarialDiagnostics()
        for _ in range(max(0, int(self.config.pretrain_steps))):
            obs_draws = torch.randint(0, active_obs_indices.shape[0], size=(batch_size,), device=self.device)
            obs_idx = active_obs_indices[obs_draws]
            real_w = w_t[obs_idx]
            real_cells = obs_cells_t[obs_idx]
            real_y = y_t[obs_idx]
            if self.y_dim == 1:
                k = max(2, int(self.config.factual_crps_samples))
                repeated_w = real_w.repeat_interleave(k, dim=0)
                repeated_cells = real_cells.repeat_interleave(k)
                generated = self.generator(
                    repeated_w,
                    repeated_cells,
                    sample_latent(batch_size * k, self.config.latent_dim, self.device),
                ).reshape(batch_size, k)
                real_values = real_y.reshape(batch_size, 1)
                energy_to_observed = torch.mean(torch.abs(generated - real_values))
                pairwise_spread = torch.mean(torch.abs(generated[:, :, None] - generated[:, None, :]))
                crps_loss = energy_to_observed - 0.5 * pairwise_spread
                mean_loss = torch.mean((generated.mean(dim=1, keepdim=True) - real_y) ** 2)
                pretrain_loss = crps_loss + self.config.pretrain_mse_weight * mean_loss
            else:
                generated = self.generator(
                    real_w,
                    real_cells,
                    sample_latent(batch_size, self.config.latent_dim, self.device),
                )
                pretrain_loss = torch.mean((generated - real_y) ** 2)
            self.generator_optimizer.zero_grad(set_to_none=True)
            pretrain_loss.backward()
            self.generator_optimizer.step()

        for step in range(self.config.num_steps):
            critic_loss_value = 0.0
            objective_gap = 0.0
            for _ in range(self.config.critic_steps):
                if self.config.cell_normalized:
                    real_cells = self._sample_cells(q_active_t, active_cells_t, batch_size)
                    fake_cells = real_cells
                    obs_idx = self._sample_from_cell_lists(real_cells, obs_index_by_cell)
                    target_idx = self._sample_from_cell_lists(fake_cells, target_index_by_cell)
                    real_y = y_t[obs_idx]
                    fake_w = target_w_t[target_idx]
                else:
                    obs_draws = torch.randint(0, active_obs_indices.shape[0], size=(batch_size,), device=self.device)
                    target_draws = torch.randint(
                        0,
                        active_target_indices.shape[0],
                        size=(batch_size,),
                        device=self.device,
                    )
                    obs_idx = active_obs_indices[obs_draws]
                    target_idx = active_target_indices[target_draws]
                    real_cells = obs_cells_t[obs_idx]
                    fake_cells = target_pool_cells_t[target_idx]
                    real_y = y_t[obs_idx]
                    fake_w = target_w_t[target_idx]

                fake_y = self.generator(
                    fake_w,
                    fake_cells,
                    sample_latent(batch_size, self.config.latent_dim, self.device),
                ).detach()
                penalty_w = w_t[obs_idx]
                penalty_fake = self.generator(
                    penalty_w,
                    real_cells,
                    sample_latent(batch_size, self.config.latent_dim, self.device),
                ).detach()

                real_w = w_t[obs_idx]
                real_score = self.critic(real_cells, real_y, real_w)
                fake_score = self.critic(fake_cells, fake_y, fake_w)
                objective = real_score.mean() - fake_score.mean()
                gp = _cell_gradient_penalty(self.critic, real_cells, real_y, penalty_fake, real_w)
                critic_loss = -objective + self.config.gradient_penalty_weight * gp

                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                self.critic_optimizer.step()

                critic_loss_value = float(critic_loss.detach().cpu().item())
                objective_gap = float(objective.detach().cpu().item())

            if self.config.cell_normalized:
                cell_batch = self._sample_cells(q_active_t, active_cells_t, batch_size)
                target_idx = self._sample_from_cell_lists(cell_batch, target_index_by_cell)
            else:
                target_draws = torch.randint(0, active_target_indices.shape[0], size=(batch_size,), device=self.device)
                target_idx = active_target_indices[target_draws]
                cell_batch = target_pool_cells_t[target_idx]
            fake_w = target_w_t[target_idx]
            fake_y = self.generator(
                fake_w,
                cell_batch,
                sample_latent(batch_size, self.config.latent_dim, self.device),
            )
            generator_loss = -self.critic(cell_batch, fake_y, fake_w).mean()
            if self.y_dim == 1 and self.config.generator_transport_weight > 0.0:
                obs_idx = self._sample_from_cell_lists(cell_batch, obs_index_by_cell)
                real_y = y_t[obs_idx]
                real_w = w_t[obs_idx]
                fake_same = self.generator(
                    real_w,
                    cell_batch,
                    sample_latent(batch_size, self.config.latent_dim, self.device),
                )
                generator_loss = generator_loss + self.config.generator_transport_weight * _cell_transport_loss(
                    cell_batch,
                    real_y,
                    fake_same,
                )
            if self.config.factual_loss_weight > 0.0:
                obs_idx = self._sample_from_cell_lists(cell_batch, obs_index_by_cell)
                real_y = y_t[obs_idx]
                real_w = w_t[obs_idx]
                fake_factual = self.generator(
                    real_w,
                    cell_batch,
                    sample_latent(batch_size, self.config.latent_dim, self.device),
                )
                generator_loss = generator_loss + self.config.factual_loss_weight * torch.mean(
                    torch.abs(fake_factual - real_y)
                )
            if self.config.factual_mse_weight > 0.0:
                obs_idx = self._sample_from_cell_lists(cell_batch, obs_index_by_cell)
                real_y = y_t[obs_idx]
                real_w = w_t[obs_idx]
                k = max(1, int(self.config.factual_mse_samples))
                repeated_w = real_w.repeat_interleave(k, dim=0)
                repeated_cells = cell_batch.repeat_interleave(k)
                generated = self.generator(
                    repeated_w,
                    repeated_cells,
                    sample_latent(batch_size * k, self.config.latent_dim, self.device),
                ).reshape(batch_size, k, self.y_dim or 1)
                generator_loss = generator_loss + self.config.factual_mse_weight * torch.mean(
                    (generated.mean(dim=1) - real_y) ** 2
                )
            if self.y_dim == 1 and self.config.factual_crps_weight > 0.0:
                obs_idx = self._sample_from_cell_lists(cell_batch, obs_index_by_cell)
                real_y = y_t[obs_idx]
                real_w = w_t[obs_idx]
                k = max(2, int(self.config.factual_crps_samples))
                repeated_w = real_w.repeat_interleave(k, dim=0)
                repeated_cells = cell_batch.repeat_interleave(k)
                generated = self.generator(
                    repeated_w,
                    repeated_cells,
                    sample_latent(batch_size * k, self.config.latent_dim, self.device),
                ).reshape(batch_size, k)
                real_values = real_y.reshape(batch_size, 1)
                energy_to_observed = torch.mean(torch.abs(generated - real_values))
                pairwise_spread = torch.mean(torch.abs(generated[:, :, None] - generated[:, None, :]))
                crps_loss = energy_to_observed - 0.5 * pairwise_spread
                generator_loss = generator_loss + self.config.factual_crps_weight * crps_loss

            self.generator_optimizer.zero_grad(set_to_none=True)
            generator_loss.backward()
            self.generator_optimizer.step()

            if step % 10 == 0 or step == self.config.num_steps - 1:
                self.diagnostics.critic_losses.append(critic_loss_value)
                self.diagnostics.generator_losses.append(float(generator_loss.detach().cpu().item()))
                self.diagnostics.objective_gaps.append(objective_gap)
        self._fit_residual_quantile_calibration(
            w_t,
            y_t,
            obs_index_by_cell,
            active_cells,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        return self

    def _route_query_cells(self, queries: NDArray[np.float32]) -> NDArray[np.int64]:
        cells = self._cell_ids(queries)
        if self.active_cells is None:
            raise ValueError("model has not been fitted")
        if self.cell_alias is not None:
            aliased = self.cell_alias[cells]
            alias_mask = aliased >= 0
            cells = cells.copy()
            cells[alias_mask] = aliased[alias_mask]
        active_set = set(int(cell) for cell in self.active_cells.tolist())
        if all(int(cell) in active_set for cell in cells):
            return cells
        active_midpoints = _cell_midpoints(self.active_cells, self.levels)
        query_midpoints = _cell_midpoints(cells, self.levels)
        routed = cells.copy()
        for idx, cell in enumerate(cells):
            if int(cell) in active_set:
                continue
            nearest = int(np.argmin(np.sum((active_midpoints - query_midpoints[idx: idx + 1]) ** 2, axis=1)))
            routed[idx] = int(self.active_cells[nearest])
        return routed

    def sample_conditional(self, w: NDArray[np.float32], n: int, seed: Optional[int] = None) -> NDArray[np.float32]:
        if self.generator is None:
            raise ValueError("model has not been fitted")
        query = ensure_row_matrix(w)
        if query.shape[0] != 1:
            raise ValueError("sample_conditional expects a single conditioning point")
        cell = self._route_query_cells(query)
        repeated_w = np.repeat(query, n, axis=0)
        repeated_cells = np.repeat(cell, n)
        self.generator.eval()
        with torch.no_grad():
            w_t = torch.as_tensor(repeated_w, dtype=torch.float32, device=self.device)
            cells_t = torch.as_tensor(repeated_cells, dtype=torch.long, device=self.device)
            samples = (
                self.generator(
                    w_t,
                    cells_t,
                    sample_latent(n, self.config.latent_dim, self.device, seed=seed),
                )
                .cpu()
                .numpy()
            )
        self.generator.train()
        samples = self._apply_residual_quantile_calibration(samples, int(cell[0]))
        return samples.astype(np.float32)

    def predict_mean(self, queries: NDArray[np.float32], n_mc: int = 1024) -> NDArray[np.float32]:
        if self.generator is None:
            raise ValueError("model has not been fitted")
        query_arr = ensure_row_matrix(queries)
        cells = self._route_query_cells(query_arr)
        total = query_arr.shape[0] * n_mc
        outputs: List[NDArray[np.float32]] = []
        self.generator.eval()
        with torch.no_grad():
            start = 0
            while start < total:
                stop = min(total, start + self.config.max_predict_batch)
                flat_indices = np.arange(start, stop, dtype=np.int64) // n_mc
                w_chunk = query_arr[flat_indices]
                cell_chunk = cells[flat_indices]
                w_t = torch.as_tensor(w_chunk, dtype=torch.float32, device=self.device)
                cells_t = torch.as_tensor(cell_chunk, dtype=torch.long, device=self.device)
                samples = (
                    self.generator(
                        w_t,
                        cells_t,
                        sample_latent(w_t.shape[0], self.config.latent_dim, self.device, seed=70_000 + start),
                    )
                    .cpu()
                    .numpy()
                )
                outputs.append(samples)
                start = stop
        self.generator.train()
        stacked = np.concatenate(outputs, axis=0).reshape(query_arr.shape[0], n_mc, self.y_dim or 1)
        if self.y_dim == 1 and self.residual_calibration:
            for row_idx, cell in enumerate(cells.tolist()):
                stacked[row_idx] = self._apply_residual_quantile_calibration(stacked[row_idx], int(cell))
        return stacked.mean(axis=1).astype(np.float32)

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
