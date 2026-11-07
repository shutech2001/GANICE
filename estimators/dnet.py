from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch.nn import functional as F


def _as_covariates(x: NDArray, *, x_dim: int) -> NDArray[np.float32]:
    """Return a covariate matrix without mistaking one multivariate row for a column."""

    array = np.asarray(x, dtype=np.float32)
    if array.ndim == 1:
        if x_dim == 1:
            array = array[:, None]
        elif array.size == x_dim:
            array = array[None, :]
        else:
            raise ValueError(f"one-dimensional x must contain exactly {x_dim} features")
    if array.ndim != 2 or array.shape[1] != x_dim:
        raise ValueError(f"x must have shape (n, {x_dim})")
    return array


def _strict_treatment(values: NDArray | int, *, n_rows: int, num_treatments: int) -> NDArray[np.int64]:
    """Validate treatment labels before casting so fractional labels cannot be truncated."""

    raw = np.asarray(values)
    if raw.ndim > 2 or (raw.ndim == 2 and raw.shape[1] != 1):
        raise ValueError("treatment must be scalar or one-dimensional")
    raw = raw.reshape(-1)
    if raw.size == 1 and n_rows > 1:
        raw = np.full(n_rows, raw.item())
    if raw.size != n_rows:
        raise ValueError("treatment must be scalar or align with x")
    valid_labels = np.arange(num_treatments)
    if not np.all(np.isin(raw, valid_labels)):
        raise ValueError(f"treatment values must be integer labels in [0, {num_treatments})")
    return raw.astype(np.int64)


def _mlp(input_dim: int, hidden_dims: Tuple[int, ...], output_dim: Optional[int] = None) -> nn.Sequential:
    layers: List[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        linear = nn.Linear(previous, width)
        nn.init.xavier_uniform_(linear.weight)
        nn.init.zeros_(linear.bias)
        layers.extend([linear, nn.ELU()])
        previous = width
    if output_dim is not None:
        output = nn.Linear(previous, output_dim)
        nn.init.xavier_uniform_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
    return nn.Sequential(*layers)


class NonCrossingQuantileLayer(nn.Module):
    """NCQ layer in Eq. (2) of Wu et al. (2023)."""

    centering_weights: torch.Tensor

    def __init__(self, input_dim: int, num_quantiles: int, initial_increment: float) -> None:
        super().__init__()
        self.num_quantiles = int(num_quantiles)
        self.value = nn.Linear(input_dim, 1)
        self.raw_increments = nn.Linear(input_dim, num_quantiles)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.zeros_(self.value.bias)
        nn.init.zeros_(self.raw_increments.weight)
        # ELU(c) + 1 = exp(c) for c < 0.  A small initial spacing avoids
        # beginning with a K-wide outcome range while retaining positivity.
        nn.init.constant_(self.raw_increments.bias, float(np.log(max(initial_increment, 1e-6))))
        centering_weights = torch.arange(num_quantiles, 0, -1, dtype=torch.float32) / float(num_quantiles)
        self.register_buffer("centering_weights", centering_weights)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = self.value(features)
        increments = F.elu(self.raw_increments(features)) + 1.0
        cumulative = torch.cumsum(increments, dim=1)
        center = torch.sum(increments * self.centering_weights.reshape(1, -1), dim=1, keepdim=True)
        return value + cumulative - center


class _DNetModule(nn.Module):
    def __init__(
        self,
        x_dim: int,
        num_treatments: int,
        base_hidden_dims: Tuple[int, ...],
        tower_hidden_dims: Tuple[int, ...],
        num_quantiles: int,
        initial_increment: float,
    ) -> None:
        super().__init__()
        if not base_hidden_dims:
            raise ValueError("base_hidden_dims cannot be empty")
        self.base = _mlp(x_dim, base_hidden_dims)
        base_dim = base_hidden_dims[-1]
        self.treatment_tower = nn.Linear(base_dim, num_treatments)
        nn.init.xavier_uniform_(self.treatment_tower.weight)
        nn.init.zeros_(self.treatment_tower.bias)
        self.outcome_towers = nn.ModuleList()
        self.ncq_layers = nn.ModuleList()
        for _ in range(num_treatments):
            tower = _mlp(base_dim, tower_hidden_dims)
            tower_dim = tower_hidden_dims[-1] if tower_hidden_dims else base_dim
            self.outcome_towers.append(tower)
            self.ncq_layers.append(NonCrossingQuantileLayer(tower_dim, num_quantiles, initial_increment))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.base(x)
        treatment_logits = self.treatment_tower(representation)
        quantiles = [
            ncq(tower(representation)) for tower, ncq in zip(self.outcome_towers, self.ncq_layers, strict=True)
        ]
        return torch.stack(quantiles, dim=1), treatment_logits


@dataclass(slots=True)
class DNetConfig:
    x_dim: int
    num_treatments: int = 2
    base_hidden_dims: Tuple[int, ...] = (200, 100)
    tower_hidden_dims: Tuple[int, ...] = (100,)
    num_quantiles: int = 50
    batch_size: int = 128
    training_steps: int = 1_000
    learning_rate: float = 1e-3
    # Explicit L2 coefficient for the treatment/action tower kernel only.
    weight_decay: float = 1e-5
    propensity_weight: float = 1.0
    huber_kappa: float = 1.0
    initial_increment: float = 0.02
    outcome_min: Optional[float] = None
    outcome_max: Optional[float] = None
    seed: int = 223
    device: str = "cpu"


@dataclass(slots=True)
class DNetDiagnostics:
    losses: list[float] = field(default_factory=list)
    outcome_losses: list[float] = field(default_factory=list)
    propensity_losses: list[float] = field(default_factory=list)
    treatment_l2_losses: list[float] = field(default_factory=list)
    quantile_crossing_rate: float = 0.0


class DNet:
    """Deep distributional treatment-effect network of Wu et al. (2023).

    The implementation follows the paper's shared BaseNet, treatment-assignment
    tower, treatment-specific response towers, and non-crossing quantile (NCQ)
    output layer for a scalar outcome and a finite number of discrete
    treatments.  The reported
    conditional mean is exactly the average of the K learned quantiles.

    Wu et al. do not define an outcome sampler between or beyond those finite
    quantile levels.  For the repository-wide distributional metrics,
    ``sample_potential`` therefore uses the explicit finite-grid empirical-law
    adapter that assigns probability ``1 / K`` to each learned quantile.  Its
    expectation equals the paper's quantile-average prediction exactly and it
    does not invent unestimated tails.
    """

    def __init__(self, config: DNetConfig) -> None:
        if config.x_dim < 1:
            raise ValueError("x_dim must be positive")
        if config.num_treatments < 2:
            raise ValueError("num_treatments must be at least two")
        if not config.base_hidden_dims or any(width < 1 for width in config.base_hidden_dims):
            raise ValueError("base_hidden_dims must contain positive widths")
        if any(width < 1 for width in config.tower_hidden_dims):
            raise ValueError("tower_hidden_dims must contain positive widths")
        if config.num_quantiles < 2:
            raise ValueError("num_quantiles must be at least two")
        if config.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if config.training_steps < 1:
            raise ValueError("training_steps must be positive")
        if not np.isfinite(config.learning_rate) or config.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        if not np.isfinite(config.weight_decay) or config.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative and finite")
        if not np.isfinite(config.propensity_weight) or config.propensity_weight < 0.0:
            raise ValueError("propensity_weight must be non-negative and finite")
        if not np.isfinite(config.huber_kappa) or config.huber_kappa < 0.0:
            raise ValueError("huber_kappa must be non-negative and finite")
        if not np.isfinite(config.initial_increment) or config.initial_increment <= 0.0:
            raise ValueError("initial_increment must be positive and finite")
        if (config.outcome_min is None) != (config.outcome_max is None):
            raise ValueError("outcome_min and outcome_max must either both be set or both be None")
        if (
            config.outcome_min is not None
            and config.outcome_max is not None
            and config.outcome_max <= config.outcome_min
        ):
            raise ValueError("outcome_max must be greater than outcome_min")
        self.config = config
        torch.manual_seed(config.seed)
        torch.set_num_threads(1)
        self.device = torch.device(config.device)
        self.model = _DNetModule(
            x_dim=config.x_dim,
            num_treatments=config.num_treatments,
            base_hidden_dims=config.base_hidden_dims,
            tower_hidden_dims=config.tower_hidden_dims,
            num_quantiles=config.num_quantiles,
            initial_increment=config.initial_increment,
        ).to(self.device)
        # The paper reports L2 regularization for the action/T-tower.  Apply it
        # explicitly in the loss rather than decoupled AdamW decay over every
        # parameter in the network.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        levels = np.arange(1, config.num_quantiles + 1, dtype=np.float32) / float(config.num_quantiles + 1)
        self.quantile_levels = levels
        self._levels_tensor = torch.as_tensor(levels, device=self.device).reshape(1, -1)
        self.diagnostics = DNetDiagnostics()
        self._is_fitted = False

    def _quantile_loss(self, residual: torch.Tensor) -> torch.Tensor:
        weights = torch.abs(self._levels_tensor - (residual < 0.0).to(residual.dtype))
        if self.config.huber_kappa > 0.0:
            absolute = torch.abs(residual)
            kappa = float(self.config.huber_kappa)
            huber = torch.where(absolute <= kappa, 0.5 * residual.square(), kappa * (absolute - 0.5 * kappa))
            return torch.mean(weights * huber / kappa)
        return torch.mean(weights * torch.abs(residual))

    def _treatment_l2_penalty(self) -> torch.Tensor:
        return float(self.config.weight_decay) * self.model.treatment_tower.weight.square().sum()

    def fit(self, x: NDArray, treatment: NDArray, y: NDArray) -> "DNet":
        x_array = _as_covariates(x, x_dim=self.config.x_dim)
        treatment_array = _strict_treatment(
            treatment,
            n_rows=x_array.shape[0],
            num_treatments=self.config.num_treatments,
        )
        raw_y = np.asarray(y, dtype=np.float32)
        if raw_y.ndim > 2 or (raw_y.ndim == 2 and raw_y.shape[1] != 1):
            raise ValueError("DNet requires a scalar outcome")
        y_array = raw_y.reshape(-1, 1)
        if x_array.shape[0] != treatment_array.shape[0] or x_array.shape[0] != y_array.shape[0]:
            raise ValueError("x, treatment, and y must have the same number of rows")
        observed_arms = np.unique(treatment_array)
        expected_arms = np.arange(self.config.num_treatments, dtype=np.int64)
        if not np.array_equal(observed_arms, expected_arms):
            raise ValueError("every configured discrete treatment arm must be present")
        if not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(y_array)):
            raise ValueError("x and y values must all be finite")

        x_tensor = torch.as_tensor(x_array, device=self.device)
        treatment_tensor = torch.as_tensor(treatment_array, dtype=torch.long, device=self.device)
        y_tensor = torch.as_tensor(y_array, device=self.device)
        n = x_tensor.shape[0]
        batch_size = min(self.config.batch_size, n)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.config.seed + 1)
        self.diagnostics = DNetDiagnostics()
        self.model.train()
        for step in range(self.config.training_steps):
            indices = torch.randint(0, n, size=(batch_size,), generator=generator, device=self.device)
            quantiles, treatment_logits = self.model(x_tensor[indices])
            factual = quantiles[torch.arange(batch_size, device=self.device), treatment_tensor[indices]]
            residual = y_tensor[indices] - factual
            outcome_loss = self._quantile_loss(residual)
            propensity_loss = F.cross_entropy(treatment_logits, treatment_tensor[indices])
            treatment_l2 = self._treatment_l2_penalty()
            loss = (
                outcome_loss
                + self.config.propensity_weight * propensity_loss
                + treatment_l2
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            if step % 25 == 0 or step == self.config.training_steps - 1:
                self.diagnostics.losses.append(float(loss.detach().cpu().item()))
                self.diagnostics.outcome_losses.append(float(outcome_loss.detach().cpu().item()))
                self.diagnostics.propensity_losses.append(float(propensity_loss.detach().cpu().item()))
                self.diagnostics.treatment_l2_losses.append(
                    float(treatment_l2.detach().cpu().item())
                )

        self._is_fitted = True
        predicted = self.predict_quantiles(x_array[: min(512, n)])
        self.diagnostics.quantile_crossing_rate = float(np.mean(np.diff(predicted, axis=2) < -1e-7))
        return self

    def predict_quantiles(self, x: NDArray) -> NDArray[np.float32]:
        if not self._is_fitted:
            raise RuntimeError("DNet must be fit before prediction")
        x_array = _as_covariates(x, x_dim=self.config.x_dim)
        self.model.eval()
        with torch.no_grad():
            quantiles = self.model(torch.as_tensor(x_array, device=self.device))[0].cpu().numpy()
        self.model.train()
        if self.config.outcome_min is not None:
            quantiles = np.clip(quantiles, self.config.outcome_min, self.config.outcome_max)
        return quantiles.astype(np.float32)

    def predict_potential_outcomes(self, x: NDArray) -> NDArray[np.float32]:
        return cast(
            NDArray[np.float32],
            np.mean(self.predict_quantiles(x), axis=2, dtype=np.float64).astype(np.float32),
        )

    def sample_potential(
        self,
        x: NDArray,
        treatment: NDArray | int,
        n_per_x: int = 1,
        seed: Optional[int] = None,
    ) -> NDArray[np.float32]:
        if isinstance(n_per_x, (bool, np.bool_)) or not isinstance(
            n_per_x,
            (int, np.integer),
        ):
            raise TypeError("n_per_x must be an integer")
        if n_per_x < 1:
            raise ValueError("n_per_x must be positive")
        quantiles = self.predict_quantiles(x)
        n = quantiles.shape[0]
        treatment_array = _strict_treatment(
            treatment,
            n_rows=n,
            num_treatments=self.config.num_treatments,
        )
        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        grid_indices = rng.integers(
            0,
            self.config.num_quantiles,
            size=(n, int(n_per_x)),
        )
        samples = np.empty((n, int(n_per_x)), dtype=np.float64)
        for row in range(n):
            arm_quantiles = quantiles[row, treatment_array[row]].astype(np.float64)
            samples[row] = arm_quantiles[grid_indices[row]]
        if self.config.outcome_min is not None:
            samples = np.clip(samples, self.config.outcome_min, self.config.outcome_max)
        return samples[:, :, None].astype(np.float32)
