from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch.nn import functional as F

from .neural import ActivationName


class TruncatedPowerBasis(nn.Module):
    """Truncated power spline basis on [0, 1] used by VCNet."""

    def __init__(self, degree: int, knots: Tuple[float, ...]) -> None:
        super().__init__()
        if not isinstance(degree, int) or degree < 1:
            raise ValueError("degree must be a positive integer")
        self.degree = degree
        self.num_basis = degree + 1 + len(knots)
        self.register_buffer("knots", torch.as_tensor(knots, dtype=torch.float32))

    def forward(self, treatment: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(treatment.reshape(-1), 0.0, 1.0)
        pieces = [torch.ones_like(t)]
        for power in range(1, self.degree + 1):
            pieces.append(t**power)
        for knot in self.knots:
            pieces.append(F.relu(t - knot) ** self.degree)
        return torch.stack(pieces, dim=1)


class VaryingCoefficientLinear(nn.Module):
    """Linear layer whose weights are spline functions of the treatment."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        basis: TruncatedPowerBasis,
        activation: ActivationName = "relu",
    ) -> None:
        super().__init__()
        self.basis = basis
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim, basis.num_basis))
        self.bias = nn.Parameter(torch.zeros(output_dim, basis.num_basis))
        if activation == "relu":
            self.activation: nn.Module | None = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "id":
            self.activation = None
        else:
            raise ValueError(f"unsupported activation: {activation}")
        nn.init.normal_(self.weight, mean=0.0, std=(2.0 / max(input_dim, 1)) ** 0.5 / basis.num_basis)

    def forward(self, treatment: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        basis_values = self.basis(treatment)
        feature_weight = torch.einsum("bi,iok->bok", features, self.weight)
        out = torch.einsum("bok,bk->bo", feature_weight, basis_values)
        out = out + torch.matmul(basis_values, self.bias.T)
        if self.activation is not None:
            out = self.activation(out)
        return out


class DensityHead(nn.Module):
    """Piecewise-linear conditional density surrogate for treatment in [0, 1]."""

    def __init__(self, input_dim: int, num_grid: int) -> None:
        super().__init__()
        if num_grid < 1:
            raise ValueError("num_grid must be positive")
        self.num_grid = int(num_grid)
        self.linear = nn.Linear(input_dim, self.num_grid + 1)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, treatment: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        grid_probs = torch.softmax(self.linear(features), dim=1)
        scaled = torch.clamp(treatment.reshape(-1), 0.0, 1.0) * self.num_grid
        left = torch.floor(scaled).long().clamp(0, self.num_grid)
        right = torch.ceil(scaled).long().clamp(0, self.num_grid)
        frac = (scaled - left.to(dtype=scaled.dtype)).reshape(-1)
        row = torch.arange(features.shape[0], device=features.device)
        interp = (1.0 - frac) * grid_probs[row, left] + frac * grid_probs[row, right]
        return torch.clamp(interp * (self.num_grid + 1), min=1e-6)


class VCNetModule(nn.Module):
    def __init__(self, x_dim: int, hidden_dim: int, num_grid: int, degree: int, knots: Tuple[float, ...]) -> None:
        super().__init__()
        self.hidden_features = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        for module in self.hidden_features:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        self.density_head = DensityHead(hidden_dim, num_grid)
        basis = TruncatedPowerBasis(degree, knots)
        self.q_layers = nn.ModuleList(
            [
                VaryingCoefficientLinear(hidden_dim, hidden_dim, basis, activation="relu"),
                VaryingCoefficientLinear(hidden_dim, 1, basis, activation="id"),
            ]
        )

    def forward(self, treatment: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.hidden_features(x)
        density = self.density_head(treatment, hidden)
        out = hidden
        for layer in self.q_layers:
            out = layer(treatment, out)
        return density, out


class TargetedRegularizer(nn.Module):
    def __init__(self, degree: int, knots: Tuple[float, ...]) -> None:
        super().__init__()
        self.basis = TruncatedPowerBasis(degree, knots)
        self.weight = nn.Parameter(torch.zeros(self.basis.num_basis))

    def forward(self, treatment: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.basis(treatment), self.weight)


@dataclass(slots=True)
class VCNetConfig:
    x_dim: int
    hidden_dim: int = 50
    num_grid: int = 10
    spline_degree: int = 2
    spline_knots: Tuple[float, ...] = (1.0 / 3.0, 2.0 / 3.0)
    batch_size: int = 128
    num_steps: int = 1_200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    density_loss_weight: float = 0.5
    standardize_outcome: bool = True
    targeted_regularization: bool = False
    tr_degree: int = 2
    tr_knots: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    tr_weight: float = 1.0
    tr_learning_rate: float = 1e-3
    max_predict_batch: int = 65_536
    seed: int = 23
    device: str = "cpu"


class VCNet:
    """VCNet baseline for continuous treatments in [0, 1].

    For SCIGAN-style benchmarks with a discrete treatment and continuous dosage,
    pass the dosage as the continuous treatment and include the discrete
    treatment indicator in x, typically as a one-hot feature.
    """

    def __init__(self, config: VCNetConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        torch.set_num_threads(1)
        self.device = torch.device(config.device)
        self.model = VCNetModule(
            x_dim=config.x_dim,
            hidden_dim=config.hidden_dim,
            num_grid=config.num_grid,
            degree=config.spline_degree,
            knots=config.spline_knots,
        ).to(self.device)
        self.targeted_regularizer = (
            TargetedRegularizer(config.tr_degree, config.tr_knots).to(self.device)
            if config.targeted_regularization
            else None
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.tr_optimizer = (
            torch.optim.Adam(
                self.targeted_regularizer.parameters(),
                lr=config.tr_learning_rate,
                weight_decay=config.weight_decay,
            )
            if self.targeted_regularizer is not None
            else None
        )
        self.y_mean = 0.0
        self.y_scale = 1.0
        self.loss_history: list[float] = []

    def _sample_batch(self, n: int) -> torch.Tensor:
        batch_size = min(self.config.batch_size, n)
        return torch.randint(0, n, size=(batch_size,), device=self.device)

    def _normalize_y(self, y: NDArray) -> NDArray:
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if not self.config.standardize_outcome:
            self.y_mean = 0.0
            self.y_scale = 1.0
            return y_arr
        self.y_mean = float(np.mean(y_arr))
        scale = float(np.std(y_arr))
        self.y_scale = scale if scale > 1e-6 else 1.0
        return ((y_arr - self.y_mean) / self.y_scale).astype(np.float32)

    def _base_loss(self, density: torch.Tensor, q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(q.reshape_as(y), y)
        density_nll = -torch.log(density + 1e-6).mean()
        return mse + self.config.density_loss_weight * density_nll

    def _targeted_loss(
        self,
        treatment: torch.Tensor,
        density: torch.Tensor,
        q: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        if self.targeted_regularizer is None:
            return torch.zeros((), device=self.device)
        eps = self.targeted_regularizer(treatment).reshape_as(y)
        targeted_q = q.reshape_as(y) + eps / (density.reshape_as(y) + 1e-6)
        return self.config.tr_weight * F.mse_loss(targeted_q, y)

    def fit(self, x: NDArray, treatment: NDArray, y: NDArray) -> "VCNet":
        x_arr = np.asarray(x, dtype=np.float32)
        t_arr = np.asarray(treatment, dtype=np.float32).reshape(-1)
        y_arr = self._normalize_y(y)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        if x_arr.shape[0] != t_arr.shape[0] or x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("x, treatment, and y must have the same number of rows")
        if x_arr.shape[1] != self.config.x_dim:
            raise ValueError(f"x has {x_arr.shape[1]} columns, expected {self.config.x_dim}")

        x_t = torch.as_tensor(x_arr, dtype=torch.float32, device=self.device)
        treatment_t = torch.as_tensor(np.clip(t_arr, 0.0, 1.0), dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y_arr, dtype=torch.float32, device=self.device)
        n = x_t.shape[0]

        for step in range(self.config.num_steps):
            idx = self._sample_batch(n)
            xb, tb, yb = x_t[idx], treatment_t[idx], y_t[idx]
            density, q = self.model(tb, xb)
            loss = self._base_loss(density, q, yb) + self._targeted_loss(tb, density, q, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if self.targeted_regularizer is not None and self.tr_optimizer is not None:
                density, q = self.model(tb, xb)
                tr_loss = self._targeted_loss(tb, density, q, yb)
                self.tr_optimizer.zero_grad(set_to_none=True)
                tr_loss.backward()
                self.tr_optimizer.step()

            if step % 25 == 0 or step == self.config.num_steps - 1:
                self.loss_history.append(float(loss.detach().cpu().item()))
        return self

    def predict_response(self, x: NDArray, treatment: NDArray | float) -> NDArray:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        t_arr = np.asarray(treatment, dtype=np.float32).reshape(-1)
        if t_arr.size == 1 and x_arr.shape[0] > 1:
            t_arr = np.full(x_arr.shape[0], float(t_arr.item()), dtype=np.float32)
        if t_arr.shape[0] != x_arr.shape[0]:
            raise ValueError("treatment must be scalar or align with x")

        outputs: List[NDArray] = []
        self.model.eval()
        if self.targeted_regularizer is not None:
            self.targeted_regularizer.eval()
        with torch.no_grad():
            start = 0
            while start < x_arr.shape[0]:
                stop = min(x_arr.shape[0], start + self.config.max_predict_batch)
                xb = torch.as_tensor(x_arr[start:stop], dtype=torch.float32, device=self.device)
                tb = torch.as_tensor(np.clip(t_arr[start:stop], 0.0, 1.0), dtype=torch.float32, device=self.device)
                density, q = self.model(tb, xb)
                pred = q
                if self.targeted_regularizer is not None:
                    eps = self.targeted_regularizer(tb).reshape_as(q)
                    pred = pred + eps / (density.reshape_as(q) + 1e-6)
                outputs.append(pred.cpu().numpy())
                start = stop
        self.model.train()
        if self.targeted_regularizer is not None:
            self.targeted_regularizer.train()
        pred_arr = np.concatenate(outputs, axis=0)
        pred_arr = pred_arr * self.y_scale + self.y_mean
        return pred_arr.astype(np.float32)
