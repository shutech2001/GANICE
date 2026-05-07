from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for width in hidden_dims:
        layers.append(nn.Linear(prev, width))
        layers.append(nn.SiLU())
        prev = width
    layers.append(nn.Linear(prev, output_dim))
    network = nn.Sequential(*layers)
    for module in network:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    return network


class _GatedResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, context_dim: int) -> None:
        super().__init__()
        self.net = _mlp(hidden_dim + context_dim, (hidden_dim,), 2 * hidden_dim)

    def forward(self, h: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        value, gate = self.net(torch.cat([h, context], dim=1)).chunk(2, dim=1)
        update = torch.tanh(value) * torch.sigmoid(gate)
        return 0.5 * (h + update)


class _POFlowVelocity(nn.Module):
    def __init__(self, x_dim: int, num_treatments: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_treatments = int(num_treatments)
        context_dim = hidden_dim
        self.context = _mlp(x_dim + num_treatments + 1, (hidden_dim,), context_dim)
        self.y_embed = nn.Linear(1, hidden_dim)
        self.film = nn.Linear(context_dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList(
            [_GatedResidualBlock(hidden_dim=hidden_dim, context_dim=context_dim) for _ in range(2)]
        )
        self.projection = nn.Linear(hidden_dim, num_treatments)

    def forward(self, y: torch.Tensor, time: torch.Tensor, x: torch.Tensor, treatment: torch.Tensor) -> torch.Tensor:
        treatment_onehot = F.one_hot(treatment, num_classes=self.num_treatments).to(dtype=x.dtype)
        context = self.context(torch.cat([x, treatment_onehot, time], dim=1))
        h = self.y_embed(y)
        gamma, beta = self.film(context).chunk(2, dim=1)
        h = torch.tanh(h * (1.0 + gamma) + beta)
        for block in self.blocks:
            h = block(h, context)
        all_velocities = self.projection(h)
        return all_velocities.gather(1, treatment.reshape(-1, 1))


@dataclass(slots=True)
class POFlowConfig:
    x_dim: int
    num_treatments: int = 2
    hidden_dim: int = 64
    batch_size: int = 128
    num_steps: int = 700
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    rk4_steps: int = 16
    outcome_min: float | None = None
    outcome_max: float | None = None
    seed: int = 41
    device: str = "cpu"


@dataclass(slots=True)
class POFlowDiagnostics:
    losses: list[float] = field(default_factory=list)


class POFlow:
    """Binary-treatment PO-Flow baseline trained with conditional flow matching."""

    def __init__(self, config: POFlowConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        torch.set_num_threads(1)
        self.device = torch.device(config.device)
        self.velocity = _POFlowVelocity(
            x_dim=config.x_dim,
            num_treatments=config.num_treatments,
            hidden_dim=config.hidden_dim,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.velocity.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.diagnostics = POFlowDiagnostics()

    def fit(
        self,
        x: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "POFlow":
        x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        treatment_t = torch.as_tensor(np.asarray(treatment, dtype=np.int64).reshape(-1), device=self.device)
        y_t = torch.as_tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1), device=self.device)
        if x_t.shape[0] != treatment_t.shape[0] or x_t.shape[0] != y_t.shape[0]:
            raise ValueError("x, treatment, and y must have the same number of rows")
        if sample_weight is None:
            weight_t = torch.ones(x_t.shape[0], 1, device=self.device)
        else:
            weight_arr = np.asarray(sample_weight, dtype=np.float32).reshape(-1, 1)
            if weight_arr.shape[0] != x_t.shape[0]:
                raise ValueError("sample_weight must align with x")
            weight_t = torch.as_tensor(weight_arr, dtype=torch.float32, device=self.device)
            weight_t = weight_t / torch.clamp(weight_t.mean(), min=1e-6)

        batch_size = min(self.config.batch_size, x_t.shape[0])
        self.diagnostics = POFlowDiagnostics()

        for step in range(self.config.num_steps):
            idx = torch.randint(0, x_t.shape[0], size=(batch_size,), device=self.device)
            xb = x_t[idx]
            tb = treatment_t[idx]
            y0 = y_t[idx]
            wb = weight_t[idx]
            z = torch.randn_like(y0)
            time = torch.rand(batch_size, 1, device=self.device)
            phi = (1.0 - time) * y0 + time * z
            target_velocity = z - y0
            pred_velocity = self.velocity(phi, time, xb, tb)
            loss = torch.mean(wb * (pred_velocity - target_velocity).pow(2))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if step % 10 == 0 or step == self.config.num_steps - 1:
                self.diagnostics.losses.append(float(loss.detach().cpu().item()))
        return self

    def _rk4_integrate(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        treatment: torch.Tensor,
        start_time: float,
        end_time: float,
    ) -> torch.Tensor:
        steps = int(self.config.rk4_steps)
        h = (end_time - start_time) / steps
        current = y
        current_time = float(start_time)
        for _ in range(steps):
            t1 = torch.full_like(current, current_time)
            k1 = self.velocity(current, t1, x, treatment)
            t2 = torch.full_like(current, current_time + 0.5 * h)
            k2 = self.velocity(current + 0.5 * h * k1, t2, x, treatment)
            k3 = self.velocity(current + 0.5 * h * k2, t2, x, treatment)
            t4 = torch.full_like(current, current_time + h)
            k4 = self.velocity(current + h * k3, t4, x, treatment)
            current = current + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            current_time += h
        return current

    def sample_potential(
        self,
        x: np.ndarray,
        treatment: np.ndarray | int,
        n_per_x: int = 1,
        seed: int | None = None,
    ) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        n = x_arr.shape[0]
        treatment_arr = np.asarray(treatment, dtype=np.int64).reshape(-1)
        if treatment_arr.size == 1:
            treatment_arr = np.full(n, int(treatment_arr.item()), dtype=np.int64)
        if treatment_arr.shape[0] != n:
            raise ValueError("treatment must be scalar or align with x")
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
        else:
            generator = None
        repeated_x = np.repeat(x_arr, n_per_x, axis=0)
        repeated_t = np.repeat(treatment_arr, n_per_x)
        x_t = torch.as_tensor(repeated_x, dtype=torch.float32, device=self.device)
        treatment_t = torch.as_tensor(repeated_t, dtype=torch.long, device=self.device)
        z = torch.randn(repeated_x.shape[0], 1, generator=generator, device=self.device)
        self.velocity.eval()
        with torch.no_grad():
            samples = self._rk4_integrate(z, x_t, treatment_t, start_time=1.0, end_time=0.0)
            if self.config.outcome_min is not None or self.config.outcome_max is not None:
                lower = -float("inf") if self.config.outcome_min is None else self.config.outcome_min
                upper = float("inf") if self.config.outcome_max is None else self.config.outcome_max
                samples = torch.clamp(samples, lower, upper)
            result = samples.reshape(n, n_per_x, 1).cpu().numpy()
        self.velocity.train()
        return result.astype(np.float32)

    def predict_potential_outcomes(self, x: np.ndarray, n_mc: int = 512) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        means = []
        for treatment in range(self.config.num_treatments):
            samples = self.sample_potential(x_arr, treatment, n_per_x=n_mc, seed=80_000 + treatment)
            means.append(samples.mean(axis=1).reshape(-1))
        return np.stack(means, axis=1).astype(np.float32)

    def encode_factual(self, x: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        treatment_t = torch.as_tensor(np.asarray(treatment, dtype=np.int64).reshape(-1), device=self.device)
        y_t = torch.as_tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1), device=self.device)
        self.velocity.eval()
        with torch.no_grad():
            z = self._rk4_integrate(y_t, x_t, treatment_t, start_time=0.0, end_time=1.0).cpu().numpy()
        self.velocity.train()
        return z.astype(np.float32)

    def _velocity_and_divergence(
        self,
        y: torch.Tensor,
        time: torch.Tensor,
        x: torch.Tensor,
        treatment: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y_req = y.detach().requires_grad_(True)
        velocity = self.velocity(y_req, time, x, treatment)
        divergence = torch.autograd.grad(
            outputs=velocity.sum(),
            inputs=y_req,
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]
        return velocity.detach(), divergence.detach()

    def _rk4_integrate_with_divergence(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        treatment: torch.Tensor,
        start_time: float,
        end_time: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        steps = int(self.config.rk4_steps)
        h = (end_time - start_time) / steps
        current = y
        current_time = float(start_time)
        divergence_integral = torch.zeros_like(y)
        for _ in range(steps):
            t1 = torch.full_like(current, current_time)
            k1, d1 = self._velocity_and_divergence(current, t1, x, treatment)
            t2 = torch.full_like(current, current_time + 0.5 * h)
            k2, d2 = self._velocity_and_divergence(current + 0.5 * h * k1, t2, x, treatment)
            k3, d3 = self._velocity_and_divergence(current + 0.5 * h * k2, t2, x, treatment)
            t4 = torch.full_like(current, current_time + h)
            k4, d4 = self._velocity_and_divergence(current + h * k3, t4, x, treatment)
            current = current + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            divergence_integral = divergence_integral + (h / 6.0) * (d1 + 2.0 * d2 + 2.0 * d3 + d4)
            current_time += h
        return current, divergence_integral

    def log_prob_potential(self, x: np.ndarray, treatment: np.ndarray | int, y: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        n = x_arr.shape[0]
        treatment_arr = np.asarray(treatment, dtype=np.int64).reshape(-1)
        if treatment_arr.size == 1:
            treatment_arr = np.full(n, int(treatment_arr.item()), dtype=np.int64)
        if treatment_arr.shape[0] != n:
            raise ValueError("treatment must be scalar or align with x")
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if y_arr.shape[0] != n:
            raise ValueError("y must align with x")
        x_t = torch.as_tensor(x_arr, dtype=torch.float32, device=self.device)
        treatment_t = torch.as_tensor(treatment_arr, dtype=torch.long, device=self.device)
        y_t = torch.as_tensor(y_arr, dtype=torch.float32, device=self.device)
        self.velocity.eval()
        z_t, divergence_integral = self._rk4_integrate_with_divergence(
            y_t,
            x_t,
            treatment_t,
            start_time=0.0,
            end_time=1.0,
        )
        log_base = -0.5 * z_t.pow(2) - 0.5 * math.log(2.0 * math.pi)
        log_prob = (log_base + divergence_integral).reshape(-1).detach().cpu().numpy()
        self.velocity.train()
        return log_prob.astype(np.float32)

    def negative_log_likelihood(self, x: np.ndarray, treatment: np.ndarray | int, y: np.ndarray) -> float:
        return float(-np.mean(self.log_prob_potential(x, treatment, y)))

    def predict_counterfactual(self, x: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> np.ndarray:
        z = self.encode_factual(x, treatment, y)
        cf_treatment = 1 - np.asarray(treatment, dtype=np.int64).reshape(-1)
        x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        treatment_t = torch.as_tensor(cf_treatment, dtype=torch.long, device=self.device)
        z_t = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        self.velocity.eval()
        with torch.no_grad():
            y_cf = self._rk4_integrate(z_t, x_t, treatment_t, start_time=1.0, end_time=0.0)
            if self.config.outcome_min is not None or self.config.outcome_max is not None:
                lower = -float("inf") if self.config.outcome_min is None else self.config.outcome_min
                upper = float("inf") if self.config.outcome_max is None else self.config.outcome_max
                y_cf = torch.clamp(y_cf, lower, upper)
            result = y_cf.cpu().numpy()
        self.velocity.train()
        return result.astype(np.float32)
