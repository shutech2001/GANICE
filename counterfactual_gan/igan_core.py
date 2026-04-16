from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ActivationName = Literal["elu", "relu", "leaky_relu", "tanh"]
KernelName = Literal["laplace", "matern12", "matern32"]


def build_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    activation: ActivationName = "elu",
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for width in hidden_dims:
        layers.append(nn.Linear(prev, width))
        if activation == "elu":
            layers.append(nn.ELU())
        elif activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "leaky_relu":
            layers.append(nn.LeakyReLU(0.2))
        elif activation == "tanh":
            layers.append(nn.Tanh())
        else:
            raise ValueError(f"unknown activation {activation}")
        prev = width
    layers.append(nn.Linear(prev, output_dim))
    network = nn.Sequential(*layers)
    for module in network:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    return network


class BoundedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        lower: float,
        upper: float,
        activation: ActivationName = "elu",
    ) -> None:
        super().__init__()
        if upper <= lower:
            raise ValueError("upper must be larger than lower")
        self.lower = float(lower)
        self.upper = float(upper)
        self.network = build_mlp(input_dim, hidden_dims, output_dim, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.network(x)
        return self.lower + (self.upper - self.lower) * torch.sigmoid(raw)


class AnchoredOutcomeCritic(nn.Module):
    def __init__(
        self,
        y_dim: int,
        hidden_dims: tuple[int, ...],
        anchor: np.ndarray,
        activation: ActivationName = "elu",
    ) -> None:
        super().__init__()
        self.network = build_mlp(y_dim, hidden_dims, 1, activation=activation)
        self.register_buffer("anchor", torch.as_tensor(anchor.reshape(1, -1), dtype=torch.float32))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        anchor = self.anchor.expand(y.shape[0], -1)
        return self.network(y) - self.network(anchor)


class AnchoredConditionalCritic(nn.Module):
    def __init__(
        self,
        w_dim: int,
        y_dim: int,
        hidden_dims: tuple[int, ...],
        anchor: np.ndarray,
        activation: ActivationName = "elu",
    ) -> None:
        super().__init__()
        self.network = build_mlp(w_dim + y_dim, hidden_dims, 1, activation=activation)
        self.register_buffer("anchor", torch.as_tensor(anchor.reshape(1, -1), dtype=torch.float32))

    def forward(self, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        anchor = self.anchor.expand(y.shape[0], -1)
        return self.network(torch.cat([w, y], dim=1)) - self.network(torch.cat([w, anchor], dim=1))


class ConditionalGenerator(nn.Module):
    def __init__(
        self,
        w_dim: int,
        latent_dim: int,
        y_dim: int,
        hidden_dims: tuple[int, ...],
        lower: float,
        upper: float,
        activation: ActivationName = "elu",
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.network = BoundedMLP(
            input_dim=w_dim + latent_dim,
            hidden_dims=hidden_dims,
            output_dim=y_dim,
            lower=lower,
            upper=upper,
            activation=activation,
        )

    def forward(self, w: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([w, z], dim=1))


class SoftVoronoiGate(nn.Module):
    def __init__(
        self,
        w_dim: int,
        num_experts: int,
        initial_temperature: float = 8.0,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(w_dim, w_dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(w_dim))
        self.centers = nn.Parameter(torch.rand(num_experts, w_dim))
        init_temp = max(float(initial_temperature), 1e-3)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor([math.expm1(init_temp)], dtype=torch.float32)))

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.log_temperature) + 1e-3

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        projected = self.linear(w)
        distances = torch.cdist(projected, self.centers, p=2.0).pow(2)
        return torch.softmax(-self.temperature * distances, dim=1)


class VoronoiGenerator(nn.Module):
    def __init__(
        self,
        w_dim: int,
        latent_dim: int,
        y_dim: int,
        num_experts: int,
        hidden_dims: tuple[int, ...],
        lower: float,
        upper: float,
        activation: ActivationName = "elu",
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.gate = SoftVoronoiGate(w_dim=w_dim, num_experts=num_experts)
        self.experts = nn.ModuleList(
            [
                BoundedMLP(
                    input_dim=latent_dim,
                    hidden_dims=hidden_dims,
                    output_dim=y_dim,
                    lower=lower,
                    upper=upper,
                    activation=activation,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, w: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        weights = self.gate(w)
        expert_outputs = torch.stack([expert(z) for expert in self.experts], dim=1)
        return torch.sum(weights.unsqueeze(-1) * expert_outputs, dim=1)


class VoronoiCritic(nn.Module):
    def __init__(
        self,
        w_dim: int,
        y_dim: int,
        num_experts: int,
        hidden_dims: tuple[int, ...],
        anchor: np.ndarray,
        activation: ActivationName = "elu",
    ) -> None:
        super().__init__()
        self.gate = SoftVoronoiGate(w_dim=w_dim, num_experts=num_experts)
        self.experts = nn.ModuleList(
            [
                AnchoredOutcomeCritic(
                    y_dim=y_dim,
                    hidden_dims=hidden_dims,
                    anchor=anchor,
                    activation=activation,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        weights = self.gate(w)
        expert_scores = torch.cat([expert(y) for expert in self.experts], dim=1)
        return torch.sum(weights * expert_scores, dim=1, keepdim=True)


class KernelOutcomeCritic(nn.Module):
    def __init__(
        self,
        w_dim: int,
        y_dim: int,
        num_anchors: int,
        hidden_dims: tuple[int, ...],
        lower: float,
        upper: float,
        anchor: np.ndarray,
        activation: ActivationName = "elu",
        kernel: KernelName = "matern32",
        bandwidth: float = 0.2,
        ridge: float = 1e-4,
    ) -> None:
        super().__init__()
        self.lower = float(lower)
        self.upper = float(upper)
        self.kernel = kernel
        self.bandwidth = nn.Parameter(torch.tensor([float(bandwidth)], dtype=torch.float32))
        self.ridge = float(ridge)
        self.coeff_net = build_mlp(w_dim, hidden_dims, num_anchors, activation=activation)
        self.anchor_raw = nn.Parameter(torch.empty(num_anchors, y_dim).uniform_(-0.5, 0.5))
        self.register_buffer("anchor", torch.as_tensor(anchor.reshape(1, -1), dtype=torch.float32))

    def anchors(self) -> torch.Tensor:
        return self.lower + (self.upper - self.lower) * torch.sigmoid(self.anchor_raw)

    def kernel_matrix(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        distance = torch.cdist(lhs, rhs, p=2.0)
        bandwidth = torch.clamp(self.bandwidth, min=1e-3)
        scaled = distance / bandwidth
        if self.kernel == "laplace" or self.kernel == "matern12":
            return torch.exp(-scaled)
        if self.kernel == "matern32":
            root3 = math.sqrt(3.0)
            term = root3 * scaled
            return (1.0 + term) * torch.exp(-term)
        raise ValueError(f"unknown kernel {self.kernel}")

    def coefficient_map(self, w: torch.Tensor) -> torch.Tensor:
        coeffs = self.coeff_net(w)
        anchors = self.anchors()
        gram = self.kernel_matrix(anchors, anchors)
        gram = gram + self.ridge * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        norm_sq = torch.einsum("bi,ij,bj->b", coeffs, gram, coeffs)
        scale = torch.clamp(norm_sq.sqrt(), min=1.0).unsqueeze(1)
        return coeffs / scale

    def forward(self, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        anchors = self.anchors()
        coeffs = self.coefficient_map(w)
        anchor = self.anchor.expand(y.shape[0], -1)
        score = torch.sum(coeffs * self.kernel_matrix(y, anchors), dim=1, keepdim=True)
        anchor_score = torch.sum(coeffs * self.kernel_matrix(anchor, anchors), dim=1, keepdim=True)
        return score - anchor_score


def sample_latent(
    batch_size: int,
    latent_dim: int,
    device: torch.device,
    seed: int | None = None,
) -> torch.Tensor:
    if seed is None:
        return torch.rand(batch_size, latent_dim, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.rand(batch_size, latent_dim, generator=generator, device=device)


def outcome_gradient_penalty(
    critic,
    w: torch.Tensor,
    real_y: torch.Tensor,
    fake_y: torch.Tensor,
) -> torch.Tensor:
    alpha = torch.rand(real_y.shape[0], 1, device=real_y.device, dtype=real_y.dtype)
    alpha = alpha.expand_as(real_y)
    interpolated_y = alpha * real_y + (1.0 - alpha) * fake_y
    interpolated_y.requires_grad_(True)
    scores = critic(w, interpolated_y)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated_y,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.shape[0], -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


def finite_difference_besov_penalty(
    critic,
    w: torch.Tensor,
    y: torch.Tensor,
    taus: torch.Tensor,
    smoothness: torch.Tensor,
    p: float,
) -> torch.Tensor:
    penalties: list[torch.Tensor] = []
    for coord in range(w.shape[1]):
        tau = float(taus[coord].item())
        shift = torch.full((w.shape[0],), tau, device=w.device, dtype=w.dtype)
        need_negative = w[:, coord] + shift > 1.0
        shift = torch.where(need_negative, -shift, shift)
        w_shift = w.clone()
        w_shift[:, coord] = torch.clamp(w_shift[:, coord] + shift, 0.0, 1.0)
        diff = torch.abs(critic(w_shift, y) - critic(w, y)).reshape(-1)
        penalties.append(torch.mean(diff.pow(p) / (tau ** (1.0 + float(smoothness[coord].item()) * p))))
    return torch.stack(penalties)


def coefficient_besov_penalty(
    coefficient_map,
    w: torch.Tensor,
    taus: torch.Tensor,
    smoothness: torch.Tensor,
    p: float,
) -> torch.Tensor:
    base = coefficient_map(w)
    penalties: list[torch.Tensor] = []
    for coord in range(w.shape[1]):
        tau = float(taus[coord].item())
        shift = torch.full((w.shape[0],), tau, device=w.device, dtype=w.dtype)
        need_negative = w[:, coord] + shift > 1.0
        shift = torch.where(need_negative, -shift, shift)
        w_shift = w.clone()
        w_shift[:, coord] = torch.clamp(w_shift[:, coord] + shift, 0.0, 1.0)
        shifted = coefficient_map(w_shift)
        diff = torch.norm(shifted - base, dim=1)
        penalties.append(torch.mean(diff.pow(p) / (tau ** (1.0 + float(smoothness[coord].item()) * p))))
    return torch.stack(penalties)


@dataclass(slots=True)
class IGANDiagnostics:
    critic_losses: list[float] = field(default_factory=list)
    generator_losses: list[float] = field(default_factory=list)
    objective_gaps: list[float] = field(default_factory=list)
