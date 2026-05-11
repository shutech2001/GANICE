from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

from numpy.typing import NDArray
import torch
from torch import nn


ActivationName = Literal["id", "elu", "relu", "leaky_relu", "tanh"]


def build_mlp(
    input_dim: int,
    hidden_dims: Tuple[int, ...],
    output_dim: int,
    activation: ActivationName = "elu",
) -> nn.Sequential:
    layers: List[nn.Module] = []
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
        hidden_dims: Tuple[int, ...],
        anchor: NDArray,
        activation: ActivationName = "elu",
    ) -> None:
        super().__init__()
        self.network = build_mlp(y_dim, hidden_dims, 1, activation=activation)
        self.register_buffer("anchor", torch.as_tensor(anchor.reshape(1, -1), dtype=torch.float32))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        anchor = self.anchor.expand(y.shape[0], -1)
        return self.network(y) - self.network(anchor)


def sample_latent(
    batch_size: int,
    latent_dim: int,
    device: torch.device,
    seed: Optional[int] = None,
) -> torch.Tensor:
    if seed is None:
        return torch.rand(batch_size, latent_dim, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.rand(batch_size, latent_dim, generator=generator, device=device)


@dataclass(slots=True)
class AdversarialDiagnostics:
    critic_losses: List[float] = field(default_factory=list)
    generator_losses: List[float] = field(default_factory=list)
    objective_gaps: List[float] = field(default_factory=list)
