from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .utils import ensure_2d


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for width in hidden_dims:
        layers.append(nn.Linear(prev, width))
        layers.append(nn.LeakyReLU(0.2))
        prev = width
    layers.append(nn.Linear(prev, output_dim))
    network = nn.Sequential(*layers)
    for module in network:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    return network


class BoundedGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        y_dim: int,
        hidden_dims: tuple[int, ...],
        outcome_bound: float,
    ) -> None:
        super().__init__()
        self.outcome_bound = float(outcome_bound)
        self.network = _mlp(latent_dim, hidden_dims, y_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.outcome_bound * torch.tanh(self.network(z))


class AnchoredCritic(nn.Module):
    def __init__(
        self,
        y_dim: int,
        hidden_dims: tuple[int, ...],
        anchor: np.ndarray,
    ) -> None:
        super().__init__()
        self.network = _mlp(y_dim, hidden_dims, 1)
        self.register_buffer("anchor", torch.as_tensor(anchor.reshape(1, -1), dtype=torch.float32))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        raw = self.network(y)
        anchor_value = self.network(self.anchor.expand(y.shape[0], -1))
        return raw - anchor_value


def gradient_penalty(
    critic: AnchoredCritic,
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    alpha = torch.rand(real.shape[0], 1, device=real.device, dtype=real.dtype)
    alpha = alpha.expand_as(real)
    interpolated = alpha * real + (1.0 - alpha) * fake
    interpolated.requires_grad_(True)
    scores = critic(interpolated)
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
class LocalWGANConfig:
    latent_dim: int = 2
    hidden_dims_generator: tuple[int, ...] = (64, 64)
    hidden_dims_critic: tuple[int, ...] = (64, 64)
    batch_size: int = 64
    num_steps: int = 250
    critic_steps: int = 4
    generator_lr: float = 2e-4
    critic_lr: float = 1e-4
    betas: tuple[float, float] = (0.0, 0.9)
    gradient_penalty_weight: float = 10.0
    adversarial_generator_weight: float = 0.1
    direct_transport_weight: float = 5.0
    max_transport_batch: int = 512
    outcome_bound: float = 3.0
    device: str = "cpu"
    seed: int = 123


@dataclass(slots=True)
class LocalWGANDiagnostics:
    critic_losses: list[float] = field(default_factory=list)
    generator_losses: list[float] = field(default_factory=list)
    num_observations: int = 0


class LocalWGAN:
    def __init__(self, y_dim: int, config: LocalWGANConfig, anchor: np.ndarray | None = None) -> None:
        self.y_dim = y_dim
        self.config = config
        self.anchor = np.zeros((y_dim,), dtype=np.float32) if anchor is None else np.asarray(anchor, dtype=np.float32)
        torch.manual_seed(config.seed)
        torch.set_num_threads(1)
        self.device = torch.device(config.device)
        self.generator = BoundedGenerator(
            latent_dim=config.latent_dim,
            y_dim=y_dim,
            hidden_dims=config.hidden_dims_generator,
            outcome_bound=config.outcome_bound,
        ).to(self.device)
        self.critic = AnchoredCritic(
            y_dim=y_dim,
            hidden_dims=config.hidden_dims_critic,
            anchor=self.anchor,
        ).to(self.device)
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.generator_lr,
            betas=config.betas,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.critic_lr,
            betas=config.betas,
        )
        self.diagnostics = LocalWGANDiagnostics()

    def _sample_latent(self, batch_size: int) -> torch.Tensor:
        return torch.rand(batch_size, self.config.latent_dim, device=self.device)

    def fit(self, observations: np.ndarray) -> LocalWGANDiagnostics:
        y = ensure_2d(observations)
        if y.shape[0] < 2:
            raise ValueError("LocalWGAN requires at least two observations")
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        batch_size = min(self.config.batch_size, y.shape[0])
        self.diagnostics = LocalWGANDiagnostics(num_observations=int(y.shape[0]))

        for step in range(self.config.num_steps):
            for _ in range(self.config.critic_steps):
                indices = torch.randint(0, y_tensor.shape[0], size=(batch_size,), device=self.device)
                real = y_tensor[indices]
                fake = self.generator(self._sample_latent(batch_size)).detach()
                critic_real = self.critic(real)
                critic_fake = self.critic(fake)
                gp = gradient_penalty(self.critic, real, fake)
                critic_loss = -(critic_real.mean() - critic_fake.mean()) + self.config.gradient_penalty_weight * gp
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                self.critic_optimizer.step()

            fake = self.generator(self._sample_latent(batch_size))
            adversarial_loss = -self.critic(fake).mean()
            generator_loss = self.config.adversarial_generator_weight * adversarial_loss
            if self.y_dim == 1 and self.config.direct_transport_weight > 0.0:
                transport_batch = min(y_tensor.shape[0], self.config.max_transport_batch)
                transport_indices = torch.randint(0, y_tensor.shape[0], size=(transport_batch,), device=self.device)
                real_transport = y_tensor[transport_indices].reshape(-1)
                fake_transport = self.generator(self._sample_latent(transport_batch)).reshape(-1)
                real_sorted = torch.sort(real_transport)[0]
                fake_sorted = torch.sort(fake_transport)[0]
                transport_loss = torch.mean(torch.abs(real_sorted - fake_sorted))
                generator_loss = generator_loss + self.config.direct_transport_weight * transport_loss
            self.generator_optimizer.zero_grad(set_to_none=True)
            generator_loss.backward()
            self.generator_optimizer.step()

            if step % 10 == 0 or step == self.config.num_steps - 1:
                self.diagnostics.critic_losses.append(float(critic_loss.detach().cpu().item()))
                self.diagnostics.generator_losses.append(float(generator_loss.detach().cpu().item()))

        return self.diagnostics

    def sample(self, n: int, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            torch.manual_seed(seed)
        self.generator.eval()
        with torch.no_grad():
            samples = self.generator(self._sample_latent(n)).cpu().numpy()
        self.generator.train()
        return samples.astype(np.float32)
