from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .igan_core import (
    AnchoredOutcomeCritic,
    BoundedMLP,
    IGANDiagnostics,
    sample_latent,
)
from .metrics import finite_state_conditional_w1
from .utils import ensure_2d


class _StatewiseGenerator(nn.Module):
    def __init__(
        self,
        num_states: int,
        latent_dim: int,
        y_dim: int,
        hidden_dims: tuple[int, ...],
        outcome_lower: float,
        outcome_upper: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.y_dim = int(y_dim)
        self.networks = nn.ModuleList(
            [
                BoundedMLP(
                    input_dim=latent_dim,
                    hidden_dims=hidden_dims,
                    output_dim=y_dim,
                    lower=outcome_lower,
                    upper=outcome_upper,
                    activation=activation,
                )
                for _ in range(num_states)
            ]
        )

    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        outputs = torch.zeros(states.shape[0], self.y_dim, device=z.device, dtype=z.dtype)
        for state in states.unique(sorted=True).tolist():
            mask = states == state
            outputs[mask] = self.networks[state](z[mask])
        return outputs


class _StatewiseCritic(nn.Module):
    def __init__(
        self,
        num_states: int,
        y_dim: int,
        hidden_dims: tuple[int, ...],
        anchor: np.ndarray,
        activation: str,
    ) -> None:
        super().__init__()
        self.networks = nn.ModuleList(
            [
                AnchoredOutcomeCritic(
                    y_dim=y_dim,
                    hidden_dims=hidden_dims,
                    anchor=anchor,
                    activation=activation,
                )
                for _ in range(num_states)
            ]
        )

    def forward(self, states: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        outputs = torch.zeros(states.shape[0], 1, device=y.device, dtype=y.dtype)
        for state in states.unique(sorted=True).tolist():
            mask = states == state
            outputs[mask] = self.networks[state](y[mask])
        return outputs


def _statewise_gradient_penalty(
    critic: _StatewiseCritic,
    states: torch.Tensor,
    real_y: torch.Tensor,
    fake_y: torch.Tensor,
) -> torch.Tensor:
    alpha = torch.rand(real_y.shape[0], 1, device=real_y.device, dtype=real_y.dtype)
    alpha = alpha.expand_as(real_y)
    interpolated = alpha * real_y + (1.0 - alpha) * fake_y
    interpolated.requires_grad_(True)
    scores = critic(states, interpolated)
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


def _statewise_batch_transport_loss(
    *,
    states: torch.Tensor,
    real_y: torch.Tensor,
    fake_y: torch.Tensor,
    target_q: torch.Tensor,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for state in states.unique(sorted=True).tolist():
        mask = states == state
        if int(mask.sum().item()) < 2:
            continue
        real_state = torch.sort(real_y[mask].reshape(-1))[0]
        fake_state = torch.sort(fake_y[mask].reshape(-1))[0]
        losses.append(target_q[state] * torch.mean(torch.abs(real_state - fake_state)))
    if not losses:
        return torch.zeros((), device=real_y.device, dtype=real_y.dtype)
    return torch.stack(losses).sum()


@dataclass(slots=True)
class FiniteStateIGANConfig:
    latent_dim: int = 2
    hidden_dims_generator: tuple[int, ...] = (64, 64)
    hidden_dims_critic: tuple[int, ...] = (64, 64)
    batch_size: int = 128
    num_steps: int = 450
    critic_steps: int = 4
    generator_lr: float = 2e-4
    critic_lr: float = 1e-4
    betas: tuple[float, float] = (0.0, 0.9)
    gradient_penalty_weight: float = 10.0
    generator_transport_weight: float = 5.0
    min_state_samples: int = 20
    outcome_lower: float = -3.0
    outcome_upper: float = 3.0
    critic_anchor: float = 0.0
    activation: str = "elu"
    device: str = "cpu"
    seed: int = 123


class FiniteStateIGAN:
    def __init__(self, num_states: int, target_q: np.ndarray, config: FiniteStateIGANConfig) -> None:
        self.num_states = int(num_states)
        self.target_q = np.asarray(target_q, dtype=np.float64)
        if self.target_q.shape != (self.num_states,):
            raise ValueError("target_q must have shape (num_states,)")
        if np.any(self.target_q < 0.0) or not np.isclose(self.target_q.sum(), 1.0):
            raise ValueError("target_q must be nonnegative and sum to one")
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        torch.set_num_threads(1)
        self.generator: _StatewiseGenerator | None = None
        self.critic: _StatewiseCritic | None = None
        self.generator_optimizer: torch.optim.Optimizer | None = None
        self.critic_optimizer: torch.optim.Optimizer | None = None
        self.state_counts: np.ndarray | None = None
        self.pi_hat: np.ndarray | None = None
        self.importance_weights: np.ndarray | None = None
        self.y_dim: int | None = None
        self.diagnostics = IGANDiagnostics()
        self._target_q_tensor = torch.as_tensor(self.target_q, dtype=torch.float32, device=self.device)

    def _build_networks(self, y_dim: int) -> None:
        anchor = np.full((y_dim,), self.config.critic_anchor, dtype=np.float32)
        self.generator = _StatewiseGenerator(
            num_states=self.num_states,
            latent_dim=self.config.latent_dim,
            y_dim=y_dim,
            hidden_dims=self.config.hidden_dims_generator,
            outcome_lower=self.config.outcome_lower,
            outcome_upper=self.config.outcome_upper,
            activation=self.config.activation,
        ).to(self.device)
        self.critic = _StatewiseCritic(
            num_states=self.num_states,
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

    def fit(self, states: np.ndarray, outcomes: np.ndarray) -> "FiniteStateIGAN":
        states_arr = np.asarray(states, dtype=np.int64).reshape(-1)
        outcomes_arr = ensure_2d(outcomes)
        if states_arr.shape[0] != outcomes_arr.shape[0]:
            raise ValueError("states and outcomes must have the same number of rows")
        counts = np.bincount(states_arr, minlength=self.num_states).astype(np.int64)
        if np.any((counts == 0) & (self.target_q > 0.0)):
            raise ValueError("observed data missed a target state with positive mass")
        if counts.min() < self.config.min_state_samples:
            missing = int(counts.min())
            raise ValueError(
                f"at least one state has only {missing} samples; "
                f"need at least {self.config.min_state_samples}"
            )

        self.state_counts = counts
        self.pi_hat = counts / counts.sum()
        self.importance_weights = self.target_q / np.maximum(self.pi_hat, np.finfo(np.float64).eps)
        self.y_dim = outcomes_arr.shape[1]
        self._build_networks(self.y_dim)

        states_t = torch.as_tensor(states_arr, dtype=torch.long, device=self.device)
        outcomes_t = torch.as_tensor(outcomes_arr, dtype=torch.float32, device=self.device)
        weights_t = torch.as_tensor(self.importance_weights[states_arr], dtype=torch.float32, device=self.device).unsqueeze(1)
        batch_size = min(self.config.batch_size, states_arr.shape[0])
        self.diagnostics = IGANDiagnostics()

        for step in range(self.config.num_steps):
            critic_loss_value = 0.0
            objective_gap = 0.0
            for _ in range(self.config.critic_steps):
                batch_idx = torch.randint(0, states_t.shape[0], size=(batch_size,), device=self.device)
                state_batch = states_t[batch_idx]
                real_y = outcomes_t[batch_idx]
                weight_batch = weights_t[batch_idx]

                fake_states = torch.multinomial(self._target_q_tensor, batch_size, replacement=True)
                fake_y = self.generator(fake_states, sample_latent(batch_size, self.config.latent_dim, self.device)).detach()
                penalty_fake = self.generator(
                    state_batch,
                    sample_latent(batch_size, self.config.latent_dim, self.device),
                ).detach()

                real_score = self.critic(state_batch, real_y)
                fake_score = self.critic(fake_states, fake_y)
                gp = _statewise_gradient_penalty(self.critic, state_batch, real_y, penalty_fake)
                critic_loss = -(weight_batch * real_score).mean() + fake_score.mean() + self.config.gradient_penalty_weight * gp

                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                self.critic_optimizer.step()

                critic_loss_value = float(critic_loss.detach().cpu().item())
                objective_gap = float(((weight_batch * real_score).mean() - fake_score.mean()).detach().cpu().item())

            fake_states = torch.multinomial(self._target_q_tensor, batch_size, replacement=True)
            fake_y = self.generator(fake_states, sample_latent(batch_size, self.config.latent_dim, self.device))
            generator_loss = -self.critic(fake_states, fake_y).mean()
            if self.y_dim == 1 and self.config.generator_transport_weight > 0.0:
                batch_idx = torch.randint(0, states_t.shape[0], size=(batch_size,), device=self.device)
                state_batch = states_t[batch_idx]
                real_batch = outcomes_t[batch_idx]
                fake_batch = self.generator(
                    state_batch,
                    sample_latent(batch_size, self.config.latent_dim, self.device),
                )
                generator_loss = generator_loss + self.config.generator_transport_weight * _statewise_batch_transport_loss(
                    states=state_batch,
                    real_y=real_batch,
                    fake_y=fake_batch,
                    target_q=self._target_q_tensor,
                )

            self.generator_optimizer.zero_grad(set_to_none=True)
            generator_loss.backward()
            self.generator_optimizer.step()

            if step % 10 == 0 or step == self.config.num_steps - 1:
                self.diagnostics.critic_losses.append(critic_loss_value)
                self.diagnostics.generator_losses.append(float(generator_loss.detach().cpu().item()))
                self.diagnostics.objective_gaps.append(objective_gap)
        return self

    def sample_state(self, state: int, n: int, seed: int | None = None) -> np.ndarray:
        if self.generator is None:
            raise ValueError("model has not been fitted")
        state_value = int(state)
        if state_value < 0 or state_value >= self.num_states:
            raise ValueError("state out of range")
        self.generator.eval()
        with torch.no_grad():
            states = torch.full((n,), state_value, dtype=torch.long, device=self.device)
            samples = self.generator(states, sample_latent(n, self.config.latent_dim, self.device, seed=seed)).cpu().numpy()
        self.generator.train()
        return samples.astype(np.float32)

    def estimated_state_means(self, n_mc: int = 2048) -> np.ndarray:
        means = np.zeros(self.num_states, dtype=np.float64)
        for state in range(self.num_states):
            means[state] = float(self.sample_state(state, n_mc, seed=50_000 + state).mean())
        return means

    def approximate_conditional_w1(
        self,
        true_sampler,
        n_per_state: int = 2048,
    ) -> float:
        return finite_state_conditional_w1(
            state_weights=self.target_q,
            true_sampler=true_sampler,
            learned_sampler=lambda state, n: self.sample_state(state, n, seed=60_000 + state),
            n_per_state=n_per_state,
        )
