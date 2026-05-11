from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch.nn import functional as F


def _mlp(input_dim: int, hidden_dims: Tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
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


class _SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("time embedding dimension must be at least 2")
        self.dim = int(dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10_000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -scale)
        args = timesteps.float().reshape(-1, 1) * freqs.reshape(1, -1)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if embedding.shape[1] < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.shape[1]))
        return embedding


class _ConditionalResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, context_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = _mlp(hidden_dim + context_dim, (hidden_dim,), 2 * hidden_dim)

    def forward(self, h: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        value, gate = self.net(torch.cat([self.norm(h), context], dim=1)).chunk(2, dim=1)
        update = torch.tanh(value) * torch.sigmoid(gate)
        return h + update


class _DiffPODenoiser(nn.Module):
    def __init__(
        self,
        x_dim: int,
        num_treatments: int,
        hidden_dim: int,
        time_embedding_dim: int,
        residual_blocks: int,
    ) -> None:
        super().__init__()
        self.num_treatments = int(num_treatments)
        self.time_embedding = nn.Sequential(
            _SinusoidalTimeEmbedding(time_embedding_dim),
            _mlp(time_embedding_dim, (hidden_dim,), hidden_dim),
        )
        self.context = _mlp(x_dim + num_treatments + hidden_dim, (hidden_dim,), hidden_dim)
        self.y_embed = nn.Linear(1, hidden_dim)
        self.blocks = nn.ModuleList(
            [_ConditionalResidualBlock(hidden_dim=hidden_dim, context_dim=hidden_dim) for _ in range(residual_blocks)]
        )
        self.output = _mlp(hidden_dim, (hidden_dim,), 1)

    def forward(
        self, y_t: torch.Tensor, timestep: torch.Tensor, x: torch.Tensor, treatment: torch.Tensor
    ) -> torch.Tensor:
        treatment_onehot = F.one_hot(treatment, num_classes=self.num_treatments).to(dtype=x.dtype)
        time_features = self.time_embedding(timestep)
        context = self.context(torch.cat([x, treatment_onehot, time_features], dim=1))
        h = self.y_embed(y_t) + context
        for block in self.blocks:
            h = block(h, context)
        return self.output(torch.tanh(h))


@dataclass(slots=True)
class DiffPOConfig:
    x_dim: int
    num_treatments: int = 2
    hidden_dim: int = 64
    time_embedding_dim: int = 128
    residual_blocks: int = 4
    batch_size: int = 128
    propensity_steps: int = 300
    diffusion_steps: int = 650
    num_diffusion_steps: int = 100
    learning_rate: float = 5e-4
    propensity_learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    min_propensity: float = 0.05
    outcome_min: Optional[float] = None
    outcome_max: Optional[float] = None
    standardize_outcome: bool = True
    max_sample_batch: int = 16_384
    seed: int = 59
    device: str = "cpu"


@dataclass(slots=True)
class DiffPODiagnostics:
    propensity_losses: List[float] = field(default_factory=list)
    diffusion_losses: List[float] = field(default_factory=list)
    mean_ipw: float = 0.0
    max_ipw: float = 0.0


class DiffPO:
    """Binary-treatment Diff-PO baseline with IPW orthogonal diffusion loss.

    This is the scalar-outcome specialization used for the GANITE-style
    benchmark: a propensity model is fit first, then a conditional DDPM predicts
    Gaussian noise under inverse-propensity weighted denoising loss.
    """

    def __init__(self, config: DiffPOConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        torch.set_num_threads(1)
        self.device = torch.device(config.device)
        self.propensity = _mlp(config.x_dim, (config.hidden_dim, config.hidden_dim), config.num_treatments).to(
            self.device
        )
        self.denoiser = _DiffPODenoiser(
            x_dim=config.x_dim,
            num_treatments=config.num_treatments,
            hidden_dim=config.hidden_dim,
            time_embedding_dim=config.time_embedding_dim,
            residual_blocks=config.residual_blocks,
        ).to(self.device)
        self.propensity_optimizer = torch.optim.AdamW(
            self.propensity.parameters(),
            lr=config.propensity_learning_rate,
            weight_decay=config.weight_decay,
        )
        self.denoiser_optimizer = torch.optim.AdamW(
            self.denoiser.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        betas = torch.linspace(config.beta_start, config.beta_end, config.num_diffusion_steps, device=self.device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, device=self.device), alpha_bars[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alpha_bars_prev) / torch.clamp(1.0 - alpha_bars, min=1e-8)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.posterior_variance = torch.clamp(posterior_variance, min=1e-8)
        self.y_mean = 0.0
        self.y_std = 1.0
        self.diagnostics = DiffPODiagnostics()

    def _normalize_y(self, y: torch.Tensor) -> torch.Tensor:
        if not self.config.standardize_outcome:
            return y
        return (y - self.y_mean) / self.y_std

    def _denormalize_y(self, y: torch.Tensor) -> torch.Tensor:
        if not self.config.standardize_outcome:
            return y
        return y * self.y_std + self.y_mean

    def _sample_batch(self, n: int) -> torch.Tensor:
        return torch.randint(0, n, size=(min(self.config.batch_size, n),), device=self.device)

    def _estimate_ipw(self, x: torch.Tensor, treatment: torch.Tensor) -> torch.Tensor:
        self.propensity.eval()
        with torch.no_grad():
            probs = torch.softmax(self.propensity(x), dim=1)
            assigned = probs.gather(1, treatment.reshape(-1, 1))
            assigned = torch.clamp(
                assigned,
                min=self.config.min_propensity,
                max=1.0 - self.config.min_propensity,
            )
            weights = 1.0 / assigned
            weights = weights / torch.clamp(weights.mean(), min=1e-6)
        self.propensity.train()
        return weights

    def fit(self, x: NDArray[np.float32], treatment: NDArray[np.int64], y: NDArray[np.float32]) -> "DiffPO":
        x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        treatment_t = torch.as_tensor(np.asarray(treatment, dtype=np.int64).reshape(-1), device=self.device)
        y_t = torch.as_tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1), device=self.device)
        if x_t.shape[0] != treatment_t.shape[0] or x_t.shape[0] != y_t.shape[0]:
            raise ValueError("x, treatment, and y must have the same number of rows")
        if self.config.standardize_outcome:
            self.y_mean = float(y_t.mean().detach().cpu().item())
            self.y_std = float(torch.clamp(y_t.std(unbiased=False), min=1e-3).detach().cpu().item())
        else:
            self.y_mean = 0.0
            self.y_std = 1.0
        y_norm = self._normalize_y(y_t)
        n = x_t.shape[0]
        self.diagnostics = DiffPODiagnostics()

        for step in range(self.config.propensity_steps):
            idx = self._sample_batch(n)
            logits = self.propensity(x_t[idx])
            loss = F.cross_entropy(logits, treatment_t[idx])
            self.propensity_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.propensity_optimizer.step()
            if step % 10 == 0 or step == self.config.propensity_steps - 1:
                self.diagnostics.propensity_losses.append(float(loss.detach().cpu().item()))

        weights_t = self._estimate_ipw(x_t, treatment_t)
        self.diagnostics.mean_ipw = float(weights_t.mean().detach().cpu().item())
        self.diagnostics.max_ipw = float(weights_t.max().detach().cpu().item())

        for step in range(self.config.diffusion_steps):
            idx = self._sample_batch(n)
            xb = x_t[idx]
            tb = treatment_t[idx]
            y0 = y_norm[idx]
            wb = weights_t[idx]
            timestep = torch.randint(0, self.config.num_diffusion_steps, size=(idx.shape[0],), device=self.device)
            eps = torch.randn_like(y0)
            alpha_bar = self.alpha_bars[timestep].reshape(-1, 1)
            y_noisy = torch.sqrt(alpha_bar) * y0 + torch.sqrt(1.0 - alpha_bar) * eps
            pred_eps = self.denoiser(y_noisy, timestep, xb, tb)
            loss = torch.mean(wb * (eps - pred_eps).pow(2))

            self.denoiser_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.denoiser_optimizer.step()
            if step % 10 == 0 or step == self.config.diffusion_steps - 1:
                self.diagnostics.diffusion_losses.append(float(loss.detach().cpu().item()))
        return self

    def _sample_normalized(
        self,
        x: torch.Tensor,
        treatment: torch.Tensor,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        y_current = torch.randn(x.shape[0], 1, generator=generator, device=self.device)
        self.denoiser.eval()
        with torch.no_grad():
            for step in range(self.config.num_diffusion_steps - 1, -1, -1):
                timestep = torch.full((x.shape[0],), step, dtype=torch.long, device=self.device)
                pred_eps = self.denoiser(y_current, timestep, x, treatment)
                beta_t = self.betas[step]
                alpha_t = self.alphas[step]
                alpha_bar_t = self.alpha_bars[step]
                mean = (y_current - beta_t * pred_eps / torch.sqrt(1.0 - alpha_bar_t)) / torch.sqrt(alpha_t)
                if step > 0:
                    noise = torch.randn(y_current.shape, generator=generator, device=self.device)
                    y_current = mean + torch.sqrt(self.posterior_variance[step]) * noise
                else:
                    y_current = mean
        self.denoiser.train()
        return y_current

    def sample_potential(
        self,
        x: NDArray[np.float32],
        treatment: NDArray[np.int64] | int,
        n_per_x: int = 1,
        seed: Optional[int] = None,
    ) -> NDArray[np.float32]:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        n = x_arr.shape[0]
        treatment_arr = np.asarray(treatment, dtype=np.int64).reshape(-1)
        if treatment_arr.size == 1:
            treatment_arr = np.full(n, int(treatment_arr.item()), dtype=np.int64)
        if treatment_arr.shape[0] != n:
            raise ValueError("treatment must be scalar or align with x")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        repeated_x = np.repeat(x_arr, n_per_x, axis=0)
        repeated_t = np.repeat(treatment_arr, n_per_x)
        outputs: list[np.ndarray] = []
        start = 0
        while start < repeated_x.shape[0]:
            stop = min(start + self.config.max_sample_batch, repeated_x.shape[0])
            x_t = torch.as_tensor(repeated_x[start:stop], dtype=torch.float32, device=self.device)
            treatment_t = torch.as_tensor(repeated_t[start:stop], dtype=torch.long, device=self.device)
            samples = self._denormalize_y(self._sample_normalized(x_t, treatment_t, generator))
            if self.config.outcome_min is not None or self.config.outcome_max is not None:
                lower = -float("inf") if self.config.outcome_min is None else self.config.outcome_min
                upper = float("inf") if self.config.outcome_max is None else self.config.outcome_max
                samples = torch.clamp(samples, lower, upper)
            outputs.append(samples.cpu().numpy())
            start = stop
        result = np.concatenate(outputs, axis=0).reshape(n, n_per_x, 1)
        return result.astype(np.float32)

    def predict_potential_outcomes(self, x: NDArray[np.float32], n_mc: int = 512) -> NDArray[np.float32]:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        means = []
        for treatment in range(self.config.num_treatments):
            samples = self.sample_potential(x_arr, treatment, n_per_x=n_mc, seed=90_000 + treatment)
            means.append(samples.mean(axis=1).reshape(-1))
        return np.stack(means, axis=1).astype(np.float32)
