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


class _INFsNuisance(nn.Module):
    def __init__(self, x_dim: int, num_treatments: int, hidden_dim: int, num_bins: int) -> None:
        super().__init__()
        self.num_treatments = int(num_treatments)
        self.representation = _mlp(x_dim, (hidden_dim,), hidden_dim)
        self.propensity = _mlp(hidden_dim, (hidden_dim,), num_treatments)
        self.conditional_density = _mlp(hidden_dim + num_treatments, (hidden_dim,), num_bins)

    def forward(self, x: torch.Tensor, treatment: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rep = self.representation(x)
        prop_logits = self.propensity(rep)
        treatment_onehot = F.one_hot(treatment, num_classes=self.num_treatments).to(dtype=x.dtype)
        density_logits = self.conditional_density(torch.cat([rep, treatment_onehot], dim=1))
        return density_logits, prop_logits

    def density_logits(self, x: torch.Tensor, treatment: torch.Tensor) -> torch.Tensor:
        return self.forward(x, treatment)[0]

    def propensity_logits(self, x: torch.Tensor) -> torch.Tensor:
        rep = self.representation(x)
        return self.propensity(rep)


@dataclass(slots=True)
class INFsConfig:
    x_dim: int
    num_treatments: int = 2
    hidden_dim: int = 64
    num_bins: int = 48
    batch_size: int = 128
    nuisance_steps: int = 650
    target_steps: int = 650
    nuisance_lr: float = 1e-3
    target_lr: float = 5e-3
    weight_decay: float = 1e-5
    prop_alpha: float = 1.0
    clip_propensity: float = 0.05
    noise_std_x: float = 0.0
    noise_std_y: float = 0.005
    outcome_min: float = 0.0
    outcome_max: float = 1.0
    seed: int = 67
    device: str = "cpu"


@dataclass(slots=True)
class INFsDiagnostics:
    nuisance_losses: list[float] = field(default_factory=list)
    target_losses: list[float] = field(default_factory=list)
    mean_ipw: float = 0.0
    max_ipw: float = 0.0


class INFs:
    """Scalar Interventional Normalizing Flows baseline for binary treatments.

    The implementation follows the two-stage INFs structure in a bounded
    one-dimensional setting: a nuisance flow estimates propensity scores and
    conditional outcome densities, then treatment-specific target flows learn
    marginal interventional densities with the A-IPTW bias-corrected objective.
    The scalar flow is an inverse-CDF piecewise-linear flow, equivalent to a
    positive normalized piecewise-constant density on the outcome interval.
    """

    def __init__(self, config: INFsConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        torch.set_num_threads(1)
        self.device = torch.device(config.device)
        self.nuisance = _INFsNuisance(
            x_dim=config.x_dim,
            num_treatments=config.num_treatments,
            hidden_dim=config.hidden_dim,
            num_bins=config.num_bins,
        ).to(self.device)
        self.target_logits = nn.Parameter(torch.zeros(config.num_treatments, config.num_bins, device=self.device))
        self.nuisance_optimizer = torch.optim.AdamW(
            self.nuisance.parameters(),
            lr=config.nuisance_lr,
            weight_decay=config.weight_decay,
        )
        self.target_optimizer = torch.optim.AdamW([self.target_logits], lr=config.target_lr)
        self.bin_width = (float(config.outcome_max) - float(config.outcome_min)) / float(config.num_bins)
        if not np.isfinite(self.bin_width) or self.bin_width <= 0.0:
            raise ValueError("outcome_max must be larger than outcome_min")
        centers = np.linspace(
            config.outcome_min + 0.5 * self.bin_width,
            config.outcome_max - 0.5 * self.bin_width,
            config.num_bins,
            dtype=np.float32,
        )
        self.bin_centers = torch.as_tensor(centers, dtype=torch.float32, device=self.device)
        self.diagnostics = INFsDiagnostics()

    def _bin_indices(self, y: torch.Tensor) -> torch.Tensor:
        scaled = (y.reshape(-1) - self.config.outcome_min) / self.bin_width
        return torch.clamp(torch.floor(scaled).long(), min=0, max=self.config.num_bins - 1)

    def _log_prob_from_logits(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        log_mass = F.log_softmax(logits, dim=-1)
        idx = self._bin_indices(y)
        if log_mass.ndim == 1:
            return log_mass[idx].reshape(-1, 1) - math.log(self.bin_width)
        return log_mass.gather(1, idx.reshape(-1, 1)) - math.log(self.bin_width)

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        n_per_row: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        idx = torch.multinomial(probs, n_per_row, replacement=True, generator=generator)
        jitter = torch.rand(idx.shape, generator=generator, device=self.device)
        y = self.config.outcome_min + (idx.to(torch.float32) + jitter) * self.bin_width
        return y.unsqueeze(-1)

    def _nuisance_logits_for_treatment(self, x: torch.Tensor, treatment: int) -> torch.Tensor:
        treatment_t = torch.full((x.shape[0],), treatment, dtype=torch.long, device=self.device)
        return self.nuisance.density_logits(x, treatment_t)

    def _target_bin_nll(self, treatment: int) -> torch.Tensor:
        return -(F.log_softmax(self.target_logits[treatment], dim=0) - math.log(self.bin_width))

    def _sample_batch(self, n: int) -> torch.Tensor:
        return torch.randint(0, n, size=(min(self.config.batch_size, n),), device=self.device)

    def fit(self, x: NDArray, treatment: NDArray, y: NDArray) -> "INFs":
        x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        treatment_t = torch.as_tensor(np.asarray(treatment, dtype=np.int64).reshape(-1), device=self.device)
        y_t = torch.as_tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1), device=self.device)
        if x_t.shape[0] != treatment_t.shape[0] or x_t.shape[0] != y_t.shape[0]:
            raise ValueError("x, treatment, and y must have the same number of rows")
        n = x_t.shape[0]
        self.diagnostics = INFsDiagnostics()

        for step in range(self.config.nuisance_steps):
            idx = self._sample_batch(n)
            xb = x_t[idx]
            tb = treatment_t[idx]
            yb = y_t[idx]
            if self.config.noise_std_x > 0.0:
                xb = xb + self.config.noise_std_x * torch.randn_like(xb)
            if self.config.noise_std_y > 0.0:
                yb = torch.clamp(
                    yb + self.config.noise_std_y * torch.randn_like(yb),
                    min=self.config.outcome_min,
                    max=np.nextafter(self.config.outcome_max, self.config.outcome_min),
                )
            density_logits, prop_logits = self.nuisance(xb, tb)
            nll = -self._log_prob_from_logits(density_logits, yb).mean()
            prop_loss = F.cross_entropy(prop_logits, tb)
            loss = nll + self.config.prop_alpha * prop_loss

            self.nuisance_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.nuisance_optimizer.step()
            if step % 10 == 0 or step == self.config.nuisance_steps - 1:
                self.diagnostics.nuisance_losses.append(float(loss.detach().cpu().item()))

        self.nuisance.eval()
        with torch.no_grad():
            prop_probs_full = torch.softmax(self.nuisance.propensity_logits(x_t), dim=1)
            assigned = prop_probs_full.gather(1, treatment_t.reshape(-1, 1))
            ipw = 1.0 / torch.clamp(assigned, min=self.config.clip_propensity)
            self.diagnostics.mean_ipw = float(ipw.mean().detach().cpu().item())
            self.diagnostics.max_ipw = float(ipw.max().detach().cpu().item())
        self.nuisance.train()

        for step in range(self.config.target_steps):
            idx = self._sample_batch(n)
            xb = x_t[idx]
            tb = treatment_t[idx]
            yb = y_t[idx]
            with torch.no_grad():
                prop_probs = torch.softmax(self.nuisance.propensity_logits(xb), dim=1)
            loss = torch.zeros((), dtype=torch.float32, device=self.device)
            for treatment_value in range(self.config.num_treatments):
                with torch.no_grad():
                    nuisance_logits = self._nuisance_logits_for_treatment(xb, treatment_value)
                    nuisance_probs = torch.softmax(nuisance_logits, dim=1)
                target_bin_nll = self._target_bin_nll(treatment_value)
                cce = torch.sum(nuisance_probs * target_bin_nll.reshape(1, -1), dim=1)
                cross_entropy = cce.mean()
                factual_nll = -self._log_prob_from_logits(self.target_logits[treatment_value], yb).reshape(-1)
                prop = prop_probs[:, treatment_value]
                mask = (tb == treatment_value) & (prop >= self.config.clip_propensity)
                weights = mask.to(torch.float32) / torch.clamp(prop, min=1e-9)
                bias_correction = torch.mean(weights * (factual_nll - cce))
                loss = loss + cross_entropy + bias_correction

            self.target_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.target_optimizer.step()
            if step % 10 == 0 or step == self.config.target_steps - 1:
                self.diagnostics.target_losses.append(float(loss.detach().cpu().item()))
        return self

    def sample_potential(
        self,
        x: NDArray,
        treatment: NDArray | int,
        n_per_x: int = 1,
        seed: Optional[int] = None,
    ) -> NDArray:
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
        samples = np.zeros((n, n_per_x, 1), dtype=np.float32)
        self.target_logits.requires_grad_(False)
        with torch.no_grad():
            for treatment_value in range(self.config.num_treatments):
                mask = treatment_arr == treatment_value
                if not np.any(mask):
                    continue
                logits = self.target_logits[treatment_value]
                y = self._sample_from_logits(logits, int(mask.sum()) * n_per_x, generator)
                samples[mask] = y.reshape(int(mask.sum()), n_per_x, 1).cpu().numpy()
        self.target_logits.requires_grad_(True)
        return samples

    def predict_potential_outcomes(self, x: NDArray, n_mc: Optional[int] = None) -> NDArray:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        probs = torch.softmax(self.target_logits.detach(), dim=1)
        means = (probs * self.bin_centers.reshape(1, -1)).sum(dim=1).cpu().numpy()
        return np.repeat(means.reshape(1, -1), x_arr.shape[0], axis=0).astype(np.float32)

    def log_prob_potential(self, x: NDArray, treatment: NDArray | int, y: NDArray) -> NDArray:
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
        treatment_t = torch.as_tensor(treatment_arr, dtype=torch.long, device=self.device)
        y_t = torch.as_tensor(y_arr, dtype=torch.float32, device=self.device)
        self.target_logits.requires_grad_(False)
        with torch.no_grad():
            logits = self.target_logits[treatment_t]
            log_prob = self._log_prob_from_logits(logits, y_t).reshape(-1).detach().cpu().numpy()
        self.target_logits.requires_grad_(True)
        return log_prob.astype(np.float32)

    def negative_log_likelihood(self, x: NDArray, treatment: NDArray | int, y: NDArray) -> float:
        return float(-np.mean(self.log_prob_potential(x, treatment, y)))

    def interventional_sample(self, treatment: int, n: int, seed: Optional[int] = None) -> NDArray:
        x_dummy = np.zeros((n, self.config.x_dim), dtype=np.float32)
        return self.sample_potential(x_dummy, treatment, n_per_x=1, seed=seed).reshape(n, 1)
