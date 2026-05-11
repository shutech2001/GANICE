from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

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
        layers.append(nn.ReLU())
        prev = width
    layers.append(nn.Linear(prev, output_dim))
    network = nn.Sequential(*layers)
    for module in network:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    return network


class _BoundedHead(nn.Module):
    def __init__(self, hidden_dim: int, outcome_min: float, outcome_max: float) -> None:
        super().__init__()
        self.network = _mlp(hidden_dim, (hidden_dim,), 1)
        self.outcome_min = float(outcome_min)
        self.outcome_max = float(outcome_max)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        raw = self.network(h)
        return self.outcome_min + (self.outcome_max - self.outcome_min) * torch.sigmoid(raw)


class CounterfactualGenerator(nn.Module):
    def __init__(
        self, x_dim: int, num_treatments: int, hidden_dim: int, outcome_min: float, outcome_max: float
    ) -> None:
        super().__init__()
        self.shared = _mlp(x_dim + num_treatments + 1, (hidden_dim, hidden_dim), hidden_dim)
        self.heads = nn.ModuleList(
            [_BoundedHead(hidden_dim, outcome_min, outcome_max) for _ in range(num_treatments)]
        )

    def forward(self, x: torch.Tensor, t_onehot: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shared = self.shared(torch.cat([x, t_onehot, y], dim=1))
        return torch.cat([head(shared) for head in self.heads], dim=1)


class CounterfactualDiscriminator(nn.Module):
    def __init__(self, x_dim: int, num_treatments: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = _mlp(x_dim + num_treatments, (hidden_dim, hidden_dim), num_treatments)

    def forward(self, x: torch.Tensor, y_bar: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([x, y_bar], dim=1))


class ITEInferenceNetwork(nn.Module):
    def __init__(
        self, x_dim: int, num_treatments: int, hidden_dim: int, outcome_min: float, outcome_max: float
    ) -> None:
        super().__init__()
        self.shared = _mlp(x_dim, (hidden_dim, hidden_dim), hidden_dim)
        self.heads = nn.ModuleList(
            [_BoundedHead(hidden_dim, outcome_min, outcome_max) for _ in range(num_treatments)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared(x)
        return torch.cat([head(shared) for head in self.heads], dim=1)


class ITEDiscriminator(nn.Module):
    def __init__(self, x_dim: int, num_treatments: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = _mlp(x_dim + num_treatments, (hidden_dim, hidden_dim), 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([x, y], dim=1))


@dataclass(slots=True)
class GANITEConfig:
    x_dim: int
    num_treatments: int = 2
    hidden_dim: int = 64
    batch_size: int = 128
    cf_iterations: int = 500
    ite_iterations: int = 500
    cf_discriminator_steps: int = 2
    ite_discriminator_steps: int = 1
    alpha: float = 1.0
    beta: float = 1.0
    learning_rate: float = 1e-3
    outcome_min: float = 0.0
    outcome_max: float = 1.0
    seed: int = 31
    device: str = "cpu"


class GANITE:
    def __init__(self, config: GANITEConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        torch.set_num_threads(1)
        self.device = torch.device(config.device)
        self.generator = CounterfactualGenerator(
            x_dim=config.x_dim,
            num_treatments=config.num_treatments,
            hidden_dim=config.hidden_dim,
            outcome_min=config.outcome_min,
            outcome_max=config.outcome_max,
        ).to(self.device)
        self.cf_discriminator = CounterfactualDiscriminator(
            x_dim=config.x_dim,
            num_treatments=config.num_treatments,
            hidden_dim=config.hidden_dim,
        ).to(self.device)
        self.inference = ITEInferenceNetwork(
            x_dim=config.x_dim,
            num_treatments=config.num_treatments,
            hidden_dim=config.hidden_dim,
            outcome_min=config.outcome_min,
            outcome_max=config.outcome_max,
        ).to(self.device)
        self.ite_discriminator = ITEDiscriminator(
            x_dim=config.x_dim,
            num_treatments=config.num_treatments,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=config.learning_rate)
        self.d_optimizer = torch.optim.Adam(self.cf_discriminator.parameters(), lr=config.learning_rate)
        self.i_optimizer = torch.optim.Adam(self.inference.parameters(), lr=config.learning_rate)
        self.id_optimizer = torch.optim.Adam(self.ite_discriminator.parameters(), lr=config.learning_rate)

    def _sample_batch(self, n: int) -> torch.Tensor:
        indices = torch.randint(0, n, size=(self.config.batch_size,), device=self.device)
        return indices

    def fit(self, x: NDArray[np.float32], t: NDArray[np.int64], y: NDArray[np.float32]) -> "GANITE":
        x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        t_t = torch.as_tensor(np.asarray(t, dtype=np.int64), device=self.device)
        y_t = torch.as_tensor(np.asarray(y, dtype=np.float32), device=self.device)
        if y_t.ndim == 1:
            y_t = y_t[:, None]
        n = x_t.shape[0]

        for _ in range(self.config.cf_iterations):
            for _ in range(self.config.cf_discriminator_steps):
                idx = self._sample_batch(n)
                xb, tb, yb = x_t[idx], t_t[idx], y_t[idx]
                t_onehot = F.one_hot(tb, num_classes=self.config.num_treatments).float()
                with torch.no_grad():
                    y_tilde = self.generator(xb, t_onehot, yb)
                    y_bar = y_tilde * (1.0 - t_onehot) + yb * t_onehot
                logits = self.cf_discriminator(xb, y_bar)
                d_loss = F.cross_entropy(logits, tb)
                self.d_optimizer.zero_grad(set_to_none=True)
                d_loss.backward()
                self.d_optimizer.step()

            idx = self._sample_batch(n)
            xb, tb, yb = x_t[idx], t_t[idx], y_t[idx]
            t_onehot = F.one_hot(tb, num_classes=self.config.num_treatments).float()
            y_tilde = self.generator(xb, t_onehot, yb)
            y_bar = y_tilde * (1.0 - t_onehot) + yb * t_onehot
            logits = self.cf_discriminator(xb, y_bar)
            g_adv = -F.cross_entropy(logits, tb)
            factual = torch.sum(y_tilde * t_onehot, dim=1, keepdim=True)
            g_rec = F.mse_loss(factual, yb)
            g_loss = g_adv + self.config.alpha * g_rec
            self.g_optimizer.zero_grad(set_to_none=True)
            g_loss.backward()
            self.g_optimizer.step()

        for _ in range(self.config.ite_iterations):
            for _ in range(self.config.ite_discriminator_steps):
                idx = self._sample_batch(n)
                xb, tb, yb = x_t[idx], t_t[idx], y_t[idx]
                t_onehot = F.one_hot(tb, num_classes=self.config.num_treatments).float()
                with torch.no_grad():
                    y_tilde = self.generator(xb, t_onehot, yb)
                    y_bar = y_tilde * (1.0 - t_onehot) + yb * t_onehot
                    y_hat = self.inference(xb)
                real_logits = self.ite_discriminator(xb, y_bar)
                fake_logits = self.ite_discriminator(xb, y_hat)
                d_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                d_loss += F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
                self.id_optimizer.zero_grad(set_to_none=True)
                d_loss.backward()
                self.id_optimizer.step()

            idx = self._sample_batch(n)
            xb, tb, yb = x_t[idx], t_t[idx], y_t[idx]
            t_onehot = F.one_hot(tb, num_classes=self.config.num_treatments).float()
            with torch.no_grad():
                y_tilde = self.generator(xb, t_onehot, yb)
                y_bar = y_tilde * (1.0 - t_onehot) + yb * t_onehot
            y_hat = self.inference(xb)
            fake_logits = self.ite_discriminator(xb, y_hat)
            i_adv = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
            if self.config.num_treatments == 2:
                sup_target = y_bar[:, 1] - y_bar[:, 0]
                sup_pred = y_hat[:, 1] - y_hat[:, 0]
                i_sup = F.mse_loss(sup_pred, sup_target)
            else:
                i_sup = F.mse_loss(y_hat, y_bar)
            i_loss = i_adv + self.config.beta * i_sup
            self.i_optimizer.zero_grad(set_to_none=True)
            i_loss.backward()
            self.i_optimizer.step()
        return self

    def predict_potential_outcomes(
        self, x: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        x_t = torch.as_tensor(
            np.asarray(x, dtype=np.float32), device=self.device
        )
        self.inference.eval()
        with torch.no_grad():
            y_hat = self.inference(x_t).cpu().numpy()
        self.inference.train()
        return y_hat.astype(np.float32)
