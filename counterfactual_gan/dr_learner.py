from __future__ import annotations

from dataclasses import dataclass, field

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


class _BoundedOutcomeHead(nn.Module):
    def __init__(self, hidden_dim: int, num_treatments: int, outcome_min: float, outcome_max: float) -> None:
        super().__init__()
        self.output = nn.Linear(hidden_dim, num_treatments)
        self.outcome_min = float(outcome_min)
        self.outcome_max = float(outcome_max)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.output(features)
        return self.outcome_min + (self.outcome_max - self.outcome_min) * torch.sigmoid(raw)


class _DRNuisanceNet(nn.Module):
    def __init__(
        self,
        x_dim: int,
        num_treatments: int,
        hidden_dim: int,
        outcome_min: float,
        outcome_max: float,
    ) -> None:
        super().__init__()
        self.num_treatments = int(num_treatments)
        self.shared = _mlp(x_dim, (hidden_dim, hidden_dim), hidden_dim)
        self.propensity = nn.Linear(hidden_dim, num_treatments)
        self.outcome = _BoundedOutcomeHead(hidden_dim, num_treatments, outcome_min, outcome_max)
        nn.init.xavier_uniform_(self.propensity.weight)
        nn.init.zeros_(self.propensity.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        return self.outcome(features), self.propensity(features)


@dataclass(slots=True)
class DRLearnerConfig:
    x_dim: int
    num_treatments: int = 2
    hidden_dim: int = 64
    batch_size: int = 128
    nuisance_steps: int = 550
    final_steps: int = 700
    n_folds: int = 2
    nuisance_lr: float = 1e-3
    final_lr: float = 1e-3
    weight_decay: float = 1e-5
    outcome_loss_weight: float = 1.0
    propensity_loss_weight: float = 0.5
    propensity_clip: float = 0.05
    outcome_min: float = 0.0
    outcome_max: float = 1.0
    standardize_pseudo_outcome: bool = True
    seed: int = 73
    device: str = "cpu"


@dataclass(slots=True)
class DRLearnerDiagnostics:
    nuisance_losses: list[list[float]] = field(default_factory=list)
    final_losses: list[float] = field(default_factory=list)
    pseudo_outcome_mean: float = 0.0
    pseudo_outcome_std: float = 1.0
    mean_propensity: float = 0.0
    min_propensity: float = 0.0
    max_propensity: float = 0.0


class DRLearner:
    """Cross-fitted binary-treatment DR-Learner from Kennedy (2020).

    The estimator follows the pseudo-outcome in Algorithm 1 of the TeX source:
    ``(A - pi(X)) / (pi(X) * (1 - pi(X))) * (Y - mu_A(X)) + mu_1(X) - mu_0(X)``.
    It is a CATE baseline, so ``sample_potential`` returns degenerate samples at
    the reconstructed potential-outcome means rather than a learned distribution.
    """

    def __init__(self, config: DRLearnerConfig) -> None:
        if config.num_treatments != 2:
            raise ValueError("DRLearner currently implements the binary-treatment DR-Learner")
        if config.n_folds < 2:
            raise ValueError("n_folds must be at least 2 for cross-fitting")
        self.config = config
        torch.manual_seed(config.seed)
        torch.set_num_threads(1)
        self.device = torch.device(config.device)
        self.nuisance_models: list[_DRNuisanceNet] = []
        self.final_model = _mlp(config.x_dim, (config.hidden_dim, config.hidden_dim), 1).to(self.device)
        self.final_optimizer = torch.optim.AdamW(
            self.final_model.parameters(),
            lr=config.final_lr,
            weight_decay=config.weight_decay,
        )
        self.pseudo_mean = 0.0
        self.pseudo_std = 1.0
        self.diagnostics = DRLearnerDiagnostics()

    def _new_nuisance_model(self, seed: int) -> tuple[_DRNuisanceNet, torch.optim.Optimizer]:
        torch.manual_seed(seed)
        model = _DRNuisanceNet(
            x_dim=self.config.x_dim,
            num_treatments=self.config.num_treatments,
            hidden_dim=self.config.hidden_dim,
            outcome_min=self.config.outcome_min,
            outcome_max=self.config.outcome_max,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.nuisance_lr,
            weight_decay=self.config.weight_decay,
        )
        return model, optimizer

    def _sample_batch(self, indices: torch.Tensor) -> torch.Tensor:
        batch_size = min(self.config.batch_size, indices.shape[0])
        selected = torch.randint(0, indices.shape[0], size=(batch_size,), device=self.device)
        return indices[selected]

    def _fit_nuisance_fold(
        self,
        x: torch.Tensor,
        treatment: torch.Tensor,
        y: torch.Tensor,
        train_indices: torch.Tensor,
        fold: int,
    ) -> tuple[_DRNuisanceNet, list[float]]:
        model, optimizer = self._new_nuisance_model(self.config.seed + 10_000 + fold)
        losses: list[float] = []
        for step in range(self.config.nuisance_steps):
            idx = self._sample_batch(train_indices)
            xb = x[idx]
            tb = treatment[idx]
            yb = y[idx]
            mu_hat, prop_logits = model(xb)
            factual_mu = mu_hat.gather(1, tb.reshape(-1, 1))
            outcome_loss = F.mse_loss(factual_mu, yb)
            propensity_loss = F.cross_entropy(prop_logits, tb)
            loss = (
                self.config.outcome_loss_weight * outcome_loss
                + self.config.propensity_loss_weight * propensity_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 25 == 0 or step == self.config.nuisance_steps - 1:
                losses.append(float(loss.detach().cpu().item()))
        return model, losses

    def _predict_nuisance_tensor(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.nuisance_models:
            raise RuntimeError("DRLearner must be fit before prediction")
        mu_sum = torch.zeros(x.shape[0], self.config.num_treatments, dtype=x.dtype, device=self.device)
        prop_sum = torch.zeros_like(mu_sum)
        with torch.no_grad():
            for model in self.nuisance_models:
                model.eval()
                mu_hat, prop_logits = model(x)
                mu_sum += mu_hat
                prop_sum += torch.softmax(prop_logits, dim=1)
                model.train()
        scale = float(len(self.nuisance_models))
        return mu_sum / scale, prop_sum / scale

    def fit(self, x: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> "DRLearner":
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        treatment_arr = np.asarray(treatment, dtype=np.int64).reshape(-1)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if x_arr.shape[0] != treatment_arr.shape[0] or x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("x, treatment, and y must have the same number of rows")
        if x_arr.shape[1] != self.config.x_dim:
            raise ValueError(f"x has {x_arr.shape[1]} columns, expected {self.config.x_dim}")

        x_t = torch.as_tensor(x_arr, dtype=torch.float32, device=self.device)
        treatment_t = torch.as_tensor(treatment_arr, dtype=torch.long, device=self.device)
        y_t = torch.as_tensor(y_arr, dtype=torch.float32, device=self.device)
        n = x_t.shape[0]
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.config.seed + 1)
        permutation = torch.randperm(n, generator=generator, device=self.device)
        folds = torch.tensor_split(permutation, self.config.n_folds)

        pseudo = torch.zeros(n, 1, dtype=torch.float32, device=self.device)
        self.nuisance_models = []
        self.diagnostics = DRLearnerDiagnostics()
        propensity_for_diag = torch.zeros(n, 2, dtype=torch.float32, device=self.device)

        for fold_idx, heldout_indices in enumerate(folds):
            train_indices = torch.cat([fold for idx, fold in enumerate(folds) if idx != fold_idx])
            model, losses = self._fit_nuisance_fold(x_t, treatment_t, y_t, train_indices, fold_idx)
            self.nuisance_models.append(model)
            self.diagnostics.nuisance_losses.append(losses)
            model.eval()
            with torch.no_grad():
                mu_hat, prop_logits = model(x_t[heldout_indices])
                probs = torch.softmax(prop_logits, dim=1)
                propensity_for_diag[heldout_indices] = probs
                pi = torch.clamp(
                    probs[:, 1:2],
                    min=self.config.propensity_clip,
                    max=1.0 - self.config.propensity_clip,
                )
                a = treatment_t[heldout_indices].to(torch.float32).reshape(-1, 1)
                mu_a = mu_hat.gather(1, treatment_t[heldout_indices].reshape(-1, 1))
                pseudo[heldout_indices] = ((a - pi) / (pi * (1.0 - pi))) * (y_t[heldout_indices] - mu_a)
                pseudo[heldout_indices] += mu_hat[:, 1:2] - mu_hat[:, 0:1]
            model.train()

        if self.config.standardize_pseudo_outcome:
            self.pseudo_mean = float(pseudo.mean().detach().cpu().item())
            self.pseudo_std = float(torch.clamp(pseudo.std(unbiased=False), min=1e-3).detach().cpu().item())
            pseudo_target = (pseudo - self.pseudo_mean) / self.pseudo_std
        else:
            self.pseudo_mean = 0.0
            self.pseudo_std = 1.0
            pseudo_target = pseudo
        self.diagnostics.pseudo_outcome_mean = self.pseudo_mean
        self.diagnostics.pseudo_outcome_std = self.pseudo_std
        pi_diag = propensity_for_diag[:, 1]
        self.diagnostics.mean_propensity = float(pi_diag.mean().detach().cpu().item())
        self.diagnostics.min_propensity = float(pi_diag.min().detach().cpu().item())
        self.diagnostics.max_propensity = float(pi_diag.max().detach().cpu().item())

        batch_size = min(self.config.batch_size, n)
        for step in range(self.config.final_steps):
            idx = torch.randint(0, n, size=(batch_size,), device=self.device)
            pred = self.final_model(x_t[idx])
            loss = F.mse_loss(pred, pseudo_target[idx])
            self.final_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.final_optimizer.step()
            if step % 25 == 0 or step == self.config.final_steps - 1:
                self.diagnostics.final_losses.append(float(loss.detach().cpu().item()))
        return self

    def predict_cate(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        outputs: list[np.ndarray] = []
        self.final_model.eval()
        with torch.no_grad():
            start = 0
            while start < x_arr.shape[0]:
                stop = min(start + 65_536, x_arr.shape[0])
                xb = torch.as_tensor(x_arr[start:stop], dtype=torch.float32, device=self.device)
                pred = self.final_model(xb) * self.pseudo_std + self.pseudo_mean
                outputs.append(pred.cpu().numpy())
                start = stop
        self.final_model.train()
        tau = np.concatenate(outputs, axis=0).reshape(-1)
        lower = self.config.outcome_min - self.config.outcome_max
        upper = self.config.outcome_max - self.config.outcome_min
        return np.clip(tau, lower, upper).astype(np.float32)

    def _predict_average_nuisance(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        outputs_mu: list[np.ndarray] = []
        outputs_prop: list[np.ndarray] = []
        start = 0
        while start < x_arr.shape[0]:
            stop = min(start + 65_536, x_arr.shape[0])
            xb = torch.as_tensor(x_arr[start:stop], dtype=torch.float32, device=self.device)
            mu_hat, prop_hat = self._predict_nuisance_tensor(xb)
            outputs_mu.append(mu_hat.cpu().numpy())
            outputs_prop.append(prop_hat.cpu().numpy())
            start = stop
        return np.concatenate(outputs_mu, axis=0), np.concatenate(outputs_prop, axis=0)

    def predict_potential_outcomes(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        tau = self.predict_cate(x_arr).reshape(-1, 1)
        mu_hat, prop_hat = self._predict_average_nuisance(x_arr)
        pi = np.clip(prop_hat[:, 1:2], self.config.propensity_clip, 1.0 - self.config.propensity_clip)
        observed_regression = (1.0 - pi) * mu_hat[:, 0:1] + pi * mu_hat[:, 1:2]
        mu0 = observed_regression - pi * tau
        mu1 = observed_regression + (1.0 - pi) * tau
        mu = np.concatenate([mu0, mu1], axis=1)
        return np.clip(mu, self.config.outcome_min, self.config.outcome_max).astype(np.float32)

    def sample_potential(
        self,
        x: np.ndarray,
        treatment: np.ndarray | int,
        n_per_x: int = 1,
        seed: int | None = None,
    ) -> np.ndarray:
        del seed
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        n = x_arr.shape[0]
        treatment_arr = np.asarray(treatment, dtype=np.int64).reshape(-1)
        if treatment_arr.size == 1:
            treatment_arr = np.full(n, int(treatment_arr.item()), dtype=np.int64)
        if treatment_arr.shape[0] != n:
            raise ValueError("treatment must be scalar or align with x")
        mu = self.predict_potential_outcomes(x_arr)
        selected = mu[np.arange(n), treatment_arr].reshape(n, 1, 1)
        return np.repeat(selected, n_per_x, axis=1).astype(np.float32)
