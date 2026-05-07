from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _make_mlp(input_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ELU())
        current_dim = hidden_dim
    network = nn.Sequential(*layers)
    for module in network:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
            nn.init.zeros_(module.bias)
    return network


class _DosageHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int, repeat_dosage: bool) -> None:
        super().__init__()
        self.repeat_dosage = bool(repeat_dosage)
        self.hidden_layers = nn.ModuleList()
        current_dim = feature_dim + 1
        for layer_idx in range(num_layers):
            self.hidden_layers.append(nn.Linear(current_dim, hidden_dim))
            if self.repeat_dosage:
                current_dim = hidden_dim + 1
            else:
                current_dim = hidden_dim
        final_dim = hidden_dim + 1 if self.repeat_dosage else hidden_dim
        self.output = nn.Linear(final_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.output.weight, -0.05, 0.05)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor, dosage: torch.Tensor) -> torch.Tensor:
        dosage = torch.clamp(dosage.reshape(-1, 1), 0.0, 1.0)
        scaled_dosage = (dosage - 0.5) * 2.0
        out = torch.cat([features, scaled_dosage], dim=1)
        for layer in self.hidden_layers:
            out = F.elu(layer(out))
            if self.repeat_dosage:
                out = torch.cat([out, scaled_dosage], dim=1)
        return self.output(out)


class DRNetModule(nn.Module):
    def __init__(
        self,
        x_dim: int,
        num_treatments: int,
        hidden_dim: int,
        num_strata: int,
        base_layers: int,
        treatment_layers: int,
        head_layers: int,
        repeat_dosage: bool,
    ) -> None:
        super().__init__()
        self.num_treatments = int(num_treatments)
        self.num_strata = int(num_strata)
        self.base = _make_mlp(x_dim, hidden_dim, base_layers)
        self.treatment_layers = nn.ModuleList(
            [_make_mlp(hidden_dim, hidden_dim, treatment_layers) for _ in range(self.num_treatments)]
        )
        self.heads = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _DosageHead(
                            feature_dim=hidden_dim,
                            hidden_dim=hidden_dim,
                            num_layers=head_layers,
                            repeat_dosage=repeat_dosage,
                        )
                        for _ in range(self.num_strata)
                    ]
                )
                for _ in range(self.num_treatments)
            ]
        )

    def _strata(self, dosage: torch.Tensor) -> torch.Tensor:
        dosage = torch.clamp(dosage.reshape(-1), 0.0, torch.nextafter(torch.tensor(1.0, device=dosage.device), torch.tensor(0.0, device=dosage.device)))
        return torch.floor(dosage * self.num_strata).long().clamp(0, self.num_strata - 1)

    def forward(self, x: torch.Tensor, treatment: torch.Tensor, dosage: torch.Tensor) -> torch.Tensor:
        shared = self.base(x)
        treatment = treatment.reshape(-1).long()
        strata = self._strata(dosage)
        out = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        for treatment_value in range(self.num_treatments):
            treatment_mask = treatment == treatment_value
            if not bool(treatment_mask.any()):
                continue
            treatment_idx = torch.nonzero(treatment_mask, as_tuple=False).reshape(-1)
            treatment_hidden = self.treatment_layers[treatment_value](shared[treatment_idx])
            treatment_strata = strata[treatment_idx]
            treatment_dosage = dosage.reshape(-1)[treatment_idx]
            for stratum in range(self.num_strata):
                stratum_mask = treatment_strata == stratum
                if not bool(stratum_mask.any()):
                    continue
                local_idx = torch.nonzero(stratum_mask, as_tuple=False).reshape(-1)
                global_idx = treatment_idx[local_idx]
                out[global_idx] = self.heads[treatment_value][stratum](
                    treatment_hidden[local_idx],
                    treatment_dosage[local_idx],
                )
        return out


@dataclass(slots=True)
class DRNetConfig:
    x_dim: int
    num_treatments: int = 2
    hidden_dim: int = 64
    num_strata: int = 5
    base_layers: int = 2
    treatment_layers: int = 1
    head_layers: int = 2
    repeat_dosage: bool = True
    batch_size: int = 128
    num_steps: int = 1_200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    standardize_outcome: bool = True
    max_predict_batch: int = 65_536
    seed: int = 29
    device: str = "cpu"


class DRNet:
    """Dose Response Network baseline for discrete treatments with continuous dosage."""

    def __init__(self, config: DRNetConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        torch.set_num_threads(1)
        self.device = torch.device(config.device)
        self.model = DRNetModule(
            x_dim=config.x_dim,
            num_treatments=config.num_treatments,
            hidden_dim=config.hidden_dim,
            num_strata=config.num_strata,
            base_layers=config.base_layers,
            treatment_layers=config.treatment_layers,
            head_layers=config.head_layers,
            repeat_dosage=config.repeat_dosage,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.y_mean = 0.0
        self.y_scale = 1.0
        self.loss_history: list[float] = []

    def _sample_batch(self, n: int) -> torch.Tensor:
        batch_size = min(self.config.batch_size, n)
        return torch.randint(0, n, size=(batch_size,), device=self.device)

    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if not self.config.standardize_outcome:
            self.y_mean = 0.0
            self.y_scale = 1.0
            return y_arr
        self.y_mean = float(np.mean(y_arr))
        scale = float(np.std(y_arr))
        self.y_scale = scale if scale > 1e-6 else 1.0
        return ((y_arr - self.y_mean) / self.y_scale).astype(np.float32)

    def fit(self, x: np.ndarray, treatment: np.ndarray, dosage: np.ndarray, y: np.ndarray) -> "DRNet":
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        treatment_arr = np.asarray(treatment, dtype=np.int64).reshape(-1)
        dosage_arr = np.asarray(dosage, dtype=np.float32).reshape(-1)
        y_arr = self._normalize_y(y)
        if x_arr.shape[0] != treatment_arr.shape[0] or x_arr.shape[0] != dosage_arr.shape[0] or x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("x, treatment, dosage, and y must have the same number of rows")
        if x_arr.shape[1] != self.config.x_dim:
            raise ValueError(f"x has {x_arr.shape[1]} columns, expected {self.config.x_dim}")

        x_t = torch.as_tensor(x_arr, dtype=torch.float32, device=self.device)
        treatment_t = torch.as_tensor(treatment_arr, dtype=torch.long, device=self.device)
        dosage_t = torch.as_tensor(np.clip(dosage_arr, 0.0, 1.0), dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y_arr, dtype=torch.float32, device=self.device)
        n = x_t.shape[0]

        for step in range(self.config.num_steps):
            idx = self._sample_batch(n)
            pred = self.model(x_t[idx], treatment_t[idx], dosage_t[idx])
            loss = F.mse_loss(pred, y_t[idx])
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            if step % 25 == 0 or step == self.config.num_steps - 1:
                self.loss_history.append(float(loss.detach().cpu().item()))
        return self

    def predict_response(self, x: np.ndarray, treatment: np.ndarray | int, dosage: np.ndarray | float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        n = x_arr.shape[0]
        treatment_arr = np.asarray(treatment, dtype=np.int64).reshape(-1)
        dosage_arr = np.asarray(dosage, dtype=np.float32).reshape(-1)
        if treatment_arr.size == 1 and n > 1:
            treatment_arr = np.full(n, int(treatment_arr.item()), dtype=np.int64)
        if dosage_arr.size == 1 and n > 1:
            dosage_arr = np.full(n, float(dosage_arr.item()), dtype=np.float32)
        if treatment_arr.shape[0] != n or dosage_arr.shape[0] != n:
            raise ValueError("treatment and dosage must be scalar or align with x")

        outputs: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            start = 0
            while start < n:
                stop = min(n, start + self.config.max_predict_batch)
                xb = torch.as_tensor(x_arr[start:stop], dtype=torch.float32, device=self.device)
                tb = torch.as_tensor(treatment_arr[start:stop], dtype=torch.long, device=self.device)
                db = torch.as_tensor(np.clip(dosage_arr[start:stop], 0.0, 1.0), dtype=torch.float32, device=self.device)
                outputs.append(self.model(xb, tb, db).cpu().numpy())
                start = stop
        self.model.train()
        pred = np.concatenate(outputs, axis=0)
        pred = pred * self.y_scale + self.y_mean
        return pred.astype(np.float32)
