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
        layers.append(nn.ELU())
        prev = width
    layers.append(nn.Linear(prev, output_dim))
    network = nn.Sequential(*layers)
    for module in network:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    return network


class SCIGANGenerator(nn.Module):
    def __init__(self, x_dim: int, num_treatments: int, noise_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_treatments = num_treatments
        self.shared = _mlp(x_dim + num_treatments + 1 + 1 + noise_dim, (hidden_dim,), hidden_dim)
        self.heads = nn.ModuleList(
            [_mlp(hidden_dim + 1, (hidden_dim, hidden_dim), 1) for _ in range(num_treatments)]
        )

    def forward(
        self,
        x: torch.Tensor,
        t_onehot: torch.Tensor,
        dosage: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        dosage_samples: torch.Tensor,
    ) -> torch.Tensor:
        shared = self.shared(torch.cat([x, t_onehot, dosage, y, z], dim=1))
        outputs: list[torch.Tensor] = []
        for treatment in range(self.num_treatments):
            d = dosage_samples[:, treatment, :]
            shared_expand = shared.unsqueeze(1).expand(-1, d.shape[1], -1)
            head_input = torch.cat([shared_expand, d.unsqueeze(-1)], dim=-1)
            out = self.heads[treatment](head_input.reshape(-1, head_input.shape[-1]))
            outputs.append(out.reshape(x.shape[0], d.shape[1]))
        return torch.stack(outputs, dim=1)


class TreatmentDiscriminator(nn.Module):
    def __init__(self, x_dim: int, num_treatments: int, hidden_dim: int, set_dim: int) -> None:
        super().__init__()
        self.num_treatments = num_treatments
        self.patient = _mlp(x_dim, (hidden_dim,), hidden_dim)
        self.phi = nn.ModuleList([_mlp(2, (set_dim,), set_dim) for _ in range(num_treatments)])
        self.out = _mlp(hidden_dim + num_treatments * set_dim, (hidden_dim,), num_treatments)

    def forward(self, x: torch.Tensor, dosage_samples: torch.Tensor, outcomes: torch.Tensor) -> torch.Tensor:
        patient = self.patient(x)
        reps: list[torch.Tensor] = []
        for treatment in range(self.num_treatments):
            elems = torch.stack([dosage_samples[:, treatment, :], outcomes[:, treatment, :]], dim=-1)
            rep = self.phi[treatment](elems.reshape(-1, 2)).reshape(x.shape[0], elems.shape[1], -1).mean(dim=1)
            reps.append(rep)
        return self.out(torch.cat([patient] + reps, dim=1))


class DosageDiscriminator(nn.Module):
    def __init__(self, x_dim: int, num_treatments: int, hidden_dim: int, set_dim: int) -> None:
        super().__init__()
        self.num_treatments = num_treatments
        self.patient = _mlp(x_dim, (hidden_dim,), hidden_dim)
        self.local_phi = nn.ModuleList([_mlp(2, (set_dim,), set_dim) for _ in range(num_treatments)])
        self.local_out = nn.ModuleList(
            [_mlp(set_dim * 2 + hidden_dim, (hidden_dim,), 1) for _ in range(num_treatments)]
        )

    def forward(self, x: torch.Tensor, dosage_samples: torch.Tensor, outcomes: torch.Tensor) -> torch.Tensor:
        patient = self.patient(x)
        treatment_logits: list[torch.Tensor] = []
        for treatment in range(self.num_treatments):
            elems = torch.stack([dosage_samples[:, treatment, :], outcomes[:, treatment, :]], dim=-1)
            local = self.local_phi[treatment](elems.reshape(-1, 2)).reshape(x.shape[0], elems.shape[1], -1)
            global_rep = local.mean(dim=1, keepdim=True).expand_as(local)
            patient_rep = patient.unsqueeze(1).expand(-1, elems.shape[1], -1)
            head_input = torch.cat([local, global_rep, patient_rep], dim=-1)
            logits = self.local_out[treatment](head_input.reshape(-1, head_input.shape[-1]))
            treatment_logits.append(logits.reshape(x.shape[0], elems.shape[1]))
        return torch.stack(treatment_logits, dim=1)


class SCIGANInference(nn.Module):
    def __init__(self, x_dim: int, num_treatments: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_treatments = num_treatments
        self.shared = _mlp(x_dim, (hidden_dim,), hidden_dim)
        self.heads = nn.ModuleList(
            [_mlp(hidden_dim + 1, (hidden_dim, hidden_dim), 1) for _ in range(num_treatments)]
        )

    def forward(self, x: torch.Tensor, dosage_samples: torch.Tensor) -> torch.Tensor:
        shared = self.shared(x)
        outputs: list[torch.Tensor] = []
        for treatment in range(self.num_treatments):
            d = dosage_samples[:, treatment, :]
            shared_expand = shared.unsqueeze(1).expand(-1, d.shape[1], -1)
            head_input = torch.cat([shared_expand, d.unsqueeze(-1)], dim=-1)
            out = self.heads[treatment](head_input.reshape(-1, head_input.shape[-1]))
            outputs.append(out.reshape(x.shape[0], d.shape[1]))
        return torch.stack(outputs, dim=1)

    def predict(self, x: torch.Tensor, treatment: int, dosage: torch.Tensor) -> torch.Tensor:
        shared = self.shared(x)
        head_input = torch.cat([shared, dosage], dim=1)
        return self.heads[treatment](head_input)


@dataclass(slots=True)
class SCIGANConfig:
    x_dim: int
    num_treatments: int = 2
    noise_dim: int = 8
    hidden_dim: int = 64
    set_dim: int = 32
    batch_size: int = 128
    gan_iterations: int = 450
    inference_iterations: int = 700
    num_dosage_samples: int = 5
    alpha: float = 1.0
    learning_rate: float = 1e-3
    seed: int = 19
    device: str = "cpu"


class SCIGAN:
    def __init__(self, config: SCIGANConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        torch.set_num_threads(1)
        self.device = torch.device(config.device)
        self.generator = SCIGANGenerator(
            x_dim=config.x_dim,
            num_treatments=config.num_treatments,
            noise_dim=config.noise_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)
        self.treatment_discriminator = TreatmentDiscriminator(
            x_dim=config.x_dim,
            num_treatments=config.num_treatments,
            hidden_dim=config.hidden_dim,
            set_dim=config.set_dim,
        ).to(self.device)
        self.dosage_discriminator = DosageDiscriminator(
            x_dim=config.x_dim,
            num_treatments=config.num_treatments,
            hidden_dim=config.hidden_dim,
            set_dim=config.set_dim,
        ).to(self.device)
        self.inference = SCIGANInference(
            x_dim=config.x_dim,
            num_treatments=config.num_treatments,
            hidden_dim=config.hidden_dim,
        ).to(self.device)
        self.g_opt = torch.optim.Adam(self.generator.parameters(), lr=config.learning_rate)
        self.dt_opt = torch.optim.Adam(self.treatment_discriminator.parameters(), lr=config.learning_rate)
        self.dd_opt = torch.optim.Adam(self.dosage_discriminator.parameters(), lr=config.learning_rate)
        self.i_opt = torch.optim.Adam(self.inference.parameters(), lr=config.learning_rate)

    def _sample_batch(self, n: int) -> torch.Tensor:
        return torch.randint(0, n, size=(self.config.batch_size,), device=self.device)

    def _sample_noise(self, batch_size: int) -> torch.Tensor:
        return torch.rand(batch_size, self.config.noise_dim, device=self.device)

    def _sample_dosage_grids(
        self,
        factual_t: torch.Tensor,
        factual_d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = factual_t.shape[0]
        m = self.config.num_dosage_samples
        dosage_samples = torch.rand(b, self.config.num_treatments, m, device=self.device)
        positions = torch.randint(0, m, size=(b,), device=self.device)
        dosage_samples[torch.arange(b), factual_t, positions] = factual_d[:, 0]
        mask = torch.zeros_like(dosage_samples)
        mask[torch.arange(b), factual_t, positions] = 1.0
        return dosage_samples, mask

    def fit(self, x: NDArray, t: NDArray, dosage: NDArray, y: NDArray) -> "SCIGAN":
        x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        t_t = torch.as_tensor(np.asarray(t, dtype=np.int64), device=self.device)
        d_t = torch.as_tensor(np.asarray(dosage, dtype=np.float32).reshape(-1, 1), device=self.device)
        y_t = torch.as_tensor(np.asarray(y, dtype=np.float32), device=self.device)
        if y_t.ndim == 1:
            y_t = y_t[:, None]
        n = x_t.shape[0]

        for _ in range(self.config.gan_iterations):
            idx = self._sample_batch(n)
            xb, tb, db, yb = x_t[idx], t_t[idx], d_t[idx], y_t[idx]
            t_onehot = F.one_hot(tb, num_classes=self.config.num_treatments).float()
            dosage_samples, mask = self._sample_dosage_grids(tb, db)
            with torch.no_grad():
                g_out = self.generator(xb, t_onehot, db, yb, self._sample_noise(xb.shape[0]), dosage_samples)
                completed = mask * yb.unsqueeze(-1) + (1.0 - mask) * g_out

            dosage_logits = self.dosage_discriminator(xb, dosage_samples, completed)
            factual_dosage_logits = dosage_logits[torch.arange(xb.shape[0]), tb, :]
            factual_mask = mask[torch.arange(xb.shape[0]), tb, :]
            dd_loss = F.binary_cross_entropy_with_logits(factual_dosage_logits, factual_mask)
            self.dd_opt.zero_grad(set_to_none=True)
            dd_loss.backward()
            self.dd_opt.step()

            treatment_logits = self.treatment_discriminator(xb, dosage_samples, completed)
            dt_loss = F.binary_cross_entropy_with_logits(treatment_logits, t_onehot)
            self.dt_opt.zero_grad(set_to_none=True)
            dt_loss.backward()
            self.dt_opt.step()

            idx = self._sample_batch(n)
            xb, tb, db, yb = x_t[idx], t_t[idx], d_t[idx], y_t[idx]
            t_onehot = F.one_hot(tb, num_classes=self.config.num_treatments).float()
            dosage_samples, mask = self._sample_dosage_grids(tb, db)
            g_out = self.generator(xb, t_onehot, db, yb, self._sample_noise(xb.shape[0]), dosage_samples)
            completed = mask * yb.unsqueeze(-1) + (1.0 - mask) * g_out
            dosage_logits = self.dosage_discriminator(xb, dosage_samples, completed)
            treatment_logits = self.treatment_discriminator(xb, dosage_samples, completed)
            combined_prob = torch.sigmoid(dosage_logits) * torch.sigmoid(treatment_logits).unsqueeze(-1)
            g_adv = -torch.mean(
                mask * torch.log(combined_prob + 1e-7) + (1.0 - mask) * torch.log(1.0 - combined_prob + 1e-7)
            )
            factual_pred = torch.sum(mask * g_out, dim=(1, 2), keepdim=True)
            g_rec = F.mse_loss(factual_pred, yb.unsqueeze(-1))
            g_loss = g_adv + self.config.alpha * torch.sqrt(g_rec + 1e-8)
            self.g_opt.zero_grad(set_to_none=True)
            g_loss.backward()
            self.g_opt.step()

        for _ in range(self.config.inference_iterations):
            idx = self._sample_batch(n)
            xb, tb, db, yb = x_t[idx], t_t[idx], d_t[idx], y_t[idx]
            t_onehot = F.one_hot(tb, num_classes=self.config.num_treatments).float()
            dosage_samples, mask = self._sample_dosage_grids(tb, db)
            with torch.no_grad():
                g_out = self.generator(xb, t_onehot, db, yb, self._sample_noise(xb.shape[0]), dosage_samples)
                completed = mask * yb.unsqueeze(-1) + (1.0 - mask) * g_out
            i_out = self.inference(xb, dosage_samples)
            factual_pred = torch.sum(mask * i_out, dim=(1, 2), keepdim=True)
            i_loss = torch.sqrt(F.mse_loss(i_out, completed) + 1e-8) + torch.sqrt(
                F.mse_loss(factual_pred, yb.unsqueeze(-1)) + 1e-8
            )
            self.i_opt.zero_grad(set_to_none=True)
            i_loss.backward()
            self.i_opt.step()
        return self

    def predict_response(self, x: NDArray, treatment: int, dosage: NDArray | float) -> NDArray:
        x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        dosage_arr = np.asarray(dosage, dtype=np.float32).reshape(-1, 1)
        if dosage_arr.shape[0] == 1 and x_t.shape[0] > 1:
            dosage_arr = np.repeat(dosage_arr, x_t.shape[0], axis=0)
        dosage_t = torch.as_tensor(dosage_arr, dtype=torch.float32, device=self.device)
        self.inference.eval()
        with torch.no_grad():
            pred = self.inference.predict(x_t, treatment=treatment, dosage=dosage_t).cpu().numpy()
        self.inference.train()
        return pred.astype(np.float32)
