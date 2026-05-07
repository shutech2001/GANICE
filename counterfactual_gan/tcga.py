from __future__ import annotations

from dataclasses import dataclass, field
import io
import sqlite3
import urllib.request
from pathlib import Path
from typing import Literal

import numpy as np

from .utils import ensure_row_matrix, make_rng


TCGA_DB_URL = "https://paperdatasets.s3.amazonaws.com/tcga.db"


def _tcga_db_is_readable(db_path: Path) -> bool:
    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='rnaseq';"
            ).fetchone()
            if row is None:
                return False
            count = connection.execute("SELECT COUNT(*) FROM rnaseq;").fetchone()
        finally:
            connection.close()
    except sqlite3.DatabaseError:
        return False
    return bool(count and int(count[0]) > 0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _decode_sqlite_array(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytes):
        with io.BytesIO(value) as handle:
            return np.load(handle, allow_pickle=False).astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def download_tcga_db(
    data_dir: str | Path,
    *,
    url: str = TCGA_DB_URL,
    force: bool = False,
) -> Path:
    """Download the DRNet TCGA SQLite database into data_dir."""

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    db_path = data_path / "tcga.db"
    if db_path.exists() and not force:
        if _tcga_db_is_readable(db_path):
            return db_path
        raise ValueError(f"{db_path} exists but is not a readable TCGA SQLite database")
    tmp_path = db_path.with_suffix(".db.part")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp_path.replace(db_path)
    return db_path


def _load_rnaseq_matrix(db_path: Path) -> tuple[list[str], np.ndarray]:
    sqlite3.register_converter("ARRAY", _decode_sqlite_array)
    connection = sqlite3.connect(str(db_path), detect_types=sqlite3.PARSE_DECLTYPES)
    try:
        connection.execute("PRAGMA query_only = ON;")
        clinical_rows = connection.execute(
            "SELECT id FROM clinical WHERE id IN (SELECT clinical_id FROM rnaseq) ORDER BY rowid;"
        ).fetchall()
        rnaseq_rows = connection.execute(
            "SELECT clinical_id, data FROM rnaseq WHERE clinical_id IN (SELECT id FROM clinical) ORDER BY rowid;"
        ).fetchall()
    finally:
        connection.close()
    if not clinical_rows or not rnaseq_rows:
        raise ValueError(f"no RNA-seq rows found in {db_path}")
    id_to_data: dict[str, np.ndarray] = {}
    for clinical_id, data in rnaseq_rows:
        id_to_data[str(clinical_id)] = _decode_sqlite_array(data)
    ids = [str(row[0]) for row in clinical_rows if str(row[0]) in id_to_data]
    matrix = np.vstack([id_to_data[clinical_id].reshape(1, -1) for clinical_id in ids]).astype(np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return ids, matrix


def extract_tcga_gene_expression(
    data_dir: str | Path,
    *,
    num_features: int = 4000,
    cache_name: str | None = None,
    force: bool = False,
    log_normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the SCIGAN/DRNet TCGA RNA-seq feature matrix.

    The DRNet release stores raw samples in ``tcga.db`` as numpy arrays inside
    SQLite.  This function reads that database with sqlite3, applies the
    TCGA-Dose preprocessing used in the experiment protocol, and caches the
    resulting 4000 most-variable-gene matrix.
    """

    data_path = Path(data_dir)
    db_path = data_path / "tcga.db"
    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path} does not exist. Run with --tcga-download or place tcga.db in {data_path}."
        )
    if cache_name is None:
        cache_name = f"tcga_rnaseq_clinical_top{num_features}.npz"
    cache_path = data_path / cache_name
    if cache_path.exists() and not force:
        cached = np.load(cache_path)
        return cached["features"].astype(np.float32), cached["selected_gene_indices"].astype(np.int64)

    _, matrix = _load_rnaseq_matrix(db_path)
    if log_normalize:
        matrix = np.log1p(np.maximum(matrix, 0.0)).astype(np.float32)

    min_val = np.min(matrix, axis=0, keepdims=True)
    max_val = np.max(matrix, axis=0, keepdims=True)
    matrix = (matrix - min_val) / (max_val - min_val + 1e-5)
    variances = np.var(matrix, axis=0)
    keep = min(int(num_features), matrix.shape[1])
    selected = np.argsort(variances)[-keep:][::-1].astype(np.int64)
    matrix = matrix[:, selected].astype(np.float32, copy=False)
    row_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(row_norm, 1e-8)
    np.savez_compressed(cache_path, features=matrix.astype(np.float32), selected_gene_indices=selected)
    return matrix.astype(np.float32), selected


@dataclass(slots=True)
class TCGADoseData:
    x_raw: np.ndarray
    x: np.ndarray
    z: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    selected_gene_indices: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    pc_components: np.ndarray


@dataclass(slots=True)
class TCGADoseDGP:
    """TCGA-Dose semi-synthetic benchmark on DRNet's TCGA SQLite data."""

    data_dir: Path | str = Path("Data/TCGA")
    num_features: int = 4000
    num_treatments: int = 3
    pc_dim: int = 8
    gamma_a: float = 1.0
    alpha_d: float = 8.0
    seed: int = 991
    data: TCGADoseData = field(init=False)
    v: np.ndarray = field(init=False)
    r: np.ndarray = field(init=False)
    theta: np.ndarray = field(init=False)
    q: np.ndarray = field(init=False)
    s: np.ndarray = field(init=False)
    u1: np.ndarray = field(init=False)
    u2: np.ndarray = field(init=False)
    lambda_a: np.ndarray = field(init=False)
    rho_a: np.ndarray = field(init=False)
    intercept: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        features, selected = extract_tcga_gene_expression(self.data_dir, num_features=self.num_features)
        self.data = self._make_data(features, selected)
        self._make_coefficients()

    @property
    def feature_dim(self) -> int:
        return int(self.data.x.shape[1])

    @property
    def d_w(self) -> int:
        return self.feature_dim + 2

    @property
    def dosage_grid(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, 21, dtype=np.float32)

    def _make_data(self, features: np.ndarray, selected: np.ndarray) -> TCGADoseData:
        rng = make_rng(self.seed)
        n = features.shape[0]
        perm = rng.permutation(n)
        n_train = int(round(0.64 * n))
        n_val = int(round(0.16 * n))
        train_idx = perm[:n_train]
        val_idx = perm[n_train : n_train + n_val]
        test_idx = perm[n_train + n_val :]

        train = features[train_idx]
        mean = train.mean(axis=0, keepdims=True)
        scale = train.std(axis=0, keepdims=True)
        scale = np.where(scale > 1e-6, scale, 1.0)
        x = ((features - mean) / scale).astype(np.float32)

        train_centered = x[train_idx] - x[train_idx].mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(train_centered, full_matrices=False)
        components = vt[: self.pc_dim].astype(np.float32)
        z = x @ components.T
        z_mean = z[train_idx].mean(axis=0, keepdims=True)
        z_scale = z[train_idx].std(axis=0, keepdims=True)
        z = ((z - z_mean) / np.maximum(z_scale, 1e-6)).astype(np.float32)

        return TCGADoseData(
            x_raw=features.astype(np.float32),
            x=x.astype(np.float32),
            z=z,
            train_idx=train_idx.astype(np.int64),
            val_idx=val_idx.astype(np.int64),
            test_idx=test_idx.astype(np.int64),
            selected_gene_indices=selected.astype(np.int64),
            feature_mean=mean.astype(np.float32),
            feature_scale=scale.astype(np.float32),
            pc_components=components,
        )

    def _make_coefficients(self) -> None:
        rng = make_rng(self.seed + 700)
        scale = 1.0 / np.sqrt(float(self.pc_dim))
        shape = (self.num_treatments, self.pc_dim)
        self.v = rng.normal(0.0, 0.9 * scale, size=shape).astype(np.float32)
        self.r = rng.normal(0.0, 1.0 * scale, size=shape).astype(np.float32)
        self.theta = rng.normal(0.0, 0.55 * scale, size=shape).astype(np.float32)
        self.q = rng.normal(0.0, 0.75 * scale, size=shape).astype(np.float32)
        self.s = rng.normal(0.0, 0.70 * scale, size=shape).astype(np.float32)
        self.u1 = rng.normal(0.0, 0.65 * scale, size=shape).astype(np.float32)
        self.u2 = rng.normal(0.0, 0.65 * scale, size=shape).astype(np.float32)
        self.lambda_a = rng.uniform(0.8, 1.5, size=self.num_treatments).astype(np.float32)
        self.rho_a = rng.uniform(-0.30, 0.30, size=self.num_treatments).astype(np.float32)
        self.intercept = rng.normal(0.0, 0.12, size=self.num_treatments).astype(np.float32)

    def split_arrays(self, split: Literal["train", "val", "test"]) -> tuple[np.ndarray, np.ndarray]:
        if split == "train":
            idx = self.data.train_idx
        elif split == "val":
            idx = self.data.val_idx
        elif split == "test":
            idx = self.data.test_idx
        else:
            raise ValueError("split must be one of train, val, or test")
        return self.data.x[idx], self.data.z[idx]

    def treatment_embedding(self, treatment: np.ndarray | int) -> np.ndarray:
        treatment_arr = np.asarray(treatment, dtype=np.float32).reshape(-1)
        return ((treatment_arr + 0.5) / float(self.num_treatments)).astype(np.float32)

    def encode_w(self, x: np.ndarray, treatment: np.ndarray | int, dosage: np.ndarray | float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
        n = x_arr.shape[0]
        treatment_arr = np.asarray(treatment, dtype=np.int64).reshape(-1)
        dosage_arr = np.asarray(dosage, dtype=np.float32).reshape(-1)
        if treatment_arr.size == 1 and n > 1:
            treatment_arr = np.full(n, int(treatment_arr.item()), dtype=np.int64)
        if dosage_arr.size == 1 and n > 1:
            dosage_arr = np.full(n, float(dosage_arr.item()), dtype=np.float32)
        if treatment_arr.shape[0] != n or dosage_arr.shape[0] != n:
            raise ValueError("treatment and dosage must be scalar or align with x")
        return np.column_stack([x_arr, self.treatment_embedding(treatment_arr), dosage_arr]).astype(np.float32)

    def sample_target_w(self, n: int, seed: int | None = None, split: Literal["train", "val", "test"] = "train") -> np.ndarray:
        rng = make_rng(seed)
        x_pool, _ = self.split_arrays(split)
        draw = rng.integers(0, x_pool.shape[0], size=n)
        treatment = rng.integers(0, self.num_treatments, size=n)
        dosage = rng.choice(self.dosage_grid, size=n, replace=True)
        return self.encode_w(x_pool[draw], treatment, dosage)

    def treatment_propensity(self, z: np.ndarray) -> np.ndarray:
        z_arr = ensure_row_matrix(z)
        logits = self.gamma_a * (z_arr @ self.v.T)
        return _softmax_rows(logits).astype(np.float32)

    def optimal_dosage(self, z: np.ndarray, treatment: int) -> np.ndarray:
        z_arr = ensure_row_matrix(z)
        return _sigmoid(z_arr @ self.r[int(treatment)].reshape(-1, 1)).reshape(-1).astype(np.float32)

    def response_mean(self, z: np.ndarray, treatment: int, dosage: np.ndarray | float) -> np.ndarray:
        z_arr = ensure_row_matrix(z)
        dosage_arr = np.asarray(dosage, dtype=np.float32).reshape(-1)
        if dosage_arr.size == 1 and z_arr.shape[0] > 1:
            dosage_arr = np.full(z_arr.shape[0], float(dosage_arr.item()), dtype=np.float32)
        d_star = self.optimal_dosage(z_arr, treatment)
        value = (
            self.intercept[int(treatment)]
            + z_arr @ self.theta[int(treatment)]
            - self.lambda_a[int(treatment)] * (dosage_arr - d_star) ** 2
            + self.rho_a[int(treatment)] * np.sin(2.0 * np.pi * dosage_arr)
        )
        return value.astype(np.float32)

    def _mixture_params(
        self,
        z: np.ndarray,
        treatment: np.ndarray | int,
        dosage: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z_arr = ensure_row_matrix(z)
        n = z_arr.shape[0]
        treatment_arr = np.asarray(treatment, dtype=np.int64).reshape(-1)
        dosage_arr = np.asarray(dosage, dtype=np.float32).reshape(-1)
        if treatment_arr.size == 1 and n > 1:
            treatment_arr = np.full(n, int(treatment_arr.item()), dtype=np.int64)
        if dosage_arr.size == 1 and n > 1:
            dosage_arr = np.full(n, float(dosage_arr.item()), dtype=np.float32)
        eta = np.empty(n, dtype=np.float32)
        pi = np.empty(n, dtype=np.float32)
        delta = np.empty(n, dtype=np.float32)
        sigma1 = np.empty(n, dtype=np.float32)
        sigma2 = np.empty(n, dtype=np.float32)
        for treatment_value in range(self.num_treatments):
            mask = treatment_arr == treatment_value
            if not bool(np.any(mask)):
                continue
            z_local = z_arr[mask]
            d_local = dosage_arr[mask]
            eta[mask] = self.response_mean(z_local, treatment_value, d_local)
            pi[mask] = _sigmoid(z_local @ self.q[treatment_value] + 2.0 * d_local - 1.0)
            delta[mask] = 0.5 + 0.5 * np.tanh(z_local @ self.s[treatment_value] + d_local)
            sigma1[mask] = 0.1 + 0.3 * _sigmoid(z_local @ self.u1[treatment_value] + d_local)
            sigma2[mask] = 0.2 + 0.5 * _sigmoid(z_local @ self.u2[treatment_value] - d_local)
        return eta, pi, delta, sigma1, sigma2

    def sample_potential(
        self,
        z: np.ndarray,
        treatment: np.ndarray | int,
        dosage: np.ndarray | float,
        seed: int | None = None,
    ) -> np.ndarray:
        rng = make_rng(seed)
        eta, pi, delta, sigma1, sigma2 = self._mixture_params(z, treatment, dosage)
        first_component = rng.random(eta.shape[0]) < pi
        mean1 = eta + (1.0 - pi) * delta
        mean2 = eta - pi * delta
        y = np.empty_like(eta)
        y[first_component] = rng.normal(mean1[first_component], sigma1[first_component])
        y[~first_component] = rng.normal(mean2[~first_component], sigma2[~first_component])
        return y.reshape(-1, 1).astype(np.float32)

    def sample_observed(self, split: Literal["train", "val", "test"], seed: int | None = None) -> dict[str, np.ndarray]:
        rng = make_rng(seed)
        x, z = self.split_arrays(split)
        probs = self.treatment_propensity(z)
        uniforms = rng.random(x.shape[0])
        treatment = np.sum(uniforms[:, None] > np.cumsum(probs, axis=1), axis=1).astype(np.int64)
        d_star = np.array([self.optimal_dosage(z[i : i + 1], int(treatment[i]))[0] for i in range(x.shape[0])])
        alpha = 1.0 + self.alpha_d * d_star
        beta = 1.0 + self.alpha_d * (1.0 - d_star)
        dosage = rng.beta(alpha, beta).astype(np.float32)
        y = self.sample_potential(z, treatment, dosage, seed=int(rng.integers(0, 2**31 - 1)))
        return {
            "x": x.astype(np.float32),
            "z": z.astype(np.float32),
            "t": treatment,
            "d": dosage,
            "y": y,
            "w": self.encode_w(x, treatment, dosage),
        }
