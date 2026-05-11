from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray


def ensure_2d(array: NDArray) -> NDArray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        return array[:, None]
    return array


def ensure_row_matrix(array: NDArray) -> NDArray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        return array[None, :]
    return array


def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def dyadic_num_cells(level: int, dim: int) -> int:
    return 2 ** (level * dim)


def dyadic_cell_indices(w: NDArray, level: int) -> NDArray:
    w = np.asarray(w, dtype=np.float32)
    if w.ndim == 1:
        w = w[None, :]
    if level < 0:
        raise ValueError("level must be nonnegative")
    side = 2**level
    clipped = np.minimum(np.maximum(w, 0.0), np.nextafter(1.0, 0.0))
    grid = np.floor(clipped * side).astype(np.int64)
    multipliers = (side ** np.arange(w.shape[1], dtype=np.int64)).reshape(1, -1)
    return (grid * multipliers).sum(axis=1)


def dyadic_cell_midpoints(level: int, dim: int) -> NDArray:
    side = 2**level
    coords = np.stack(np.unravel_index(np.arange(side**dim), (side,) * dim), axis=1)
    return (coords.astype(np.float32) + 0.5) / side


def chunked(iterable: List[int], chunk_size: int) -> List[List[int]]:
    return [iterable[i: i + chunk_size] for i in range(0, len(iterable), chunk_size)]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
