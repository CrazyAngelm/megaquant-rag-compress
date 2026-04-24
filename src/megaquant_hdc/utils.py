from __future__ import annotations

import random
import time
from typing import Any, Callable, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is expected but keep helpers robust
    torch = None


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def ensure_2d_float32(x: Any) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(x, dtype=np.float32)
    squeezed = False
    if arr.ndim == 1:
        arr = arr[None, :]
        squeezed = True
    elif arr.ndim != 2:
        raise ValueError(f'Expected 1D or 2D array, got shape {arr.shape}')
    return arr, squeezed


def restore_from_2d(x: np.ndarray, squeezed: bool) -> np.ndarray:
    return x[0] if squeezed else x


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr, squeezed = ensure_2d_float32(x)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    out = arr / np.maximum(norms, eps)
    return restore_from_2d(out.astype(np.float32), squeezed)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    return float(np.mean((aa - bb) ** 2))


def mean_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    aa, _ = ensure_2d_float32(a)
    bb, _ = ensure_2d_float32(b)
    num = np.sum(aa * bb, axis=1)
    den = np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1)
    return float(np.mean(num / np.maximum(den, eps)))


def corrcoef_flat(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).ravel()
    bb = np.asarray(b, dtype=np.float32).ravel()
    if aa.size < 2:
        return 1.0
    if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
        return 1.0 if np.allclose(aa, bb) else 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def safe_softmax_1d(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float32)
    z = s - np.max(s)
    ez = np.exp(z)
    return ez / np.maximum(np.sum(ez), 1e-12)


def topk_contains(target_index: int, scores: np.ndarray, k: int = 5) -> float:
    s = np.asarray(scores, dtype=np.float32)
    if s.size == 0:
        return 0.0
    kk = min(k, s.size)
    topk = np.argpartition(-s, kth=np.arange(kk))[:kk]
    return float(target_index in topk.tolist())


def nominal_payload_bytes_per_vec(dim: int, bits: float) -> float:
    return float(dim * bits / 8.0)


def time_call(fn: Callable[[], Any], repeats: int = 1) -> tuple[Any, float]:
    result = None
    t0 = time.perf_counter()
    for _ in range(max(1, repeats)):
        result = fn()
    elapsed_ms = 1e3 * (time.perf_counter() - t0) / max(1, repeats)
    return result, float(elapsed_ms)
