from __future__ import annotations

import numpy as np

from .methods import _four_level_quantize_block, _robust_block_scale, _seven_level_quantize_block
from .utils import ensure_2d_float32, restore_from_2d


# Experimental low-overhead KV-cache quantizers.
# These functions still return dequantized float32 tensors for the existing
# benchmark path, but their companion metadata specs in registry.py describe
# honest storage accounting: code bits + scale/mean metadata.


def channelwise_four_level_2bit(x: np.ndarray, split: float = 0.75) -> np.ndarray:
    """2-bit signed four-level quantizer with one scale per channel.

    This is inspired by KIVI/KVQuant's observation that key-cache outliers are
    often channel-structured. It avoids per-token/per-block metadata: for an
    (tokens x dim) matrix it stores dim scales, not tokens*dim/block scales.
    """
    arr, squeezed = ensure_2d_float32(x)
    s = (0.6 * np.median(np.abs(arr), axis=0, keepdims=True) + 0.4 * np.mean(np.abs(arr), axis=0, keepdims=True)).astype(np.float32) + 1e-8
    out = _four_level_quantize_block(arr, s, split)
    return restore_from_2d(out.astype(np.float32), squeezed)


def tokenwise_four_level_2bit_g32(x: np.ndarray, group: int = 32, split: float = 0.75) -> np.ndarray:
    """2-bit signed four-level quantizer with larger per-token groups.

    Compared with the old block=4 path, group=32 cuts FP16 scale overhead from
    4 bits/dim to 0.5 bits/dim for non-centered methods.
    """
    arr, squeezed = ensure_2d_float32(x)
    n, d = arr.shape
    out = np.zeros_like(arr)
    for i in range(0, d, group):
        j = min(i + group, d)
        block_arr = arr[:, i:j]
        s = _robust_block_scale(block_arr)
        out[:, i:j] = _four_level_quantize_block(block_arr, s, split)
    return restore_from_2d(out.astype(np.float32), squeezed)


def tokenwise_seven_level_3bit_g32(x: np.ndarray, group: int = 32) -> np.ndarray:
    """3-bit seven-level quantizer with larger per-token groups."""
    arr, squeezed = ensure_2d_float32(x)
    n, d = arr.shape
    out = np.zeros_like(arr)
    for i in range(0, d, group):
        j = min(i + group, d)
        block_arr = arr[:, i:j]
        s = _robust_block_scale(block_arr)
        out[:, i:j] = _seven_level_quantize_block(block_arr, s)
    return restore_from_2d(out.astype(np.float32), squeezed)


def global_centered_four_level_2bit(x: np.ndarray, split: float = 0.75) -> np.ndarray:
    """2-bit four-level quantizer with one mean+scale per vector.

    This keeps the centering benefit but amortizes metadata across the full head
    dimension instead of every 4 elements.
    """
    arr, squeezed = ensure_2d_float32(x)
    mu = np.mean(arr, axis=1, keepdims=True).astype(np.float32)
    centered = arr - mu
    s = _robust_block_scale(centered)
    out = mu + _four_level_quantize_block(centered, s, split)
    return restore_from_2d(out.astype(np.float32), squeezed)


def global_centered_seven_level_3bit(x: np.ndarray) -> np.ndarray:
    """3-bit seven-level quantizer with one mean+scale per vector."""
    arr, squeezed = ensure_2d_float32(x)
    mu = np.mean(arr, axis=1, keepdims=True).astype(np.float32)
    centered = arr - mu
    s = _robust_block_scale(centered)
    out = mu + _seven_level_quantize_block(centered, s)
    return restore_from_2d(out.astype(np.float32), squeezed)


def channelwise_seven_level_3bit(x: np.ndarray) -> np.ndarray:
    """3-bit seven-level quantizer with one scale per channel across tokens."""
    arr, squeezed = ensure_2d_float32(x)
    s = (0.6 * np.median(np.abs(arr), axis=0, keepdims=True) + 0.4 * np.mean(np.abs(arr), axis=0, keepdims=True)).astype(np.float32) + 1e-8
    out = _seven_level_quantize_block(arr, s)
    return restore_from_2d(out.astype(np.float32), squeezed)


def _quantize_metadata_uniform(meta: np.ndarray, bits: int = 8) -> np.ndarray:
    """Simulate low-bit metadata storage by quantizing metadata then dequantizing."""
    arr = np.asarray(meta, dtype=np.float32)
    if arr.size == 0 or bits >= 16:
        return arr.astype(np.float32)
    mn = np.min(arr).astype(np.float32)
    mx = np.max(arr).astype(np.float32)
    qmax = float((1 << bits) - 1)
    s = (mx - mn) / qmax + 1e-12
    q = np.clip(np.round((arr - mn) / s), 0, qmax)
    return (q * s + mn).astype(np.float32)


def affine_four_level_2bit_g16(x: np.ndarray) -> np.ndarray:
    return affine_four_level_2bit_g32(x, group=16)


def affine_four_level_2bit_g64(x: np.ndarray) -> np.ndarray:
    return affine_four_level_2bit_g32(x, group=64)


def affine_four_level_2bit_g64_meta8(x: np.ndarray) -> np.ndarray:
    return affine_four_level_2bit_g32(x, group=64, meta_bits=8)


def affine_four_level_2bit_g64_meta4(x: np.ndarray) -> np.ndarray:
    return affine_four_level_2bit_g32(x, group=64, meta_bits=4)


def affine_seven_level_3bit_g64(x: np.ndarray) -> np.ndarray:
    return affine_seven_level_3bit_g32(x, group=64)


def affine_seven_level_3bit_g64_meta8(x: np.ndarray) -> np.ndarray:
    return affine_seven_level_3bit_g32(x, group=64, meta_bits=8)


def affine_seven_level_3bit_g64_meta4(x: np.ndarray) -> np.ndarray:
    return affine_seven_level_3bit_g32(x, group=64, meta_bits=4)


def tokenwise_four_level_2bit_g64(x: np.ndarray) -> np.ndarray:
    return tokenwise_four_level_2bit_g32(x, group=64)


def tokenwise_seven_level_3bit_g64(x: np.ndarray) -> np.ndarray:
    return tokenwise_seven_level_3bit_g32(x, group=64)


def affine_four_level_2bit_g32(x: np.ndarray, group: int = 32, meta_bits: int = 16) -> np.ndarray:
    """Uniform asymmetric 2-bit affine quantizer with group=32."""
    arr, squeezed = ensure_2d_float32(x)
    n, d = arr.shape
    out = np.zeros_like(arr)
    qmax = 3.0
    for i in range(0, d, group):
        j = min(i + group, d)
        block_arr = arr[:, i:j]
        mn = np.min(block_arr, axis=1, keepdims=True).astype(np.float32)
        mx = np.max(block_arr, axis=1, keepdims=True).astype(np.float32)
        s = (mx - mn) / qmax + 1e-8
        if meta_bits < 16:
            mn = _quantize_metadata_uniform(mn, bits=meta_bits)
            s = _quantize_metadata_uniform(s, bits=meta_bits)
        q = np.clip(np.round((block_arr - mn) / s), 0, qmax)
        out[:, i:j] = q * s + mn
    return restore_from_2d(out.astype(np.float32), squeezed)


def affine_seven_level_3bit_g32(x: np.ndarray, group: int = 32, meta_bits: int = 16) -> np.ndarray:
    """Uniform asymmetric 3-bit affine quantizer with group=32."""
    arr, squeezed = ensure_2d_float32(x)
    n, d = arr.shape
    out = np.zeros_like(arr)
    qmax = 7.0
    for i in range(0, d, group):
        j = min(i + group, d)
        block_arr = arr[:, i:j]
        mn = np.min(block_arr, axis=1, keepdims=True).astype(np.float32)
        mx = np.max(block_arr, axis=1, keepdims=True).astype(np.float32)
        s = (mx - mn) / qmax + 1e-8
        if meta_bits < 16:
            mn = _quantize_metadata_uniform(mn, bits=meta_bits)
            s = _quantize_metadata_uniform(s, bits=meta_bits)
        q = np.clip(np.round((block_arr - mn) / s), 0, qmax)
        out[:, i:j] = q * s + mn
    return restore_from_2d(out.astype(np.float32), squeezed)


def _with_sink1(x: np.ndarray, fn) -> np.ndarray:
    arr, squeezed = ensure_2d_float32(x)
    out = np.asarray(fn(arr), dtype=np.float32)
    if arr.shape[0] > 0:
        out[0] = arr[0]
    return restore_from_2d(out, squeezed)


def sink1_affine_four_level_2bit_g32(x: np.ndarray) -> np.ndarray:
    return _with_sink1(x, affine_four_level_2bit_g32)


def sink1_affine_four_level_2bit_g64_meta8(x: np.ndarray) -> np.ndarray:
    return _with_sink1(x, affine_four_level_2bit_g64_meta8)


def sink1_channelwise_four_level_2bit(x: np.ndarray) -> np.ndarray:
    return _with_sink1(x, channelwise_four_level_2bit)


def sink1_affine_seven_level_3bit_g64_meta8(x: np.ndarray) -> np.ndarray:
    return _with_sink1(x, affine_seven_level_3bit_g64_meta8)


def _sparse_topk_restore(x: np.ndarray, fn, k: int = 1) -> np.ndarray:
    """Quantize clipped dense part then restore top-k absolute entries per vector."""
    arr, squeezed = ensure_2d_float32(x)
    if arr.shape[1] == 0 or k <= 0:
        return fn(x)
    k = min(k, arr.shape[1])
    idx = np.argpartition(np.abs(arr), -k, axis=1)[:, -k:]
    dense = arr.copy()
    row = np.arange(arr.shape[0])[:, None]
    mask = np.ones(arr.shape, dtype=bool)
    mask[row, idx] = False
    fallback = np.median(np.abs(arr), axis=1, keepdims=True).astype(np.float32)
    clipped = np.where(mask, arr, np.sign(arr) * fallback)
    out = np.asarray(fn(clipped), dtype=np.float32)
    out[row, idx] = arr[row, idx]
    return restore_from_2d(out, squeezed)


def sparse1_affine_four_level_2bit_g64_meta8(x: np.ndarray) -> np.ndarray:
    return _sparse_topk_restore(x, affine_four_level_2bit_g64_meta8, k=1)


def sparse1_affine_four_level_2bit_g32(x: np.ndarray) -> np.ndarray:
    return _sparse_topk_restore(x, affine_four_level_2bit_g32, k=1)


def sparse1_channelwise_four_level_2bit(x: np.ndarray) -> np.ndarray:
    return _sparse_topk_restore(x, channelwise_four_level_2bit, k=1)


def sparse1_affine_seven_level_3bit_g64_meta8(x: np.ndarray) -> np.ndarray:
    return _sparse_topk_restore(x, affine_seven_level_3bit_g64_meta8, k=1)


def _fwht_rows(a: np.ndarray) -> np.ndarray:
    y = np.asarray(a, dtype=np.float32).copy()
    n = y.shape[1]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            u = y[:, i:i+h].copy()
            v = y[:, i+h:i+2*h].copy()
            y[:, i:i+h] = u + v
            y[:, i+h:i+2*h] = u - v
        h *= 2
    return y / np.sqrt(float(n))


def _hadamard_wrap(x: np.ndarray, fn) -> np.ndarray:
    arr, squeezed = ensure_2d_float32(x)
    d = arr.shape[1]
    if d == 0 or (d & (d - 1)) != 0:
        return fn(x)
    signs = np.where((np.arange(d) * 1103515245 + 12345) & 1, -1.0, 1.0).astype(np.float32)
    y = _fwht_rows(arr * signs)
    q = np.asarray(fn(y), dtype=np.float32)
    out = _fwht_rows(q) * signs
    return restore_from_2d(out.astype(np.float32), squeezed)


def hadamard_affine_four_level_2bit_g64_meta8(x: np.ndarray) -> np.ndarray:
    return _hadamard_wrap(x, affine_four_level_2bit_g64_meta8)


def hadamard_affine_four_level_2bit_g32(x: np.ndarray) -> np.ndarray:
    return _hadamard_wrap(x, affine_four_level_2bit_g32)


def hadamard_affine_seven_level_3bit_g64_meta8(x: np.ndarray) -> np.ndarray:
    return _hadamard_wrap(x, affine_seven_level_3bit_g64_meta8)


_NF2_LEVELS = np.asarray([-1.0, -0.246, 0.246, 1.0], dtype=np.float32)
_NF3_LEVELS = np.asarray([-1.0, -0.566, -0.282, -0.092, 0.092, 0.282, 0.566, 1.0], dtype=np.float32)


def _nf_quantize(x: np.ndarray, levels: np.ndarray, group: int = 64, meta_bits: int = 8) -> np.ndarray:
    arr, squeezed = ensure_2d_float32(x)
    out = np.zeros_like(arr)
    for i in range(0, arr.shape[1], group):
        j = min(i + group, arr.shape[1])
        block = arr[:, i:j]
        s = np.max(np.abs(block), axis=1, keepdims=True).astype(np.float32) + 1e-8
        if meta_bits < 16:
            s = _quantize_metadata_uniform(s, bits=meta_bits)
        y = np.clip(block / s, -1.0, 1.0)
        idx = np.argmin(np.abs(y[..., None] - levels.reshape(1, 1, -1)), axis=-1)
        out[:, i:j] = levels[idx] * s
    return restore_from_2d(out.astype(np.float32), squeezed)


def nf2_g64_meta8(x: np.ndarray) -> np.ndarray:
    return _nf_quantize(x, _NF2_LEVELS, group=64, meta_bits=8)


def nf3_g64_meta8(x: np.ndarray) -> np.ndarray:
    return _nf_quantize(x, _NF3_LEVELS, group=64, meta_bits=8)


EFFICIENT_FUNCTIONS = {
    'channelwise_four_level_2bit': channelwise_four_level_2bit,
    'tokenwise_four_level_2bit_g32': tokenwise_four_level_2bit_g32,
    'tokenwise_seven_level_3bit_g32': tokenwise_seven_level_3bit_g32,
    'global_centered_four_level_2bit': global_centered_four_level_2bit,
    'global_centered_seven_level_3bit': global_centered_seven_level_3bit,
    'channelwise_seven_level_3bit': channelwise_seven_level_3bit,
    'affine_four_level_2bit_g16': affine_four_level_2bit_g16,
    'affine_four_level_2bit_g32': affine_four_level_2bit_g32,
    'affine_four_level_2bit_g64': affine_four_level_2bit_g64,
    'affine_four_level_2bit_g64_meta8': affine_four_level_2bit_g64_meta8,
    'affine_four_level_2bit_g64_meta4': affine_four_level_2bit_g64_meta4,
    'affine_seven_level_3bit_g32': affine_seven_level_3bit_g32,
    'affine_seven_level_3bit_g64': affine_seven_level_3bit_g64,
    'affine_seven_level_3bit_g64_meta8': affine_seven_level_3bit_g64_meta8,
    'affine_seven_level_3bit_g64_meta4': affine_seven_level_3bit_g64_meta4,
    'tokenwise_four_level_2bit_g64': tokenwise_four_level_2bit_g64,
    'tokenwise_seven_level_3bit_g64': tokenwise_seven_level_3bit_g64,
    'sink1_affine_four_level_2bit_g32': sink1_affine_four_level_2bit_g32,
    'sink1_affine_four_level_2bit_g64_meta8': sink1_affine_four_level_2bit_g64_meta8,
    'sink1_channelwise_four_level_2bit': sink1_channelwise_four_level_2bit,
    'sink1_affine_seven_level_3bit_g64_meta8': sink1_affine_seven_level_3bit_g64_meta8,
    'sparse1_affine_four_level_2bit_g64_meta8': sparse1_affine_four_level_2bit_g64_meta8,
    'sparse1_affine_four_level_2bit_g32': sparse1_affine_four_level_2bit_g32,
    'sparse1_channelwise_four_level_2bit': sparse1_channelwise_four_level_2bit,
    'sparse1_affine_seven_level_3bit_g64_meta8': sparse1_affine_seven_level_3bit_g64_meta8,
    'hadamard_affine_four_level_2bit_g64_meta8': hadamard_affine_four_level_2bit_g64_meta8,
    'hadamard_affine_four_level_2bit_g32': hadamard_affine_four_level_2bit_g32,
    'hadamard_affine_seven_level_3bit_g64_meta8': hadamard_affine_seven_level_3bit_g64_meta8,
    'nf2_g64_meta8': nf2_g64_meta8,
    'nf3_g64_meta8': nf3_g64_meta8,
}
