from __future__ import annotations

import numpy as np

from .utils import ensure_2d_float32, restore_from_2d


# Reference implementations of the MegaQuant quantizers.
# These are intentionally readable and deterministic, not kernel-optimized.
#
# Canonical public names:
# - blockwise_ternary_2bit
# - five_level_3bit
# - blockwise_seven_level_3bit
# - blockwise_four_level_2bit
# - mixed_precision_avg_2bit


def _robust_block_scale(block: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    med = np.median(np.abs(block), axis=1, keepdims=True).astype(np.float32)
    mean = np.mean(np.abs(block), axis=1, keepdims=True).astype(np.float32)
    scale = 0.6 * med + 0.4 * mean
    return scale + eps


def sign1_dequantize(x: np.ndarray) -> np.ndarray:
    arr, squeezed = ensure_2d_float32(x)
    s = np.mean(np.abs(arr)).astype(np.float32) + 1e-8
    out = np.where(arr >= 0, s, -s).astype(np.float32)
    return restore_from_2d(out, squeezed)


def centered_sign_1bit(x: np.ndarray, block: int = 4) -> np.ndarray:
    """Centered blockwise 1-bit quantizer using per-block mean plus signed residual magnitude."""
    arr, squeezed = ensure_2d_float32(x)
    n, d = arr.shape
    out = np.zeros_like(arr)
    for i in range(0, d, block):
        j = min(i + block, d)
        block_arr = arr[:, i:j]
        mu = np.mean(block_arr, axis=1, keepdims=True).astype(np.float32)
        centered = block_arr - mu
        s = _robust_block_scale(centered)
        out[:, i:j] = mu + np.where(centered >= 0, s, -s)
    return restore_from_2d(out.astype(np.float32), squeezed)


def blockwise_ternary_2bit(x: np.ndarray, block: int = 4) -> np.ndarray:
    """Early 2-bit baseline with blockwise ternary-style reconstruction."""
    arr, squeezed = ensure_2d_float32(x)
    n, d = arr.shape
    out = np.zeros_like(arr)
    for i in range(0, d, block):
        j = min(i + block, d)
        block_arr = arr[:, i:j]
        s = _robust_block_scale(block_arr)
        thr = 0.75 * s
        out[:, i:j] = np.where(np.abs(block_arr) < thr, 0.0, np.where(block_arr >= 0, s, -s))
    return restore_from_2d(out.astype(np.float32), squeezed)


def five_level_3bit(x: np.ndarray) -> np.ndarray:
    """Simple 5-level sign-preserving 3-bit quantizer."""
    arr, squeezed = ensure_2d_float32(x)
    s = (0.5 * np.median(np.abs(arr)) + 0.5 * np.mean(np.abs(arr))).astype(np.float32) + 1e-8
    a = 0.65 * s
    b = 1.75 * s
    abs_arr = np.abs(arr)
    out = np.zeros_like(arr)
    out = np.where((abs_arr >= 0.35 * s) & (abs_arr < 1.2 * s), np.where(arr >= 0, a, -a), out)
    out = np.where(abs_arr >= 1.2 * s, np.where(arr >= 0, b, -b), out)
    return restore_from_2d(out.astype(np.float32), squeezed)


def _seven_level_quantize_block(block_arr: np.ndarray, s: np.ndarray) -> np.ndarray:
    c1 = 0.45 * s
    c2 = 1.05 * s
    c3 = 2.10 * s
    abs_arr = np.abs(block_arr)
    y = np.zeros_like(block_arr)
    y = np.where((abs_arr >= 0.20 * s) & (abs_arr < 0.75 * s), np.where(block_arr >= 0, c1, -c1), y)
    y = np.where((abs_arr >= 0.75 * s) & (abs_arr < 1.55 * s), np.where(block_arr >= 0, c2, -c2), y)
    y = np.where(abs_arr >= 1.55 * s, np.where(block_arr >= 0, c3, -c3), y)
    return y


def blockwise_seven_level_3bit(x: np.ndarray, block: int = 4) -> np.ndarray:
    """Main 3-bit blockwise signed-magnitude 7-level quantizer."""
    arr, squeezed = ensure_2d_float32(x)
    n, d = arr.shape
    out = np.zeros_like(arr)
    for i in range(0, d, block):
        j = min(i + block, d)
        block_arr = arr[:, i:j]
        s = _robust_block_scale(block_arr)
        out[:, i:j] = _seven_level_quantize_block(block_arr, s)
    return restore_from_2d(out.astype(np.float32), squeezed)


def centered_seven_level_3bit(x: np.ndarray, block: int = 4) -> np.ndarray:
    """Centered 3-bit seven-level quantizer using per-block mean plus quantized residuals."""
    arr, squeezed = ensure_2d_float32(x)
    n, d = arr.shape
    out = np.zeros_like(arr)
    for i in range(0, d, block):
        j = min(i + block, d)
        block_arr = arr[:, i:j]
        mu = np.mean(block_arr, axis=1, keepdims=True).astype(np.float32)
        centered = block_arr - mu
        s = _robust_block_scale(centered)
        out[:, i:j] = mu + _seven_level_quantize_block(centered, s)
    return restore_from_2d(out.astype(np.float32), squeezed)


def _four_level_quantize_block(block_arr: np.ndarray, s: np.ndarray, split: float) -> np.ndarray:
    a = 0.55 * s
    b = 1.85 * s
    thr = split * s
    return np.where(np.abs(block_arr) < thr, np.where(block_arr >= 0, a, -a), np.where(block_arr >= 0, b, -b))


def blockwise_four_level_2bit(x: np.ndarray, block: int = 4, split: float = 0.75) -> np.ndarray:
    """Strict 2-bit descendant of the 7-level design with 4 signed non-uniform levels."""
    arr, squeezed = ensure_2d_float32(x)
    n, d = arr.shape
    out = np.zeros_like(arr)
    for i in range(0, d, block):
        j = min(i + block, d)
        block_arr = arr[:, i:j]
        s = _robust_block_scale(block_arr)
        out[:, i:j] = _four_level_quantize_block(block_arr, s, split)
    return restore_from_2d(out.astype(np.float32), squeezed)


def centered_four_level_2bit(x: np.ndarray, block: int = 4, split: float = 0.75) -> np.ndarray:
    """Centered 2-bit four-level quantizer using per-block mean plus quantized residuals."""
    arr, squeezed = ensure_2d_float32(x)
    n, d = arr.shape
    out = np.zeros_like(arr)
    for i in range(0, d, block):
        j = min(i + block, d)
        block_arr = arr[:, i:j]
        mu = np.mean(block_arr, axis=1, keepdims=True).astype(np.float32)
        centered = block_arr - mu
        s = _robust_block_scale(centered)
        out[:, i:j] = mu + _four_level_quantize_block(centered, s, split)
    return restore_from_2d(out.astype(np.float32), squeezed)


def mixed_precision_avg_2bit(x: np.ndarray, block: int = 4, qsplit: float = 0.75) -> np.ndarray:
    """Average-2-bit mixed precision variant using a fixed block pattern."""
    arr, squeezed = ensure_2d_float32(x)
    n, d = arr.shape
    out = np.zeros_like(arr)
    blocks = [(i, min(i + block, d)) for i in range(0, d, block)]
    pattern = ['bw7', 'bw4', 'bw4', 'sign1']
    for idx, (i, j) in enumerate(blocks):
        block_arr = arr[:, i:j]
        mode = pattern[idx % len(pattern)]
        if mode == 'bw7':
            out[:, i:j] = blockwise_seven_level_3bit(block_arr, block=(j - i))
        elif mode == 'bw4':
            out[:, i:j] = blockwise_four_level_2bit(block_arr, block=(j - i), split=qsplit)
        else:
            out[:, i:j] = sign1_dequantize(block_arr)
    return restore_from_2d(out.astype(np.float32), squeezed)


def mixed_precision_avg_2bit_centered_sign(x: np.ndarray, block: int = 4, qsplit: float = 0.75) -> np.ndarray:
    """Average-2-bit mixed precision variant that replaces the 1-bit branch with centered sign quantization."""
    arr, squeezed = ensure_2d_float32(x)
    n, d = arr.shape
    out = np.zeros_like(arr)
    blocks = [(i, min(i + block, d)) for i in range(0, d, block)]
    pattern = ['bw7', 'bw4', 'bw4', 'csign1']
    for idx, (i, j) in enumerate(blocks):
        block_arr = arr[:, i:j]
        mode = pattern[idx % len(pattern)]
        if mode == 'bw7':
            out[:, i:j] = blockwise_seven_level_3bit(block_arr, block=(j - i))
        elif mode == 'bw4':
            out[:, i:j] = blockwise_four_level_2bit(block_arr, block=(j - i), split=qsplit)
        else:
            out[:, i:j] = centered_sign_1bit(block_arr, block=(j - i))
    return restore_from_2d(out.astype(np.float32), squeezed)


HDC_FUNCTIONS = {
    'blockwise_ternary_2bit': blockwise_ternary_2bit,
    'centered_sign_1bit': centered_sign_1bit,
    'five_level_3bit': five_level_3bit,
    'blockwise_seven_level_3bit': blockwise_seven_level_3bit,
    'centered_seven_level_3bit': centered_seven_level_3bit,
    'blockwise_four_level_2bit': blockwise_four_level_2bit,
    'centered_four_level_2bit': centered_four_level_2bit,
    'mixed_precision_avg_2bit': mixed_precision_avg_2bit,
    'mixed_precision_avg_2bit_centered_sign': mixed_precision_avg_2bit_centered_sign,
}
