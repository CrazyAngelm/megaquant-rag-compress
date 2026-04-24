from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .utils import corrcoef_flat, mean_cosine, mse, safe_softmax_1d, topk_contains


def reconstruction_metrics(x: np.ndarray, x_hat: np.ndarray) -> Dict[str, float]:
    return {
        'recon_mse': mse(x, x_hat),
        'recon_cos': mean_cosine(x, x_hat),
    }


def pairwise_inner_product_metrics(true_ip: np.ndarray, est_ip: np.ndarray) -> Dict[str, float]:
    true_ip = np.asarray(true_ip, dtype=np.float32)
    est_ip = np.asarray(est_ip, dtype=np.float32)
    diff = est_ip - true_ip
    return {
        'ip_bias': float(np.mean(diff)),
        'ip_rmse': float(np.sqrt(np.mean(diff ** 2))),
        'ip_corr': corrcoef_flat(true_ip, est_ip),
    }


def needle_metrics(scores: np.ndarray, needle_index: int) -> Dict[str, float]:
    s = np.asarray(scores, dtype=np.float32)
    return {
        'needle_top1': float(int(int(np.argmax(s)) == int(needle_index))),
        'needle_top5': topk_contains(int(needle_index), s, k=5),
    }


def attention_case_metrics(
    true_scores: np.ndarray,
    est_scores: np.ndarray,
    true_out: np.ndarray,
    est_out: np.ndarray,
    key_true: np.ndarray,
    key_hat: np.ndarray,
    value_true: np.ndarray,
    value_hat: np.ndarray,
) -> Dict[str, float]:
    top1 = float(int(int(np.argmax(est_scores)) == int(np.argmax(true_scores))))
    top5 = topk_contains(int(np.argmax(true_scores)), est_scores, k=5)
    return {
        'score_cos': mean_cosine(true_scores[None, :], est_scores[None, :]),
        'attn_out_cos': mean_cosine(true_out[None, :], est_out[None, :]),
        'top1_match': top1,
        'top5_hit': top5,
        'key_mse': mse(key_true, key_hat),
        'value_mse': mse(value_true, value_hat),
    }


def attention_output(scores: np.ndarray, values: np.ndarray) -> np.ndarray:
    weights = safe_softmax_1d(scores)
    return (weights[:, None] * values).sum(axis=0).astype(np.float32)
