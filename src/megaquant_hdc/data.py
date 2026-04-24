from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .utils import l2_normalize_rows


def load_json(path: str | Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_squad_contexts(obj) -> list[str]:
    contexts: list[str] = []
    for article in obj['data']:
        for paragraph in article['paragraphs']:
            ctx = paragraph['context'].strip().replace('\n', ' ')
            contexts.append(ctx)
    return contexts


def _resolve_existing_path(candidates: list[str | Path]) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f'Could not find any of the expected dataset paths: {candidates}')


def build_squad_text_corpus(
    train_path: str | Path = 'squad_train_v1.1.json',
    dev_path: str | Path = 'squad_dev_v1.1.json',
    split: str = 'both',
    max_contexts: int = 32,
) -> str:
    root = Path(__file__).resolve().parents[2]
    train_path = _resolve_existing_path([train_path, root / 'squad_train_v1.1.json'])
    dev_path = _resolve_existing_path([dev_path, root / 'squad_dev_v1.1.json'])

    contexts: list[str] = []
    if split in ('train', 'both'):
        contexts.extend(extract_squad_contexts(load_json(train_path)))
    if split in ('dev', 'both'):
        contexts.extend(extract_squad_contexts(load_json(dev_path)))
    contexts = contexts[:max_contexts]
    return '\n\n'.join(contexts)


def make_unit_vectors(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    return l2_normalize_rows(x)


def make_structured_vectors(n: int, d: int, rank: int, seed: int, noise: float = 0.15) -> np.ndarray:
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(rank, d)).astype(np.float32)
    basis = l2_normalize_rows(basis)
    coeffs = rng.normal(size=(n, rank)).astype(np.float32)
    x = coeffs @ basis
    x += noise * rng.normal(size=(n, d)).astype(np.float32)
    return l2_normalize_rows(x)


def make_needle_dataset(seq_len: int, d: int, seed: int, structured: bool = False, rank: int = 16):
    if structured:
        keys = make_structured_vectors(seq_len, d, rank=rank, seed=seed)
    else:
        keys = make_unit_vectors(seq_len, d, seed=seed)
    needle_idx = seq_len // 3
    query = keys[needle_idx].copy()
    return keys.astype(np.float32), query.astype(np.float32), needle_idx
