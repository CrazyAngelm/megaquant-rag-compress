from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.random_projection import GaussianRandomProjection

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from megaquant_hdc.registry import get_method_spec
from megaquant_hdc.efficient_methods import (
    affine_four_level_2bit_g64_meta8,
    affine_seven_level_3bit_g64_meta4,
    affine_seven_level_3bit_g64_meta8,
    hadamard_affine_four_level_2bit_g64_meta8,
    nf3_g64_meta8,
)

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]")


def extract_squad(obj):
    docs = []
    qas = []
    for art in obj['data']:
        title = art.get('title', '')
        for p in art['paragraphs']:
            context = p['context'].strip().replace('\n', ' ')
            doc_id = len(docs)
            docs.append((title, context))
            for qa in p['qas']:
                if qa.get('is_impossible', False):
                    continue
                qas.append((qa['question'].strip(), doc_id))
    return docs, qas


def retrieval_metrics(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> dict[str, float]:
    pred = scores.argmax(axis=1)
    order = np.argsort(-scores, axis=1)
    topk = order[:, :k]
    rr = []
    ndcg = []
    for i in range(len(labels)):
        rank = int(np.where(order[i] == labels[i])[0][0]) + 1
        rr.append(1.0 / rank)
        ndcg.append(1.0 / math.log2(rank + 1) if rank <= k else 0.0)
    return {
        'recall@1': float(np.mean(pred == labels)),
        'recall@5': float(np.mean([labels[i] in topk[i, :5] for i in range(len(labels))])),
        'recall@10': float(np.mean([labels[i] in topk[i] for i in range(len(labels))])),
        'mrr': float(np.mean(rr)),
        'ndcg@10': float(np.mean(ndcg)),
    }


def corr_flat(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


@dataclass
class RagMethod:
    name: str
    family: str
    bits_per_dim: float
    effective_bits_per_dim: float
    fn: Callable[[np.ndarray], np.ndarray]
    encode_query: bool = True
    notes: str = ''

    def encode_docs(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.fn(x), dtype=np.float32)

    def encode_queries(self, x: np.ndarray) -> np.ndarray:
        if self.encode_query:
            return np.asarray(self.fn(x), dtype=np.float32)
        return x.astype(np.float32)


def _global_sign1(x: np.ndarray) -> np.ndarray:
    s = np.mean(np.abs(x)).astype(np.float32) + 1e-8
    return np.where(x >= 0, s, -s).astype(np.float32)


def _affine2_g64_meta4(x: np.ndarray) -> np.ndarray:
    from megaquant_hdc.efficient_methods import affine_four_level_2bit_g32
    return affine_four_level_2bit_g32(x, group=64, meta_bits=4)


def _centered_l2(fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    def wrapped(x: np.ndarray) -> np.ndarray:
        y = np.asarray(fn(x), dtype=np.float32)
        return normalize(y).astype(np.float32)
    return wrapped


def _docs_only(fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    return fn


def eff(name: str, dim: int, docs: int) -> float:
    return float(get_method_spec(name).effective_bits_per_dim(dim=dim, seq_len=docs))


def build_methods(dim: int, doc_count: int) -> list[RagMethod]:
    # Effective bits are for the compressed document index. Query encoding is compute-only and not stored.
    return [
        RagMethod('float32', 'baseline', 32.0, 32.0, lambda x: x.astype(np.float32), True, 'uncompressed'),
        RagMethod('sign1', 'baseline', 1.0, 1.0, _centered_l2(_global_sign1), True, 'global sign baseline'),
        RagMethod('affine2_g64_meta4', 'megaquant_rag_frontier', 2.0, eff('affine_four_level_2bit_g64_meta4', dim, doc_count), _centered_l2(_affine2_g64_meta4), True, '2-bit affine g64 meta4'),
        RagMethod('affine2_g64_meta8', 'megaquant_rag_frontier', 2.0, eff('affine_four_level_2bit_g64_meta8', dim, doc_count), _centered_l2(affine_four_level_2bit_g64_meta8), True, '2-bit affine g64 meta8'),
        RagMethod('hadamard_affine2_g64_meta8', 'megaquant_rag_frontier', 2.0, eff('hadamard_affine_four_level_2bit_g64_meta8', dim, doc_count), _centered_l2(hadamard_affine_four_level_2bit_g64_meta8), True, 'Hadamard + 2-bit affine g64 meta8'),
        RagMethod('affine3_g64_meta4', 'megaquant_rag_frontier', 3.0, eff('affine_seven_level_3bit_g64_meta4', dim, doc_count), _centered_l2(affine_seven_level_3bit_g64_meta4), True, '3-bit affine g64 meta4'),
        RagMethod('affine3_g64_meta8', 'megaquant_rag_frontier', 3.0, eff('affine_seven_level_3bit_g64_meta8', dim, doc_count), _centered_l2(affine_seven_level_3bit_g64_meta8), True, '3-bit affine g64 meta8'),
        RagMethod('nf3_g64_meta8', 'megaquant_rag_frontier', 3.0, eff('nf3_g64_meta8', dim, doc_count), _centered_l2(nf3_g64_meta8), True, 'NF3-like fixed nonuniform codebook'),
        RagMethod('doconly_affine2_g64_meta4', 'megaquant_rag_index_only', 2.0, eff('affine_four_level_2bit_g64_meta4', dim, doc_count), _centered_l2(_affine2_g64_meta4), False, 'compress docs only; keep query float32'),
        RagMethod('doconly_affine3_g64_meta4', 'megaquant_rag_index_only', 3.0, eff('affine_seven_level_3bit_g64_meta4', dim, doc_count), _centered_l2(affine_seven_level_3bit_g64_meta4), False, 'compress docs only; keep query float32'),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dev-path', default='squad_dev_v1.1.json')
    ap.add_argument('--docs', type=int, default=800)
    ap.add_argument('--components', type=int, default=256)
    ap.add_argument('--output-csv', default='results/frontier_rag_benchmark.csv')
    ap.add_argument('--output-md', default='reports/frontier_benchmark_report.md')
    args = ap.parse_args()

    dev = json.loads(Path(args.dev_path).read_text(encoding='utf-8'))
    dev_docs, dev_qas = extract_squad(dev)
    rag_doc_count = min(args.docs, len(dev_docs))
    doc_ids = list(range(rag_doc_count))
    doc_map = {old: i for i, old in enumerate(doc_ids)}
    docs_text = [dev_docs[i][1] for i in doc_ids]
    queries_text = []
    labels = []
    for q, doc_id in dev_qas:
        if doc_id in doc_map:
            queries_text.append(q)
            labels.append(doc_map[doc_id])
    labels = np.asarray(labels, dtype=np.int32)

    vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2, max_features=15000, sublinear_tf=True)
    x_docs_sparse = vec.fit_transform(docs_text)
    x_queries_sparse = vec.transform(queries_text)
    rp = GaussianRandomProjection(n_components=args.components, random_state=42)
    docs = normalize(rp.fit_transform(x_docs_sparse)).astype(np.float32)
    queries = normalize(rp.transform(x_queries_sparse)).astype(np.float32)
    float_scores = queries @ docs.T

    rows = []
    for method in build_methods(docs.shape[1], len(docs)):
        t0 = time.perf_counter()
        d_enc = method.encode_docs(docs)
        q_enc = method.encode_queries(queries)
        encode_us = 1e6 * (time.perf_counter() - t0) / (len(docs) + (len(queries) if method.encode_query else 0))
        search_t0 = time.perf_counter()
        scores = q_enc @ d_enc.T
        search_us_per_query = 1e6 * (time.perf_counter() - search_t0) / len(queries)
        m = retrieval_metrics(scores, labels, k=10)
        m.update({
            'method': method.name,
            'family': method.family,
            'bits_per_dim': method.bits_per_dim,
            'effective_bits_per_dim': method.effective_bits_per_dim,
            'compression_x_vs_float32': 32.0 / method.effective_bits_per_dim,
            'index_size_pct_float32': 100.0 * method.effective_bits_per_dim / 32.0,
            'index_memory_saved_pct': 100.0 * (1.0 - method.effective_bits_per_dim / 32.0),
            'payload_bytes_per_vec': docs.shape[1] * method.effective_bits_per_dim / 8.0,
            'score_corr_vs_float32': corr_flat(scores, float_scores),
            'encode_us_per_vec': float(encode_us),
            'search_us_per_query': float(search_us_per_query),
            'notes': method.notes,
        })
        rows.append(m)

    df = pd.DataFrame(rows).sort_values(['recall@1', 'mrr'], ascending=False)
    base = df[df.method == 'float32'].iloc[0]
    df['recall1_delta_vs_float32'] = df['recall@1'] - float(base['recall@1'])
    df['recall1_retention_pct'] = 100.0 * df['recall@1'] / float(base['recall@1'])
    df['mrr_retention_pct'] = 100.0 * df['mrr'] / float(base['mrr'])

    out_csv = Path(args.output_csv)
    out_md = Path(args.output_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    best_under = []
    for limit in [2.0, eff('affine_four_level_2bit_g64_meta4', docs.shape[1], len(docs)), eff('affine_four_level_2bit_g64_meta8', docs.shape[1], len(docs)), 3.0, eff('affine_seven_level_3bit_g64_meta4', docs.shape[1], len(docs)), eff('affine_seven_level_3bit_g64_meta8', docs.shape[1], len(docs)), 4.0]:
        sub = df[df.effective_bits_per_dim <= limit]
        if len(sub):
            r = sub.sort_values(['recall@1', 'mrr'], ascending=False).iloc[0]
            best_under.append({'budget': limit, 'method': r.method, 'effective_bits': r.effective_bits_per_dim, 'recall@1': r['recall@1'], 'mrr': r.mrr})

    report = []
    report.append('# MegaQuant RAG Frontier Benchmark')
    report.append('')
    report.append(f'- docs: {len(docs_text)}')
    report.append(f'- queries: {len(queries_text)}')
    report.append(f'- embedding pipeline: TF-IDF + GaussianRandomProjection -> {args.components}d + L2 normalize')
    report.append('- metric focus: recall@1, MRR, score correlation, index memory')
    report.append('')
    report.append('## Best under budgets')
    report.append(pd.DataFrame(best_under).to_markdown(index=False))
    report.append('')
    report.append('## Full results')
    show_cols = ['method','family','effective_bits_per_dim','compression_x_vs_float32','index_memory_saved_pct','recall@1','recall1_retention_pct','recall@5','recall@10','mrr','mrr_retention_pct','ndcg@10','score_corr_vs_float32','encode_us_per_vec','search_us_per_query']
    report.append(df[show_cols].to_markdown(index=False))
    report.append('')
    report.append('## Caveat')
    report.append('This benchmark compresses dense retrieval vectors in a CPU/Python exact-matrix-search setup. Effective bits are modeled stored-vector payload plus declared metadata and small shared metadata-range overhead. It is not a production ANN/vector-database throughput benchmark.')
    out_md.write_text('\n'.join(report), encoding='utf-8')

    print(df[['method','family','effective_bits_per_dim','recall@1','mrr','score_corr_vs_float32','index_memory_saved_pct','encode_us_per_vec','search_us_per_query']].to_string(index=False))
    print(f'Saved CSV: {out_csv}')
    print(f'Saved MD: {out_md}')


if __name__ == '__main__':
    main()
