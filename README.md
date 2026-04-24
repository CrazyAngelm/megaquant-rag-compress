# MegaQuant RAG Compress

MegaQuant RAG Compress is a **CPU/Python research proof-of-concept** for low-bit compression of stored RAG/document vectors.

It tests whether MegaQuant-style quantization can reduce stored vector payload size while preserving retrieval metrics in a small exact-search benchmark.

> Scope warning: this is not a production vector database, ANN, FAISS, Qdrant, Milvus, LanceDB, GPU, or large-scale embedding benchmark.

## Current headline, scoped carefully

In this CPU/Python exact-search benchmark, the best-performing configuration we tested is **document-only index compression**:

```text
doconly_affine3_g64_meta4
```

It compresses stored document vectors but keeps query vectors float32 at search time.

Why this appears to be the best tradeoff under this benchmark's assumptions:

- document vectors are the stored payload being compressed;
- query vectors are transient in this setup;
- in this benchmark, keeping queries float32 preserves retrieval quality better than quantizing both sides.

## Related repository

KV-cache companion project:

- https://github.com/CrazyAngelm/megaquant-kv-cache

## Benchmark setup

```text
Dataset: SQuAD v1.1 dev paragraphs/questions
Docs: 800
Queries: 4460
Embedding proxy: TF-IDF + GaussianRandomProjection -> 256d + L2 normalize
Search: exact dense matrix search on CPU/Python
```

This is a micro-scale proxy benchmark. TF-IDF + random projection is not a modern semantic embedding model such as BGE, E5, GTE, or OpenAI embeddings. Results may change on larger corpora, denser candidate sets, real embedding models, or ANN indexes.

## Headline result

Best method in this benchmark:

```text
doconly_affine3_g64_meta4
```

Result:

```text
effective_bits_per_dim = 3.126250
stored-vector memory saved = 90.230%
compression vs float32     = 10.236x
recall@1                   = 0.440009
recall@1 retention         = 98.69% of float32
MRR retention              = 98.99% of float32
score correlation          = 0.984499
```

Float32 baseline:

```text
recall@1 = 0.445840
MRR      = 0.542880
```

Plain-language summary for this benchmark:

> The stored vector payload is about **10x smaller**, while recall@1 retention remains about **98.7%** versus float32.

The memory number refers to compressed vector payload accounting, not total vector database footprint. For simulated low-bit metadata, it includes a small shared metadata-range overhead term. It does not include HNSW/IVF graph structures, IDs, metadata columns, allocator overhead, or packed-kernel layout overhead.

## Current frontier table

| Method | Effective bits/dim | Stored-vector memory saved | Recall@1 | Recall@1 retention | MRR retention | Notes |
|---|---:|---:|---:|---:|---:|---|
| `doconly_affine2_g64_meta4` | 2.126250 | 93.355% | 0.409509 | 91.85% | 93.50% | best ultra-compact point tested here |
| `doconly_affine3_g64_meta4` | 3.126250 | 90.230% | 0.440009 | 98.69% | 98.99% | best tradeoff tested here |
| `affine3_g64_meta4` | 3.126250 | 90.230% | 0.430366 | 96.53% | 97.47% | compress docs and queries |
| `nf3_g64_meta8` | 3.125625 | 90.232% | 0.428347 | 96.08% | 97.06% | nonuniform codebook variant |

## Recommended methods

### Main method

```text
doconly_affine3_g64_meta4
```

Use as the main PoC configuration when you want near-float32 metrics in this small exact-search benchmark with about 10.236x smaller modeled stored-vector payload.

### Ultra-compact method

```text
doconly_affine2_g64_meta4
```

Result:

```text
effective_bits_per_dim = 2.126250
stored-vector memory saved = 93.355%
recall@1 retention = 91.85%
MRR retention      = 93.50%
```

Use when stored-vector memory is more important than maximum benchmark recall.

## Reproduce

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Place SQuAD files in the repository root as described in `DATA.md`.

Run the current frontier benchmark from the repository root:

```bash
python scripts/run_frontier_benchmark.py \
  --docs 800 \
  --components 256 \
  --output-csv results/frontier_rag_benchmark.csv \
  --output-md reports/frontier_benchmark_report.md
```

## Reports

Current reports:

- `reports/frontier_summary.md`
- `reports/frontier_benchmark_report.md`

## Results

- `results/frontier_rag_benchmark.csv`

## Changelog

- `CHANGELOG.md`

## Related prior-work topics

A public-facing version should position this against standard retrieval compression and ANN topics: scalar quantization, product quantization (PQ/OPQ), residual quantization, binary quantization, HNSW/IVF/ScaNN, BEIR/MTEB evaluation, exact search versus ANN serving, and static index compression versus query-time compression.

## Honest limitations

This project currently demonstrates a **CPU/Python exact-search quality and modeled stored-vector-memory result**.

Not yet proven:

- production ANN/vector database speed,
- HNSW/IVF/PQ integration,
- GPU search,
- large embedding models such as OpenAI/text-embedding, BGE, E5, GTE,
- large-scale BEIR/MTEB retrieval quality,
- packed integer index implementation,
- total vector database memory savings including graph/ID/metadata overhead.

Conservative claim:

> In this small CPU/Python exact-search proxy benchmark, document-only `affine3_g64_meta4` compression gives the best observed stored-vector memory/quality tradeoff among the tested MegaQuant RAG configurations: about 90% modeled stored-vector memory saving while retaining about 98.7% of float32 recall@1.

---

## Repository positioning

This repository is public as a research PoC for compressed RAG/vector indexes. It is not a production vector database engine.

Suggested GitHub topics after public release:

```text
rag vector-search embeddings compression quantization retrieval ai-search
```
