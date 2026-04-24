# MegaQuant RAG Frontier Summary

- docs: 800
- queries: 4460
- embedding pipeline: TF-IDF + GaussianRandomProjection -> 256d + L2 normalize
- search: exact dense matrix multiplication on CPU/Python
- memory accounting: modeled stored-vector payload + declared metadata + small shared metadata-range parameters for simulated low-bit metadata

## Selected results

| method | effective_bits_per_dim | compression_x_vs_float32 | stored-vector memory saved | recall@1 | recall@1 retention | MRR retention | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `doconly_affine3_g64_meta4` | 3.126250 | 10.236 | 90.230% | 0.440009 | 98.69% | 98.99% | main current point |
| `doconly_affine2_g64_meta4` | 2.126250 | 15.050 | 93.355% | 0.409509 | 91.85% | 93.50% | ultra-compact point |
| `affine3_g64_meta4` | 3.126250 | 10.236 | 90.230% | 0.430366 | 96.53% | 97.47% | compress docs and queries |
| `nf3_g64_meta8` | 3.125625 | 10.238 | 90.232% | 0.428347 | 96.08% | 97.06% | fixed nonuniform codebook variant |

## Caveat

This is a small CPU/Python exact-search proxy benchmark using non-neural TF-IDF + random-projection embeddings. It is not BEIR/MTEB, not a production ANN/vector-database benchmark, and the memory figures are modeled stored-vector payload rather than total database footprint.
