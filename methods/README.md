# RAG methods

This folder documents the current MegaQuant RAG/vector-search methods.

## Selected methods

### `doconly_affine3_g64_meta4`

Main RAG method in the current benchmark.

- Compresses stored document vectors only.
- Keeps query vectors float32 at search time.
- Uses 3-bit affine quantization, group size 64, simulated int4 scale/zero metadata.

Result:

```text
effective_bits_per_dim = 3.126250
stored-vector memory saved = 90.230%
recall@1 retention = 98.69%
MRR retention      = 98.99%
```

### `doconly_affine2_g64_meta4`

Ultra-compact RAG method.

```text
effective_bits_per_dim = 2.126250
stored-vector memory saved = 93.355%
recall@1 retention = 91.85%
MRR retention      = 93.50%
```

## Design choice

The current best setup compresses the stored document vectors and leaves query vectors float32. In this benchmark, that gives better retrieval metrics than quantizing both sides while still reducing stored-vector payload.

## Implementation

```text
scripts/run_frontier_benchmark.py
src/megaquant_hdc/efficient_methods.py
```

## Accounting note

Effective bits include modeled code bits, declared metadata bits, and a small shared metadata-range overhead term for simulated low-bit metadata. They are not measured packed vector-database memory.
