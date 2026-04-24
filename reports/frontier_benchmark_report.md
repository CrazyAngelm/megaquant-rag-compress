# MegaQuant RAG Frontier Benchmark

- docs: 800
- queries: 4459
- embedding pipeline: TF-IDF + GaussianRandomProjection -> 256d + L2 normalize
- metric focus: recall@1, MRR, score correlation, index memory

## Best under budgets
|   budget | method                    |   effective_bits |   recall@1 |      mrr |
|---------:|:--------------------------|-----------------:|-----------:|---------:|
|  2       | sign1                     |          1       |   0.263063 | 0.356814 |
|  2.12625 | doconly_affine2_g64_meta4 |          2.12625 |   0.409509 | 0.507607 |
|  2.25125 | doconly_affine2_g64_meta4 |          2.12625 |   0.409509 | 0.507607 |
|  3       | doconly_affine2_g64_meta4 |          2.12625 |   0.409509 | 0.507607 |
|  3.12625 | doconly_affine3_g64_meta4 |          3.12625 |   0.440009 | 0.537392 |
|  3.25125 | doconly_affine3_g64_meta4 |          3.12625 |   0.440009 | 0.537392 |
|  4       | doconly_affine3_g64_meta4 |          3.12625 |   0.440009 | 0.537392 |

## Full results
| method                     | family                   |   effective_bits_per_dim |   compression_x_vs_float32 |   index_memory_saved_pct |   recall@1 |   recall1_retention_pct |   recall@5 |   recall@10 |      mrr |   mrr_retention_pct |   ndcg@10 |   score_corr_vs_float32 |   encode_us_per_vec |   search_us_per_query |
|:---------------------------|:-------------------------|-------------------------:|---------------------------:|-------------------------:|-----------:|------------------------:|-----------:|------------:|---------:|--------------------:|----------:|------------------------:|--------------------:|----------------------:|
| float32                    | baseline                 |                 32       |                    1       |                   0      |   0.44584  |                100      |   0.652613 |    0.716528 | 0.54288  |            100      |  0.578868 |                1        |            0.393402 |               2.15589 |
| doconly_affine3_g64_meta4  | megaquant_rag_index_only |                  3.12625 |                   10.2359  |                  90.2305 |   0.440009 |                 98.6922 |   0.64723  |    0.711819 | 0.537392 |             98.9891 |  0.573536 |                0.984499 |            9.62862  |               2.81969 |
| affine3_g64_meta8          | megaquant_rag_frontier   |                  3.25125 |                    9.84237 |                  89.8398 |   0.431038 |                 96.6801 |   0.644091 |    0.706661 | 0.529234 |             97.4862 |  0.566127 |                0.969272 |            8.76874  |               2.77704 |
| affine3_g64_meta4          | megaquant_rag_frontier   |                  3.12625 |                   10.2359  |                  90.2305 |   0.430366 |                 96.5292 |   0.639381 |    0.707333 | 0.529156 |             97.472  |  0.566262 |                0.968962 |            9.01308  |               2.69224 |
| nf3_g64_meta8              | megaquant_rag_frontier   |                  3.12562 |                   10.238   |                  90.2324 |   0.428347 |                 96.0765 |   0.636466 |    0.704867 | 0.526899 |             97.0562 |  0.563802 |                0.968027 |           31.4783   |               2.52144 |
| doconly_affine2_g64_meta4  | megaquant_rag_index_only |                  2.12625 |                   15.05    |                  93.3555 |   0.409509 |                 91.8511 |   0.620318 |    0.693653 | 0.507607 |             93.5026 |  0.546208 |                0.923834 |            8.84988  |               2.44925 |
| affine2_g64_meta8          | megaquant_rag_frontier   |                  2.25125 |                   14.2143  |                  92.9648 |   0.37856  |                 84.9095 |   0.585109 |    0.654407 | 0.475676 |             87.6208 |  0.511906 |                0.852697 |           10.99     |               2.92139 |
| affine2_g64_meta4          | megaquant_rag_frontier   |                  2.12625 |                   15.05    |                  93.3555 |   0.378112 |                 84.8089 |   0.583763 |    0.655977 | 0.475461 |             87.5811 |  0.512041 |                0.852354 |            8.76492  |               2.7109  |
| hadamard_affine2_g64_meta8 | megaquant_rag_frontier   |                  2.25125 |                   14.2143  |                  92.9648 |   0.374299 |                 83.9537 |   0.583315 |    0.650819 | 0.472742 |             87.0802 |  0.508846 |                0.853026 |           38.3171   |               2.57695 |
| sign1                      | baseline                 |                  1       |                   32       |                  96.875  |   0.263063 |                 59.004  |   0.453241 |    0.52523  | 0.356814 |             65.7261 |  0.388541 |                0.665593 |            4.89589  |               2.8638  |

## Caveat
This benchmark compresses dense retrieval vectors in a CPU/Python exact-matrix-search setup. Effective bits are modeled stored-vector payload plus declared metadata and small shared metadata-range overhead. It is not a production ANN/vector-database throughput benchmark.