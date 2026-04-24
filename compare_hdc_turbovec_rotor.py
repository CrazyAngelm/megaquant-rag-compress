import numpy as np, math, pandas as pd, time, runpy
from pathlib import Path

ns = runpy.run_path('/mnt/data/hdc_rotor_v2_compact.py')
fwht = ns['fwht']; random_signs=ns['random_signs']; hadamard_rotate=ns['hadamard_rotate']; uniform_dequantize=ns['uniform_dequantize']
five_level_dequantize=ns['five_level_dequantize']; random_orth_block=ns['random_orth_block']; make_block_rotation=ns['make_block_rotation']; apply_block_rotation=ns['apply_block_rotation']
blockwise_signedmag7_dequantize=ns['blockwise_signedmag7_dequantize']

def make_rag_dataset(d=512, n_classes=160, query_per_class=5, noise=0.24, seed=0):
    rng = np.random.default_rng(seed)
    docs = rng.normal(size=(n_classes, d)).astype(np.float32)
    docs /= np.linalg.norm(docs, axis=1, keepdims=True) + 1e-8
    queries=[]; labels=[]
    for c in range(n_classes):
        Q = docs[c] + noise * rng.normal(size=(query_per_class, d)).astype(np.float32)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8
        queries.append(Q); labels.extend([c]*query_per_class)
    return docs.astype(np.float32), np.concatenate(queries, axis=0).astype(np.float32), np.array(labels, dtype=np.int32)

def make_cache_dataset(d=128, n_keys=320, n_queries=160, value_dim=None, seed=0):
    if value_dim is None:
        value_dim = d
    rng = np.random.default_rng(seed)
    K = rng.normal(size=(n_keys, d)).astype(np.float32)
    V = rng.normal(size=(n_keys, value_dim)).astype(np.float32)
    Q = rng.normal(size=(n_queries, d)).astype(np.float32)
    basis = rng.normal(size=(8, d)).astype(np.float32)
    coeffK = rng.normal(size=(n_keys, 8)).astype(np.float32)
    coeffQ = rng.normal(size=(n_queries, 8)).astype(np.float32)
    K = 0.7*K + 0.6*(coeffK @ basis)
    Q = 0.7*Q + 0.6*(coeffQ @ basis)
    return K.astype(np.float32), V.astype(np.float32), Q.astype(np.float32)

def make_block_rotation_d3(d, rng):
    n_blocks = d // 3
    mats = np.stack([random_orth_block(3, rng) for _ in range(n_blocks)], axis=0)
    rem = d - 3*n_blocks
    return mats, rem

def apply_block_rotation_d3(X, mats, rem=0):
    if rem == 0:
        Xr = X.reshape(-1, mats.shape[0], 3)
        Y = np.einsum('mnb,nbc->mnc', Xr, mats, optimize=True)
        return Y.reshape(X.shape).astype(np.float32)
    main = X[:, :mats.shape[0]*3]
    tail = X[:, mats.shape[0]*3:]
    Xr = main.reshape(-1, mats.shape[0], 3)
    Y = np.einsum('mnb,nbc->mnc', Xr, mats, optimize=True).reshape(main.shape)
    return np.concatenate([Y, tail], axis=1).astype(np.float32)

def apply_block_rotation_d3_inv(X, mats, rem=0):
    inv = np.transpose(mats, (0,2,1)).copy()
    return apply_block_rotation_d3(X, inv, rem)

class Method:
    def __init__(self, name, bits, family, q_kind, transform_kind='none'):
        self.name=name; self.bits=bits; self.family=family; self.q_kind=q_kind; self.transform_kind=transform_kind
    def setup(self, d, seed=0):
        rng = np.random.default_rng(seed)
        if self.transform_kind=='hadamard':
            self.signs = random_signs(d, rng)
        elif self.transform_kind=='rotor3':
            self.block3, self.rem3 = make_block_rotation_d3(d, rng)
        return self
    def fwd(self, X):
        X = X.astype(np.float32)
        if self.transform_kind=='none':
            return X
        if self.transform_kind=='hadamard':
            return hadamard_rotate(X, self.signs)
        if self.transform_kind=='rotor3':
            return apply_block_rotation_d3(X, self.block3, self.rem3)
        raise ValueError(self.transform_kind)
    def inv(self, X):
        X = X.astype(np.float32)
        if self.transform_kind=='none':
            return X
        if self.transform_kind=='hadamard':
            return hadamard_rotate(X, self.signs)
        if self.transform_kind=='rotor3':
            return apply_block_rotation_d3_inv(X, self.block3, self.rem3)
        raise ValueError(self.transform_kind)
    def quant(self, X):
        X = X.astype(np.float32)
        if self.q_kind=='none':
            return X
        if self.q_kind=='uniform3':
            return uniform_dequantize(X, bits=3)
        if self.q_kind=='hdc5level3':
            return five_level_dequantize(X)
        if self.q_kind=='bw_signedmag7_3':
            return blockwise_signedmag7_dequantize(X, block=4)
        raise ValueError(self.q_kind)
    def encode_search(self, X):
        return self.quant(self.fwd(X))
    def encode_values(self, X):
        return self.inv(self.quant(self.fwd(X)))

def rag_metrics(docs, queries, labels, method):
    D = method.encode_search(docs)
    Q = method.encode_search(queries)
    S = Q @ D.T
    Sf = queries @ docs.T
    pred = S.argmax(axis=1)
    k=10
    top10 = np.argpartition(-S, kth=np.arange(k), axis=1)[:, :k]
    predf = Sf.argmax(axis=1)
    top10f = np.argpartition(-Sf, kth=np.arange(k), axis=1)[:, :k]
    return {
        'recall1': float(np.mean(pred == labels)),
        'recall10': float(np.mean([labels[i] in top10[i] for i in range(len(labels))])),
        'top1_agreement': float(np.mean(pred == predf)),
        'top10_overlap': float(np.mean([len(set(top10[i].tolist()) & set(top10f[i].tolist()))/k for i in range(len(labels))])),
        'score_corr': float(np.corrcoef(S.ravel(), Sf.ravel())[0,1]),
    }

def softmax(x):
    z = x - x.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / (ez.sum(axis=1, keepdims=True)+1e-12)

def cache_metrics(K, V, Q, method):
    Kq = method.encode_search(K)
    Qr = method.fwd(Q)
    Vq = method.encode_values(V)
    scale = 1.0 / math.sqrt(K.shape[1])
    Lf = (Q @ K.T) * scale
    Af = softmax(Lf)
    Of = Af @ V
    L = (Qr @ Kq.T) * scale
    A = softmax(L)
    O = A @ Vq
    top8 = np.argpartition(-L, kth=np.arange(8), axis=1)[:, :8]
    top8f = np.argpartition(-Lf, kth=np.arange(8), axis=1)[:, :8]
    return {
        'logits_corr': float(np.corrcoef(L.ravel(), Lf.ravel())[0,1]),
        'top1_agreement': float(np.mean(L.argmax(axis=1)==Lf.argmax(axis=1))),
        'top8_overlap': float(np.mean([len(set(top8[i].tolist()) & set(top8f[i].tolist()))/8 for i in range(L.shape[0])])),
        'attn_kl': float(np.mean(np.sum(Af*(np.log(Af+1e-12)-np.log(A+1e-12)), axis=1))),
        'out_cos': float(np.mean(np.sum(O*Of,axis=1)/(np.linalg.norm(O,axis=1)*np.linalg.norm(Of,axis=1)+1e-12))),
        'rel_mse': float(np.mean((O-Of)**2) / (np.mean(Of**2)+1e-12)),
    }

def encode_timing(X, method, repeats=10):
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = method.encode_search(X)
    return 1e6 * (time.perf_counter()-t0) / repeats / len(X)

methods = [
    Method('float32', 32, 'baseline', 'none', 'none'),
    Method('turbovec_orig_u3', 3, 'original', 'uniform3', 'hadamard'),
    Method('rotorquant_orig_u3_proxy', 3, 'original', 'uniform3', 'rotor3'),
    Method('hdc_only_5level3', 3, 'hdc_only', 'hdc5level3', 'none'),
    Method('hdc_only_bw7_3', 3, 'hdc_only', 'bw_signedmag7_3', 'none'),
    Method('turbovec_plus_hdc5', 3, 'hybrid', 'hdc5level3', 'hadamard'),
    Method('turbovec_plus_hdc_bw7', 3, 'hybrid', 'bw_signedmag7_3', 'hadamard'),
    Method('rotorquant_plus_hdc5_proxy', 3, 'hybrid', 'hdc5level3', 'rotor3'),
    Method('rotorquant_plus_hdc_bw7_proxy', 3, 'hybrid', 'bw_signedmag7_3', 'rotor3'),
]

rag_rows=[]; cache_rows=[]; timing_rows=[]
for seed in range(5):
    docs, queries, labels = make_rag_dataset(seed=100+seed)
    K,V,Q = make_cache_dataset(seed=200+seed)
    for m in methods:
        mr = Method(m.name, m.bits, m.family, m.q_kind, m.transform_kind).setup(d=docs.shape[1], seed=999)
        mc = Method(m.name, m.bits, m.family, m.q_kind, m.transform_kind).setup(d=K.shape[1], seed=999)
        rag_rows.append({'seed':seed,'method':m.name,'family':m.family,'bits':m.bits, **rag_metrics(docs,queries,labels,mr)})
        cache_rows.append({'seed':seed,'method':m.name,'family':m.family,'bits':m.bits, **cache_metrics(K,V,Q,mc)})
        if seed==0:
            timing_rows.append({'task':'rag_encode_docs','method':m.name,'family':m.family,'bits':m.bits,'us_per_vec':encode_timing(docs,mr,8)})
            timing_rows.append({'task':'cache_encode_keys','method':m.name,'family':m.family,'bits':m.bits,'us_per_vec':encode_timing(K,mc,12)})

rag = pd.DataFrame(rag_rows)
cache = pd.DataFrame(cache_rows)
timing = pd.DataFrame(timing_rows)
rag_sum = rag.groupby(['method','family','bits'], as_index=False).mean(numeric_only=True).sort_values('recall1', ascending=False)
cache_sum = cache.groupby(['method','family','bits'], as_index=False).mean(numeric_only=True).sort_values('out_cos', ascending=False)
timing_sum = timing.groupby(['task','method','family','bits'], as_index=False).mean(numeric_only=True)

# relative deltas vs originals

def add_deltas(df, metric, baseline_method):
    base = float(df.loc[df['method']==baseline_method, metric].iloc[0])
    out = df.copy()
    out[f'{metric}_delta_vs_{baseline_method}'] = out[metric] - base
    return out

rag_sum2 = add_deltas(rag_sum, 'recall1', 'turbovec_orig_u3')
cache_sum2 = add_deltas(cache_sum, 'out_cos', 'rotorquant_orig_u3_proxy')

# family bests
rag_best_family = rag_sum.sort_values('recall1', ascending=False).groupby('family', as_index=False).first()
cache_best_family = cache_sum.sort_values('out_cos', ascending=False).groupby('family', as_index=False).first()

# save
rag_sum2.to_csv('/mnt/data/hdc_combo_rag_summary.csv', index=False)
cache_sum2.to_csv('/mnt/data/hdc_combo_cache_summary.csv', index=False)
timing_sum.to_csv('/mnt/data/hdc_combo_timing.csv', index=False)
rag.to_csv('/mnt/data/hdc_combo_rag_raw.csv', index=False)
cache.to_csv('/mnt/data/hdc_combo_cache_raw.csv', index=False)

# markdown report
report = []
report.append('# Hybrid comparison: originals vs MegaQuant-only vs hybrids\n')
report.append('This is a synthetic CPU-side proxy benchmark. It is useful for relative comparison, not as a claim that the exact upstream kernels behave identically.\n')
report.append('Original-method proxies used here:\n')
report.append('- `turbovec_orig_u3`: global Hadamard-style TurboQuant/turbovec proxy (rotation + 3-bit scalar quantization)\n')
report.append('- `rotorquant_orig_u3_proxy`: 3D block-rotation RotorQuant proxy (local rotation + 3-bit scalar quantization)\n')
report.append('MegaQuant-only methods:\n')
report.append('- `hdc_only_5level3`: pure sign-preserving 5-level quantizer\n')
report.append('- `hdc_only_bw7_3`: blockwise signed-magnitude 7-level quantizer\n')
report.append('Hybrids:\n')
report.append('- original rotation + MegaQuant quantizer\n')
report.append('\n## RAG-like retrieval summary\n')
report.append(rag_sum2[['method','family','bits','recall1','recall10','top1_agreement','top10_overlap','score_corr','recall1_delta_vs_turbovec_orig_u3']].to_markdown(index=False))
report.append('\n\n## KV-cache-like summary\n')
report.append(cache_sum2[['method','family','bits','out_cos','logits_corr','top1_agreement','top8_overlap','rel_mse','out_cos_delta_vs_rotorquant_orig_u3_proxy']].to_markdown(index=False))
report.append('\n\n## Encode timing (lower is better, CPU proxy)\n')
report.append(timing_sum.to_markdown(index=False))
report.append('\n\n## Best by family\n')
report.append('### RAG\n')
report.append(rag_best_family[['family','method','recall1','recall10','score_corr']].to_markdown(index=False))
report.append('\n\n### KV-cache\n')
report.append(cache_best_family[['family','method','out_cos','logits_corr','rel_mse']].to_markdown(index=False))

Path('/mnt/data/hdc_combo_report.md').write_text('\n'.join(report), encoding='utf-8')
print(rag_sum2[['method','family','recall1','recall1_delta_vs_turbovec_orig_u3']].to_string(index=False))
print('\nCACHE')
print(cache_sum2[['method','family','out_cos','out_cos_delta_vs_rotorquant_orig_u3_proxy','rel_mse']].to_string(index=False))
print('\nSaved files.')
