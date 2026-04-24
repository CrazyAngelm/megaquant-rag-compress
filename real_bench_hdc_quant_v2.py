import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import json, re, math, time, random, runpy
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(1)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def extract_squad(obj):
    docs=[]
    qas=[]
    for art in obj['data']:
        title = art.get('title','')
        for p in art['paragraphs']:
            context = p['context'].strip().replace('\n', ' ')
            doc_id = len(docs)
            docs.append((title, context))
            for qa in p['qas']:
                if qa.get('is_impossible', False):
                    continue
                qas.append((qa['question'].strip(), doc_id, [a['text'] for a in qa.get('answers', [])]))
    return docs, qas

# load helper fns from earlier file
ns = runpy.run_path('/mnt/data/hdc_rotor_v2_compact.py')
random_signs=ns['random_signs']; hadamard_rotate=ns['hadamard_rotate']; uniform_dequantize=ns['uniform_dequantize']
five_level_dequantize=ns['five_level_dequantize']; random_orth_block=ns['random_orth_block']
blockwise_ternary_dequantize=ns['blockwise_ternary_dequantize']; blockwise_signedmag7_dequantize=ns['blockwise_signedmag7_dequantize']

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

def sign1_dequantize(X):
    s = np.mean(np.abs(X)).astype(np.float32) + 1e-8
    return np.where(X>=0, s, -s).astype(np.float32)

class Method:
    def __init__(self, name, bits, family, q_kind, transform_kind='none'):
        self.name=name; self.bits=bits; self.family=family; self.q_kind=q_kind; self.transform_kind=transform_kind
    def setup(self, d, seed=0):
        rng = np.random.default_rng(seed)
        if self.transform_kind=='hadamard':
            self.signs = random_signs(d, rng)
        elif self.transform_kind=='rotor3':
            self.block3, self.rem3 = make_block_rotation_d3(d, rng)
        self.d = d
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
        if self.q_kind=='sign1':
            return sign1_dequantize(X)
        if self.q_kind=='uniform2':
            return uniform_dequantize(X, bits=2)
        if self.q_kind=='uniform3':
            return uniform_dequantize(X, bits=3)
        if self.q_kind=='blockwise_ternary_2bit':
            return blockwise_ternary_dequantize(X, block=4)
        if self.q_kind=='hdc5level3':
            return five_level_dequantize(X)
        if self.q_kind=='bw_signedmag7_3':
            return blockwise_signedmag7_dequantize(X, block=4)
        raise ValueError(self.q_kind)
    def encode_search(self, X):
        return self.quant(self.fwd(X))
    def encode_values(self, X):
        return self.inv(self.quant(self.fwd(X)))

METHODS = [
    Method('float32',32,'baseline','none','none'),
    Method('sign1',1,'baseline','sign1','none'),
    Method('turbovec_u2',2,'original','uniform2','hadamard'),
    Method('turbovec_u3',3,'original','uniform3','hadamard'),
    Method('rotorquant_u3_proxy',3,'original','uniform3','rotor3'),
    Method('blockwise_ternary_2bit',2,'megaquant_only','blockwise_ternary_2bit','none'),
    Method('five_level_3bit',3,'megaquant_only','hdc5level3','none'),
    Method('blockwise_seven_level_3bit',3,'megaquant_only','bw_signedmag7_3','none'),
    Method('turbovec_plus_hdc5',3,'hybrid','hdc5level3','hadamard'),
    Method('turbovec_plus_bw7',3,'hybrid','bw_signedmag7_3','hadamard'),
    Method('rotorquant_plus_hdc5_proxy',3,'hybrid','hdc5level3','rotor3'),
    Method('rotorquant_plus_bw7_proxy',3,'hybrid','bw_signedmag7_3','rotor3'),
]

def retrieval_metrics(S, labels, k=10):
    n = len(labels)
    pred = S.argmax(axis=1)
    order = np.argsort(-S, axis=1)
    topk = order[:, :k]
    rr = []
    ndcg = []
    for i in range(n):
        rank = int(np.where(order[i] == labels[i])[0][0]) + 1
        rr.append(1.0 / rank)
        ndcg.append(1.0 / math.log2(rank + 1) if rank <= k else 0.0)
    return {
        'recall@1': float(np.mean(pred == labels)),
        'recall@5': float(np.mean([labels[i] in topk[i, :5] for i in range(n)])),
        'recall@10': float(np.mean([labels[i] in topk[i] for i in range(n)])),
        'mrr': float(np.mean(rr)),
        'ndcg@10': float(np.mean(ndcg)),
    }

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]")
def tokenize(text):
    return TOKEN_RE.findall(text.lower())

class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=96, max_len=32):
        super().__init__()
        self.vocab_size=vocab_size
        self.d_model=d_model
        self.n_heads=n_heads
        self.head_dim=d_model//n_heads
        self.max_len=max_len
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
    def _attend(self, x, quant_method=None):
        B,T,D = x.shape
        h = self.n_heads
        hd = self.head_dim
        xn = self.ln1(x)
        q = self.q_proj(xn)
        k = self.k_proj(xn)
        v = self.v_proj(xn)
        if quant_method is not None:
            k_np = k.detach().cpu().numpy().reshape(-1, D).astype(np.float32)
            v_np = v.detach().cpu().numpy().reshape(-1, D).astype(np.float32)
            k = torch.from_numpy(quant_method.encode_search(k_np).reshape(B,T,D)).to(x.device)
            v = torch.from_numpy(quant_method.encode_values(v_np).reshape(B,T,D)).to(x.device)
        qh = q.view(B,T,h,hd).transpose(1,2)
        kh = k.view(B,T,h,hd).transpose(1,2)
        vh = v.view(B,T,h,hd).transpose(1,2)
        scores = (qh @ kh.transpose(-2,-1)) / math.sqrt(hd)
        mask = torch.triu(torch.ones(T,T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        att = scores.softmax(dim=-1)
        out = att @ vh
        out = out.transpose(1,2).contiguous().view(B,T,D)
        return out, scores.detach(), att.detach()
    def forward(self, idx, quant_method=None, return_aux=False):
        B,T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok(idx) + self.pos(pos)
        att_out, scores, att = self._attend(x, quant_method=quant_method)
        x = x + self.o_proj(att_out)
        x = x + self.ff(x)
        logits = self.head(self.ln_f(x))
        if return_aux:
            return logits, att_out.detach(), scores, att
        return logits

def encode_stream(texts, vocab, max_tokens):
    ids=[]
    eos = vocab['<eos>']
    unk = vocab['<unk>']
    for txt in texts:
        for tok in tokenize(txt):
            ids.append(vocab.get(tok, unk))
            if len(ids) >= max_tokens:
                return np.array(ids, dtype=np.int64)
        ids.append(eos)
        if len(ids) >= max_tokens:
            return np.array(ids[:max_tokens], dtype=np.int64)
    return np.array(ids, dtype=np.int64)

def make_sequences(ids, seq_len=32, stride=32):
    xs=[]; ys=[]
    for start in range(0, len(ids) - seq_len - 1, stride):
        chunk = ids[start:start+seq_len+1]
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
    return np.array(xs, dtype=np.int64), np.array(ys, dtype=np.int64)


def main():
    outdir = Path('/mnt/data')
    dev = json.load(open(outdir/'squad_dev_v1.1.json'))
    train = json.load(open(outdir/'squad_train_v1.1.json'))
    dev_docs, dev_qas = extract_squad(dev)
    train_docs, _ = extract_squad(train)
    print('Loaded SQuAD.', flush=True)

    # RAG setup
    rag_doc_count = 800
    doc_ids = list(range(rag_doc_count))
    doc_map = {old:i for i,old in enumerate(doc_ids)}
    docs_text = [dev_docs[i][1] for i in doc_ids]
    queries_text=[]; labels=[]
    for q, doc_id, _ in dev_qas:
        if doc_id in doc_map:
            queries_text.append(q)
            labels.append(doc_map[doc_id])
    labels = np.array(labels, dtype=np.int32)
    print(f'RAG subset: {len(docs_text)} docs, {len(queries_text)} queries', flush=True)

    vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=2, max_features=15000, sublinear_tf=True)
    X_docs_sparse = vec.fit_transform(docs_text)
    X_queries_sparse = vec.transform(queries_text)
    rp = GaussianRandomProjection(n_components=256, random_state=SEED)
    X_docs = normalize(rp.fit_transform(X_docs_sparse)).astype(np.float32)
    X_queries = normalize(rp.transform(X_queries_sparse)).astype(np.float32)
    S_float = X_queries @ X_docs.T
    print('Built real-text retrieval embeddings.', flush=True)

    rag_rows=[]
    for m in METHODS:
        mm = Method(m.name, m.bits, m.family, m.q_kind, m.transform_kind).setup(d=X_docs.shape[1], seed=999)
        t0 = time.perf_counter()
        D = mm.encode_search(X_docs)
        Q = mm.encode_search(X_queries)
        encode_us = 1e6 * (time.perf_counter()-t0) / (len(X_docs)+len(X_queries))
        S = Q @ D.T
        metrics = retrieval_metrics(S, labels, k=10)
        metrics['score_corr_vs_float32'] = float(np.corrcoef(S.ravel(), S_float.ravel())[0,1])
        metrics['bits_per_dim'] = m.bits
        metrics['compression_x'] = 32.0 / m.bits
        metrics['payload_bytes_per_vec'] = X_docs.shape[1] * m.bits / 8.0
        metrics['encode_us_per_vec'] = encode_us
        rag_rows.append({'method':m.name,'family':m.family, **metrics})
    rag_df = pd.DataFrame(rag_rows).sort_values(['recall@1','mrr'], ascending=False)
    print('Finished RAG benchmarks.', flush=True)

    # KV / tiny LM setup
    train_contexts = [c for _,c in train_docs[:1500]]
    dev_contexts = [c for _,c in dev_docs[:300]]
    train_tokens=[]
    for txt in train_contexts:
        train_tokens.extend(tokenize(txt))
        train_tokens.append('<eos>')
        if len(train_tokens) >= 40000:
            break
    cnt = Counter(train_tokens)
    vocab = {'<pad>':0, '<unk>':1, '<eos>':2}
    for tok,_ in cnt.most_common(1500 - len(vocab)):
        if tok not in vocab:
            vocab[tok] = len(vocab)
    train_ids = encode_stream(train_contexts, vocab, max_tokens=40000)
    eval_ids = encode_stream(dev_contexts, vocab, max_tokens=12000)
    seq_len = 32
    xtr, ytr = make_sequences(train_ids, seq_len=seq_len, stride=32)
    xev, yev = make_sequences(eval_ids, seq_len=seq_len, stride=32)
    xev, yev = xev[:96], yev[:96]
    print(f'LM data: train_tokens={len(train_ids)}, eval_tokens={len(eval_ids)}, train_seqs={len(xtr)}, eval_seqs={len(xev)}', flush=True)

    model = TinyCausalLM(vocab_size=len(vocab), d_model=64, n_heads=4, d_ff=96, max_len=seq_len)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    ds = TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr))
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    model.train()
    max_steps = 25
    step = 0
    for xb, yb in dl:
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step >= max_steps:
            break
    print(f'Trained tiny LM for {step} steps.', flush=True)

    model.eval()
    xe = torch.from_numpy(xev)
    ye = torch.from_numpy(yev)
    with torch.no_grad():
        logits_f, att_out_f, scores_f, att_f = model(xe, quant_method=None, return_aux=True)
        loss_f = F.cross_entropy(logits_f.view(-1, logits_f.size(-1)), ye.view(-1)).item()
    ppl_f = math.exp(loss_f)
    pred_f = logits_f.argmax(dim=-1)

    cache_rows=[]
    with torch.no_grad():
        xn = model.ln1(model.tok(xe[:8]) + model.pos(torch.arange(seq_len).unsqueeze(0)))
        k_sample = model.k_proj(xn).reshape(-1, model.d_model).cpu().numpy().astype(np.float32)
    for m in METHODS:
        mm = Method(m.name, m.bits, m.family, m.q_kind, m.transform_kind).setup(d=model.d_model, seed=999)
        t0 = time.perf_counter(); _ = mm.encode_search(k_sample); _ = mm.encode_values(k_sample); elapsed = time.perf_counter()-t0
        enc_us_tok = 1e6 * elapsed / (2*len(k_sample))
        with torch.no_grad():
            logits_q, att_out_q, scores_q, att_q = model(xe, quant_method=mm, return_aux=True)
            loss_q = F.cross_entropy(logits_q.view(-1, logits_q.size(-1)), ye.view(-1)).item()
        ppl_q = math.exp(loss_q)
        logit_cos = F.cosine_similarity(logits_q.reshape(-1, logits_q.size(-1)), logits_f.reshape(-1, logits_f.size(-1)), dim=-1).mean().item()
        out_cos = F.cosine_similarity(att_out_q.reshape(-1, model.d_model), att_out_f.reshape(-1, model.d_model), dim=-1).mean().item()
        nexttok_agree = (logits_q.argmax(dim=-1) == pred_f).float().mean().item()
        att_kl = (att_f * ((att_f+1e-12).log() - (att_q+1e-12).log())).sum(dim=-1).mean().item()
        cache_rows.append({
            'method':m.name,
            'family':m.family,
            'bits_per_dim':m.bits,
            'compression_x':32.0 / m.bits,
            'kv_payload_bytes_per_token': 2 * model.d_model * m.bits / 8.0,
            'ppl': ppl_q,
            'delta_ppl': ppl_q - ppl_f,
            'logit_cos_vs_float32': logit_cos,
            'attn_out_cos_vs_float32': out_cos,
            'nexttok_agree_vs_float32': nexttok_agree,
            'attn_kl_vs_float32': att_kl,
            'encode_us_per_token': enc_us_tok,
        })
    cache_df = pd.DataFrame(cache_rows).sort_values(['ppl','attn_out_cos_vs_float32'], ascending=[True, False])
    print('Finished KV benchmarks.', flush=True)

    rag_df.to_csv(outdir/'real_rag_benchmark.csv', index=False)
    cache_df.to_csv(outdir/'real_kv_benchmark.csv', index=False)
    summary = []
    rag_base = rag_df.loc[rag_df.method=='float32'].iloc[0]
    for _, row in rag_df.iterrows():
        c = cache_df.loc[cache_df.method==row['method']].iloc[0]
        summary.append({
            'method': row['method'],
            'family': row['family'],
            'bits_per_dim': row['bits_per_dim'],
            'compression_x': row['compression_x'],
            'rag_recall@1': row['recall@1'],
            'rag_delta_vs_float32': row['recall@1'] - rag_base['recall@1'],
            'rag_mrr': row['mrr'],
            'kv_ppl': c['ppl'],
            'kv_delta_ppl_vs_float32': c['delta_ppl'],
            'kv_attn_out_cos': c['attn_out_cos_vs_float32'],
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(outdir/'real_combined_summary.csv', index=False)

    report = []
    report.append('# Real-data benchmark: RAG and KV-cache\n')
    report.append(f'- SQuAD v1.1 real texts/questions.\n')
    report.append(f'- RAG: {len(docs_text)} docs, {len(queries_text)} queries, TF-IDF + GaussianRandomProjection -> 256d.\n')
    report.append(f'- KV: tiny 1-layer causal LM on SQuAD contexts, d_model=64, heads=4, seq_len={seq_len}, train_steps={step}, float32 ppl={ppl_f:.3f}.\n')
    report.append('\n## RAG results\n')
    report.append(rag_df.to_markdown(index=False))
    report.append('\n\n## KV results\n')
    report.append(cache_df.to_markdown(index=False))
    (outdir/'real_benchmark_report.md').write_text('\n'.join(report), encoding='utf-8')

    print('RAG top:', flush=True)
    print(rag_df[['method','family','bits_per_dim','compression_x','recall@1','mrr','ndcg@10']].head(8).to_string(index=False), flush=True)
    print('\nKV top:', flush=True)
    print(cache_df[['method','family','bits_per_dim','compression_x','ppl','delta_ppl','attn_out_cos_vs_float32']].head(8).to_string(index=False), flush=True)
    print('\nSaved files.', flush=True)

if __name__ == '__main__':
    main()
