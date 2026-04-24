
import numpy as np, math, pandas as pd, time
from pathlib import Path

def fwht(a):
    a = a.copy().astype(np.float32)
    n = a.shape[-1]
    h = 1
    while h < n:
        a = a.reshape(*a.shape[:-1], -1, h * 2)
        x = a[..., :h].copy()
        y = a[..., h:2*h].copy()
        a[..., :h] = x + y
        a[..., h:2*h] = x - y
        a = a.reshape(*a.shape[:-2], -1)
        h *= 2
    return a / math.sqrt(n)

def random_signs(d, rng):
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=d)

def hadamard_rotate(X, signs):
    return fwht(X * signs)

def uniform_dequantize(X, bits=2, clip=None):
    levels = 2**bits
    if clip is None:
        clip = 3 * np.std(X)
    clip = float(clip) + 1e-8
    step = 2 * clip / (levels - 1)
    Q = np.clip(np.round((X + clip) / step), 0, levels - 1).astype(np.uint8)
    return Q.astype(np.float32) * step - clip

def five_level_dequantize(X):
    s = np.std(X)
    t1, t2 = 0.35*s, 1.1*s
    sel1 = (np.abs(X)>t1) & (np.abs(X)<=t2)
    a = np.mean(np.abs(X[sel1])) if np.any(sel1) else 0.7*s
    sel2 = np.abs(X)>t2
    b = np.mean(np.abs(X[sel2])) if np.any(sel2) else 1.7*s
    Q = np.zeros(X.shape, dtype=np.uint8)
    ax = np.abs(X)
    Q[(X>t1) & (ax<=t2)] = 1
    Q[(X<-t1) & (ax<=t2)] = 2
    Q[(X>t2)] = 3
    Q[(X<-t2)] = 4
    deq = np.zeros(X.shape, dtype=np.float32)
    deq[Q==1]=a; deq[Q==2]=-a; deq[Q==3]=b; deq[Q==4]=-b
    return deq

def random_orth_block(block, rng):
    A = rng.normal(size=(block, block))
    Q, R = np.linalg.qr(A)
    d = np.sign(np.diag(R)); d[d == 0] = 1
    return (Q * d).astype(np.float32)

def make_block_rotation(d, block, rng):
    return np.stack([random_orth_block(block, rng) for _ in range(d // block)], axis=0)

def apply_block_rotation(X, mats):
    b = mats.shape[1]
    Xr = X.reshape(-1, mats.shape[0], b)
    Y = np.einsum('mnb,nbc->mnc', Xr, mats, optimize=True)
    return Y.reshape(X.shape).astype(np.float32)

def blockwise_ternary_dequantize(X, block=4):
    Xr = X.reshape(-1, X.shape[-1]//block, block).astype(np.float32)
    std = np.std(Xr, axis=(0,2), keepdims=True).astype(np.float32) + 1e-8
    thr = 0.5 * std
    sel = np.abs(Xr) > thr
    absx = np.abs(Xr)
    num = np.sum(absx * sel, axis=(0,2), keepdims=True)
    den = np.sum(sel, axis=(0,2), keepdims=True)
    fallback = np.mean(absx, axis=(0,2), keepdims=True) + 1e-6
    amp = np.where(den > 0, num / np.maximum(den,1), fallback).astype(np.float32)
    Y = np.zeros_like(Xr, dtype=np.float32)
    Y = np.where(Xr > thr, amp, Y)
    Y = np.where(Xr < -thr, -amp, Y)
    return Y.reshape(X.shape).astype(np.float32)

def blockwise_signedmag7_dequantize(X, block=4):
    Xr = X.reshape(-1, X.shape[-1]//block, block).astype(np.float32)
    s = np.std(Xr, axis=(0,2), keepdims=True).astype(np.float32) + 1e-8
    t1, t2, t3 = 0.25*s, 0.75*s, 1.5*s
    absx = np.abs(Xr)
    def avg(mask, fb):
        num = np.sum(absx * mask, axis=(0,2), keepdims=True)
        den = np.sum(mask, axis=(0,2), keepdims=True)
        return np.where(den > 0, num / np.maximum(den,1), fb)
    a = avg((absx>t1)&(absx<=t2), 0.45*s)
    b = avg((absx>t2)&(absx<=t3), 1.0*s)
    c = avg(absx>t3, 1.9*s)
    Y = np.zeros_like(Xr, dtype=np.float32)
    Y = np.where((Xr>t1)&(absx<=t2), a, Y)
    Y = np.where((Xr<-t1)&(absx<=t2), -a, Y)
    Y = np.where((Xr>t2)&(absx<=t3), b, Y)
    Y = np.where((Xr<-t2)&(absx<=t3), -b, Y)
    Y = np.where(Xr>t3, c, Y)
    Y = np.where(Xr<-t3, -c, Y)
    return Y.reshape(X.shape).astype(np.float32)

def make_signed_permutation(d, rng):
    perm = rng.permutation(d)
    signs = rng.choice(np.array([-1.0,1.0], dtype=np.float32), size=d)
    return perm, signs

def apply_signed_permutation(X, perm, signs):
    return (X[:, perm] * signs).astype(np.float32)

def accuracy(scores, labels):
    return float(np.mean(scores.argmax(axis=1) == labels))

def scores(Q, P):
    return Q @ P.T

def make_bundled_dataset(d=512, n_classes=200, train_per_class=8, test_per_class=32, flip_p=0.36, seed=0):
    rng = np.random.default_rng(seed)
    bases = rng.choice(np.array([-1,1], dtype=np.int8), size=(n_classes,d))
    prototypes = np.zeros((n_classes,d), dtype=np.float32)
    for c in range(n_classes):
        X = np.repeat(bases[c:c+1], train_per_class, axis=0).copy()
        flips = rng.random((train_per_class,d)) < flip_p
        X[flips] *= -1
        prototypes[c] = X.astype(np.float32).sum(axis=0)
    tests=[]; labels=[]
    for c in range(n_classes):
        X = np.repeat(bases[c:c+1], test_per_class, axis=0).copy()
        flips = rng.random((test_per_class,d)) < flip_p
        X[flips] *= -1
        tests.append(X.astype(np.float32))
        labels.extend([c]*test_per_class)
    return prototypes, np.concatenate(tests, axis=0), np.array(labels, dtype=np.int32)

def make_symbolic_dataset(d=256, n_items=512, n_roles=32, n_memories=1200, n_bindings=16, seed=0):
    rng = np.random.default_rng(seed)
    item_bank = rng.choice(np.array([-1,1], dtype=np.int8), size=(n_items,d)).astype(np.float32)
    role_bank = rng.choice(np.array([-1,1], dtype=np.int8), size=(n_roles,d)).astype(np.float32)
    memories = np.zeros((n_memories,d), dtype=np.float32)
    q_roles = np.zeros((n_memories,d), dtype=np.float32)
    labels = np.zeros(n_memories, dtype=np.int32)
    for i in range(n_memories):
        roles = rng.choice(n_roles, size=n_bindings, replace=False)
        items = rng.integers(0, n_items, size=n_bindings)
        mem = np.zeros(d, dtype=np.float32)
        for r,it in zip(roles, items):
            mem += role_bank[r] * item_bank[it]
        pick = rng.integers(0, n_bindings)
        memories[i] = mem
        q_roles[i] = role_bank[roles[pick]]
        labels[i] = items[pick]
    return item_bank, memories, q_roles, labels

def make_real_dataset(d=512, n_classes=160, train_per_class=8, test_per_class=24, noise=0.2, seed=0):
    rng=np.random.default_rng(seed)
    P = rng.normal(size=(n_classes,d)).astype(np.float32)
    P /= np.linalg.norm(P, axis=1, keepdims=True)+1e-8
    train=[]
    for c in range(n_classes):
        X = P[c] + noise * rng.normal(size=(train_per_class,d)).astype(np.float32)
        train.append(X)
    prototypes = np.array([x.mean(axis=0) for x in train], dtype=np.float32)
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)+1e-8
    tests=[]; labels=[]
    for c in range(n_classes):
        X = P[c] + noise * rng.normal(size=(test_per_class,d)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)+1e-8
        tests.append(X)
        labels.extend([c]*test_per_class)
    return prototypes, np.concatenate(tests, axis=0), np.array(labels, dtype=np.int32)

# main omitted; notebook / user can compose from these
