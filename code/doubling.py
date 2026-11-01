#!/usr/bin/env python3
# doubling.py -- the repetition lemma, for "On the maximality conjecture for quantum permutation groups"
# (Claude and Teo), Section 3 (Conjecture "repetition lemma": <S_{4m}, phi_m(S_4^+)> = S_{4m}^+).
#
#   python3 doubling.py            m = 1, 2, 3 and k = 4..7: dim Fix_{<S_4m, phi_m(S_4^+)>}(u^{(x)k})
#
# Method.  phi_m(S_4^+) has fundamental representation u (x) 1_m, so its fixed vectors at level k are
# Fix_{S_4^+}(k) (x) (C^m)^{(x)k}.  A vector x = sum_pi a_pi xi_pi^{(4m)} has, at the color multi-index c, the slice
# sum_{pi <= ker c} a_pi xi_pi^{(4)}, so x is fixed iff for every partition kappa of the legs with at most m blocks,
#       sum_{pi <= kappa} a_pi xi_pi^{(4)}  lies in  span{ xi_sigma^{(4)} : sigma in NC(k) }.
# With the Gram matrix G of the xi_pi^{(4)} (entries 4^{#blocks of pi v sigma}), this is S D_kappa a = 0, where
# S = G - G_{.,NC} G_NC^{-1} G_{NC,.} is the Schur complement and D_kappa keeps the pi <= kappa.  So
#       dim = dim span{xi_pi^{(4m)}} - rank (S D_kappa)_kappa ,
# computed exactly modulo p = 1048573 after a random compression (the rank can only drop, so the output is an upper
# bound, and a Catalan output proves equality).
import sys, itertools, time
import numpy as np

P = 1048573

def set_partitions(k):
    def rec(l):
        if l == 0: yield []; return
        for q in rec(l - 1):
            for i in range(len(q)): yield q[:i] + [q[i] + [l - 1]] + q[i + 1:]
            yield q + [[l - 1]]
    return [sorted(sorted(b) for b in q) for q in rec(k)]

def noncrossing(q):
    for A, B in itertools.permutations(q, 2):
        for a1, a2 in itertools.combinations(A, 2):
            for b1, b2 in itertools.combinations(B, 2):
                if a1 < b1 < a2 < b2: return False
    return True

def labels(q, k):
    lab = [0] * k
    for j, b in enumerate(q):
        for l in b: lab[l] = j
    return lab

def join_blocks(L1, L2, k):
    """Number of blocks of the join of the partitions with label arrays L1, L2 (union-find on k points)."""
    par = list(range(k))
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for L in (L1, L2):
        first = {}
        for l in range(k):
            if L[l] in first: par[f(l)] = f(first[L[l]])
            else: first[L[l]] = l
    return len({f(l) for l in range(k)})

def solve_mod_p(A, B):
    """X with A X = B (mod P), A square invertible mod P, by Gauss-Jordan elimination."""
    n = A.shape[0]; M = np.mod(np.hstack([A, B]), P).astype(np.float64)
    for c in range(n):
        nz = np.flatnonzero(M[c:, c]); assert len(nz), 'singular mod p'
        pr = c + nz[0]
        if pr != c: M[[c, pr]] = M[[pr, c]]
        M[c] = np.mod(M[c] * pow(int(M[c, c]), P - 2, P), P)
        f = M[:, c].copy(); f[c] = 0; nzf = np.flatnonzero(f)
        if len(nzf): M[nzf] = np.mod(M[nzf] - np.outer(f[nzf], M[c]), P)
    return M[:, n:]

def rank_mod_p(A, bs=64):
    A = np.mod(A, P).astype(np.float64); n, m = A.shape; r = 0; c = 0
    while c < m and r < n:
        c1 = min(c + bs, m); w = c1 - c
        panel = A[r:, c:c1].copy(); F = np.zeros((n - r, w)); inv = []; q = 0
        for col in range(w):
            if q >= n - r: break
            nz = np.flatnonzero(panel[q:, col])
            if len(nz) == 0: continue
            pr = q + nz[0]
            if pr != q:
                panel[[q, pr]] = panel[[pr, q]]; F[[q, pr]] = F[[pr, q]]; A[[r + q, r + pr]] = A[[r + pr, r + q]]
            iv = pow(int(panel[q, col]), P - 2, P); inv.append(iv)
            panel[q] = np.mod(panel[q] * iv, P)
            f = panel[q + 1:, col].copy()
            panel[q + 1:] = np.mod(panel[q + 1:] - np.outer(f, panel[q]), P); F[q + 1:, q] = f
            q += 1
        T = A[r:, c1:]
        if T.shape[1]:
            for j in range(q): T[j] = np.mod(np.mod(T[j] - F[j, :j] @ T[:j], P) * inv[j], P)
            if q < n - r: T[q:] = np.mod(T[q:] - F[q:, :q] @ T[:q], P)
        r += q; c = c1
    return r

def mulmod(A, B, chunk=4096):
    """A @ B mod P, exact: inner dimension split in chunks of <= 4096 (sums < 2^52)."""
    out = np.zeros((A.shape[0], B.shape[1]))
    for i in range(0, A.shape[1], chunk): out = np.mod(out + A[:, i:i + chunk] @ B[i:i + chunk], P)
    return out

def joint_dim(m, k, seed=1):
    t0 = time.time(); parts = set_partitions(k); B = len(parts)
    L = [labels(q, k) for q in parts]; nb = [len(q) for q in parts]
    nc = np.array([noncrossing(q) for q in parts]); NC = np.flatnonzero(nc)
    G = np.empty((B, B))
    for i in range(B):
        for j in range(i, B):
            G[i, j] = G[j, i] = pow(4, join_blocks(L[i], L[j], k), P)
    # Schur complement S = G - G[:,NC] G_NC^{-1} G[NC,:]  (mod P)
    X = solve_mod_p(G[np.ix_(NC, NC)], G[NC, :])
    S = np.mod(G - mulmod(G[:, NC], X), P)
    # refinement: pi <= kappa
    def refines(pi, ka):
        return all(len({L[ka][l] for l in b}) == 1 for b in parts[pi])
    kappas = [i for i in range(B) if nb[i] <= m]
    dimspan = sum(1 for i in range(B) if nb[i] <= 4 * m)
    target = dimspan - len(NC); R_ = max(target, 0) + 20
    rng = np.random.default_rng(seed); Mc = np.zeros((R_, B))
    for ka in kappas:
        D = np.array([refines(pi, ka) for pi in range(B)])
        Rk = rng.integers(0, 1 << 12, size=(R_, B)).astype(np.float64)
        Mc = np.mod(Mc + mulmod(mulmod(Rk, S), np.diag(D.astype(np.float64))), P)
    rank = rank_mod_p(Mc)
    return dimspan - rank, time.time() - t0

CAT = {4: 14, 5: 42, 6: 132, 7: 429, 8: 1430}

if __name__ == '__main__':
    todo = [(1, k) for k in range(4, 8)] + [(2, k) for k in range(4, 8)] + [(3, k) for k in range(4, 8)]
    if len(sys.argv) > 2: todo = [(int(sys.argv[1]), int(sys.argv[2]))]
    for (m, k) in todo:
        d, secs = joint_dim(m, k)
        print(f'm={m} (N={4*m}), k={k}: dim = {d}  (Catalan {CAT[k]})  [{secs:.0f} s]', flush=True)
