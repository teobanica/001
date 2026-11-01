# Degree-3 Macaulay test for the level-8 quadrics (N=11) modulo p = 1 mod 8, split by rotation characters.
# If every character block has full rank, the quadrics generate all cubics, so their common zero set is empty.
import sys, time, itertools, numpy as np
from engine import crossing_parts, canon, BatchRank
p = 1048433; k = 8
DATA = np.load('data.npz'); W = DATA['W'].astype(np.int64); free = list(DATA['free']); d = W.shape[1]
B = DATA['span'].astype(np.int64)
BR = BatchRank(d * (d + 1) // 2, p); BR.add(B); span = BR.B.astype(np.int64)
print('quadric span dimension mod p:', span.shape[0], flush=True)
iu = np.triu_indices(d)
crk = crossing_parts(k); idx = {P: a for a, P in enumerate(crk)}
perm = np.array([idx[canon([set((x + 1) % k for x in Bk) for Bk in P])] for P in crk])
PW = np.zeros_like(W); PW[perm] = W                      # (Pm W)[perm[a]] = W[a]
Rot = PW[free] % p                                        # 66 x 66, action on coordinates
# primitive 8th root of unity mod p
g = next(x for x in range(2, p) if pow(x, (p - 1) // 2, p) != 1)
om = pow(g, (p - 1) // 8, p); assert pow(om, 4, p) == p - 1
def nullspace_mod(M):
    M = M.copy() % p; rows, cols = M.shape; r = 0; piv = []
    for c in range(cols):
        nz = np.flatnonzero(M[r:, c])
        if len(nz) == 0: continue
        pr = r + nz[0]; M[[r, pr]] = M[[pr, r]]
        M[r] = (M[r] * pow(int(M[r, c]), p - 2, p)) % p
        oth = np.flatnonzero(M[:, c]); oth = oth[oth != r]
        if len(oth): M[oth] = (M[oth] - (M[oth, c][:, None] * M[r][None, :]) % p) % p
        piv.append(c); r += 1
        if r == rows: break
    fr = [c for c in range(cols) if c not in piv]
    N_ = np.zeros((cols, len(fr)), dtype=np.int64)
    for b, f in enumerate(fr):
        N_[f, b] = 1
        for i, pc in enumerate(piv): N_[pc, b] = (-M[i, f]) % p
    return N_
cols_E = []; chars = []
for j in range(8):
    Ej = nullspace_mod((Rot - pow(om, j, p) * np.eye(d, dtype=np.int64)) % p)
    cols_E.append(Ej); chars += [j] * Ej.shape[1]
E = np.hstack(cols_E) % p
print('eigenspace dimensions:', [e.shape[1] for e in cols_E], ' total', E.shape[1], flush=True)
assert E.shape[1] == d
chars = np.array(chars)
def matmul_mod(A, Bm):
    Bl = Bm & 0x3FF; Bh = Bm >> 10
    return ((A @ Bl) % p + ((A @ Bh) % p) * 1024) % p
# forms in y: S' = E^T S E, polynomial coefficients
ymon = {}; ylist = []
for a in range(d):
    for b in range(a, d): ymon[(a, b)] = len(ylist); ylist.append((a, b))
mchar = np.array([(chars[a] + chars[b]) % 8 for a, b in ylist])
comps = {chi: [] for chi in range(8)}
Et = E.T.copy()
for v in span:
    S = np.zeros((d, d), dtype=np.int64); S[iu] = v; S = (S + S.T - np.diag(np.diag(S))) % p
    Sp = matmul_mod(matmul_mod(Et, S), E)
    coef = np.array([Sp[a, b] if a == b else (2 * Sp[a, b]) % p for a, b in ylist], dtype=np.int64)
    for chi in range(8):
        cc = np.where(mchar == chi, coef, 0)
        if cc.any(): comps[chi].append(cc)
spans = {}
for chi in range(8):
    if comps[chi]:
        R = BatchRank(len(ylist), p); R.add(np.array(comps[chi])); spans[chi] = R.B.astype(np.int64)
    else: spans[chi] = np.zeros((0, len(ylist)), dtype=np.int64)
print('span dimension by character:', [spans[c].shape[0] for c in range(8)], ' total', sum(spans[c].shape[0] for c in range(8)), flush=True)
# cubic monomials by character
cub = [m for m in itertools.combinations_with_replacement(range(d), 3)]
cchar = {}
for m in cub: cchar.setdefault(int(sum(chars[list(m)]) % 8), []).append(m)
def rank_mod(M):
    M = M % p; r = 0; rows, cols = M.shape
    for c in range(cols):
        nz = np.flatnonzero(M[r:, c])
        if len(nz) == 0: continue
        pr = r + nz[0]
        if pr != r: M[[r, pr]] = M[[pr, r]]
        M[r, c:] = (M[r, c:] * pow(int(M[r, c]), p - 2, p)) % p
        below = r + 1 + np.flatnonzero(M[r + 1:, c])
        if len(below):
            M[below, c:] = (M[below, c:] - (M[below, c][:, None] * M[r, c:][None, :]) % p) % p
        r += 1
        if r == rows: break
    return r
which = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else list(range(8))
for psi in which:
    t = time.time(); cols = cchar[psi]; ci = {m: i for i, m in enumerate(cols)}
    rows = []
    for i in range(d):
        chi = (psi - chars[i]) % 8
        for q in spans[chi]:
            row = np.zeros(len(cols), dtype=np.int64)
            for mi in np.flatnonzero(q):
                a, b = ylist[mi]; m = tuple(sorted((a, b, i))); row[ci[m]] = (row[ci[m]] + q[mi]) % p
            rows.append(row)
    M = np.array(rows)
    rk = rank_mod(M)
    print(f'character {psi}: {M.shape[0]} rows x {len(cols)} cubic monomials, rank {rk} -> {"FULL" if rk == len(cols) else "deficient by " + str(len(cols) - rk)}   ({time.time()-t:.0f}s)', flush=True)
