# Exact check at N=6, level 7: c_0-coefficients (relation direction) of all quadric rows
# from the configurations needed to reach rank 91, in exact integer arithmetic.
import numpy as np
from math import lcm
exec(open('nsix.py').read().split("basisr = []")[0])
from fractions import Fraction as Fr
L = lcm(*[Fr(x).denominator for v in vecs for x in v.values()])
WI = np.empty((len(crk), m), dtype=object); WI[:] = 0
for b, v in enumerate(vecs):
    for c, x in v.items(): WI[c, b] = int(Fr(x) * L)
k = 7
confs = [(s_, t, j, o1, o2) for s_ in range(k) for t in range(k) for j in range(4, k + 1)
         for o1 in (0, 1) for o2 in (0, 1) if 2 * k - 2 * j + o1 + o2 <= k - 1]
target = (m - 1) * m // 2
def red(v):
    for pc, bv in basisr:
        if v[pc]: v = (v - v[pc] * bv) % p
    return v
basisr = []; bad = 0; rowsdone = 0
for conf in confs:
    K, C, keys = pair_tensor(crk, k, conf)
    Npow = np.array([N ** e for e in range(int(C.max()) + 1)], dtype=object); Mw = Npow[C.astype(np.int64)]
    for kid, key in enumerate(keys):
        if noncrossing(rgs_to_partition(key)): continue
        mask = (K == kid)
        if not mask.any(): continue
        pi, qi = np.nonzero(mask)
        wts = Mw[pi, qi]
        # exact c_0 row: Q[0,b] + Q[b,0] for b, with Q[a,b] = sum w_a[p] w_b[q] N^c
        a0 = WI[pi, 0] * wts                      # object vector over pairs
        row0 = (a0[:, None] * WI[qi, :]).sum(axis=0) + ((WI[pi, :] * wts[:, None]) * WI[qi, 0][:, None]).sum(axis=0)
        if any(int(x) != 0 for x in row0): bad += 1
        rowsdone += 1
        # mod-p rank bookkeeping (same as before)
        Em = np.where(mask, np.array([pow(N, e, p) for e in range(int(C.max()) + 1)], dtype=np.int64)[C], 0)
        Q = mulmod(mulmod(np.ascontiguousarray(W.T), Em), W); Q = (Q + Q.T) % p
        v = np.array([Q[a, b] for a, b in idx], dtype=np.int64)
        v = red(v[rest].copy()); nz = np.nonzero(v)[0]
        if len(nz):
            pc = nz[0]; v = (v * pow(int(v[pc]), p - 2, p)) % p; basisr.append((pc, v))
    if len(basisr) == target: break
print(f'rows examined: {rowsdone}; rows with a nonzero exact c_0 coefficient: {bad}; rank reached: {len(basisr)} of {target}')
