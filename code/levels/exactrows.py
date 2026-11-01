# Exact level-7 quadric rows (selected 105) at integer N, then interpolation in N.
import pickle, sys, time, numpy as np
from fractions import Fraction as Fr
from math import lcm
from ribbon import pair_tensor
import poly as P
k = 7
S = pickle.load(open('stepone_xxxxxxx.pkl', 'rb')); crk, rec = S['crk'], S['rec']; d = len(rec)
chosen = pickle.load(open('select.pkl', 'rb'))
sigma = lcm(*[c.denominator for v in rec for pol in v.values() for c in pol])
recI = [{c: [int(x * sigma) for x in pol] for c, pol in v.items()} for v in rec]   # integer polys
confs = sorted({c for c, _ in chosen})
t = time.time(); tens = {conf: pair_tensor(crk, k, conf) for conf in confs}
print(f'pair tensors for {len(confs)} configs: {time.time()-t:.0f}s', flush=True)
rowspec = []
for conf, key in chosen:
    K, C, keys = tens[conf]; kid = keys.index(key)
    pi, qi = np.nonzero(K == kid); rowspec.append((pi, qi, C[pi, qi].astype(np.int64)))
def evI(pol, N):
    r = 0
    for c in reversed(pol): r = r * N + c
    return r
def rows_at(N):
    W = np.empty((len(crk), d), dtype=object); W[:] = 0
    for b in range(d):
        for c, pol in recI[b].items(): W[c, b] = evI(pol, N)
    out = []
    Npow = [N ** e for e in range(8)]
    for pi, qi, ci in rowspec:
        Sx = np.empty((len(crk), d), dtype=object); Sx[:] = 0
        contrib = W[qi, :] * np.array([Npow[c] for c in ci], dtype=object)[:, None]
        np.add.at(Sx, pi, contrib)
        Q = W.T.dot(Sx); Q = Q + Q.T
        out.append([int(Q[a, b]) for a in range(d) for b in range(a, d)])
    return out
Ns = list(range(1, 33))
data = {}
for N in Ns:
    t = time.time(); data[N] = rows_at(N); print(f'N={N}: {time.time()-t:.1f}s', flush=True)
DEGB = 25
polys = []
for r in range(len(chosen)):
    row = []
    for e in range(d * (d + 1) // 2):
        ys = [data[N][r][e] for N in Ns[:DEGB + 1]]
        pol = P.interp(Ns[:DEGB + 1], ys)
        assert all(P.ev(pol, N) == data[N][r][e] for N in Ns[DEGB + 1:]), (r, e)
        row.append(pol)
    polys.append(row)
maxdeg = max(P.deg(p_) for row in polys for p_ in row)
print('interpolation checked on 6 extra points; max degree', maxdeg, flush=True)
print('row degrees:', sorted(max(P.deg(p_) for p_ in row) for row in polys), flush=True)
pickle.dump({'polys': polys, 'chosen': chosen, 'sigma': sigma}, open('exactrows.pkl', 'wb'))
