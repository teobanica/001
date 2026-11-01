# Select 105 independent level-7 quadric rows at N=11 (mod p), using the symbolic basis w(N) of stepone.
import pickle, numpy as np
from fractions import Fraction as Fr
from ribbon import pair_tensor, rgs_to_partition, mulmod, P_MOD
from levlin import noncrossing
import poly as P
p = P_MOD; k = 7; N = 11
S = pickle.load(open('stepone_xxxxxxx.pkl', 'rb'))
crk, rec = S['crk'], S['rec']; d = len(rec)
def md(x): return x.numerator * pow(x.denominator, -1, p) % p
W = np.zeros((len(crk), d), dtype=np.int64)
for b in range(d):
    for c, pol in rec[b].items(): W[c, b] = md(P.ev(pol, N))
confs = [(s, t, j, o1, o2) for s in range(2) for t in range(k) for j in range(4, k + 1)
         for o1 in (0, 1) for o2 in (0, 1) if 2 * k - 2 * j + o1 + o2 <= k - 1]
basis = []; chosen = []
def reduce(v):
    v = v.copy()
    for pc, bv in basis:
        if v[pc]: v = (v - v[pc] * bv) % p
    return v
for conf in confs:
    K, C, keys = pair_tensor(crk, k, conf)
    Npow = np.array([pow(N, e, p) for e in range(int(C.max()) + 1)], dtype=np.int64); Mw = Npow[C]
    for kid, key in enumerate(keys):
        if noncrossing(rgs_to_partition(key)): continue
        E = np.where(K == kid, Mw, 0)
        if not E.any(): continue
        Q = mulmod(mulmod(np.ascontiguousarray(W.T), E), W); Q = (Q + Q.T) % p
        v = np.array([Q[a, b] for a in range(d) for b in range(a, d)], dtype=np.int64)
        v = reduce(v); nz = np.nonzero(v)[0]
        if len(nz):
            pc = nz[0]; v = (v * pow(int(v[pc]), p - 2, p)) % p
            basis.append((pc, v)); chosen.append((conf, key))
    print(conf, len(chosen), flush=True)
    if len(chosen) == d * (d + 1) // 2: break
print('selected rows:', len(chosen), ' configs used:', len({c for c, _ in chosen}))
pickle.dump(chosen, open('select.pkl', 'wb'))
