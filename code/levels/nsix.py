# Level 7 at N = 6: work modulo the Moebius relation sum_pi mu(0,pi) xi_pi = 0.
import sys, numpy as np, math
from fractions import Fraction as Fr
sys.argv = ['x', '7']
exec(open('stepone.py').read().split("t = time.time(); free, basis = kernel(11)")[0])
from ribbon import pair_tensor, rgs_to_partition, mulmod, P_MOD
from levlin import noncrossing
p = P_MOD; N = 6
free, basis = kernel(N)
m = len(free); print('N=6: dim ker M(6) =', m, flush=True)
# Moebius relation, crossing part
mu = lambda P_: math.prod((-1) ** (len(B) - 1) * math.factorial(len(B) - 1) for B in P_)
r = {a: Fr(mu(P_)) for a, P_ in enumerate(crk)}
# r in ker M(6)?
inker = all(sum(ra.get(c, 0) * r[c] + N * rb.get(c, 0) * r[c] for c in set(ra) | set(rb)) == 0 for ra, rb in zip(rowsA, rowsB))
print('crossing part of the relation lies in ker M(6):', inker, flush=True)
# basis with r first: express r in the free-column basis (coordinates = values at free columns)
coords = [r[f] for f in free]
j0 = next(i for i, x in enumerate(coords) if x != 0)
vecs = [r] + [basis[i] for i in range(m) if i != j0]       # still a basis of ker M(6)
def md(x): return x.numerator * pow(x.denominator, -1, p) % p
W = np.zeros((len(crk), m), dtype=np.int64)
for b, v in enumerate(vecs):
    for c, x in v.items(): W[c, b] = md(Fr(x))
idx = [(a, b) for a in range(m) for b in range(a, m)]
with0 = [i for i, (a, b) in enumerate(idx) if a == 0]
rest = [i for i, (a, b) in enumerate(idx) if a > 0]
basisr = []; maxc0 = 0; rank = 0
def red(v):
    for pc, bv in basisr:
        if v[pc]: v = (v - v[pc] * bv) % p
    return v
k = 7
confs = [(s, t, j, o1, o2) for s in range(k) for t in range(k) for j in range(4, k + 1)
         for o1 in (0, 1) for o2 in (0, 1) if 2 * k - 2 * j + o1 + o2 <= k - 1]
target = (m - 1) * m // 2
for conf in confs:
    K, C, keys = pair_tensor(crk, k, conf)
    Npow = np.array([pow(N, e, p) for e in range(int(C.max()) + 1)], dtype=np.int64); Mw = Npow[C]
    for kid, key in enumerate(keys):
        if noncrossing(rgs_to_partition(key)): continue
        E = np.where(K == kid, Mw, 0)
        if not E.any(): continue
        Q = mulmod(mulmod(np.ascontiguousarray(W.T), E), W); Q = (Q + Q.T) % p
        v = np.array([Q[a, b] for a, b in idx], dtype=np.int64)
        if v[with0].any(): maxc0 += 1
        v = red(v[rest].copy()); nz = np.nonzero(v)[0]
        if len(nz):
            pc = nz[0]; v = (v * pow(int(v[pc]), p - 2, p)) % p; basisr.append((pc, v))
    if len(basisr) == target: break
print(f'quadric rows involving c_0 (the relation direction): {maxc0}  (expected 0)')
print(f'rank of quadrics in the remaining {m-1} variables: {len(basisr)} of {target}')
