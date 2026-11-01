# Level 8, N = 11: fill the even rotation-character blocks of the degree-3 Macaulay space with triple-contraction cubics.
# Keeps one RREF basis per even block, reduces new cubic rows on the fly, reports remaining gaps.
import sys, time, itertools, numpy as np
from engine import parts, crossing_parts, labels, BatchRank
from triple8 import first_forms, second_cubics
p = 1048433; k = 8; N = 11
exec(open('cubic.py').read().split("which = [int(x)")[0])          # builds spans by character, chars, cub, ylist, etc.
blocks = [int(x) for x in sys.argv[1].split(',')]
cubidx = {m: i for i, m in enumerate(cub)}
def mm(A, B):                                                       # exact (A @ B) mod p for float arrays with entries < p
    Bl = np.mod(B, 1024); Bh = np.floor_divide(B, 1024)
    return np.mod(np.mod(A @ Bl, p) + np.mod(np.mod(A @ Bh, p) * 1024, p), p)
class Block:
    def __init__(self, psi):
        self.cols = cchar[psi]; self.colsel = np.array([cubidx[m] for m in self.cols]); self.psi = psi
        ci = {m: i for i, m in enumerate(self.cols)}; rows = []
        for i in range(d):
            chi = (psi - chars[i]) % 8
            for q in spans[chi]:
                row = np.zeros(len(self.cols), dtype=np.int64)
                for mi in np.flatnonzero(q):
                    a, b = ylist[mi]; m = tuple(sorted((a, b, i))); row[ci[m]] = (row[ci[m]] + q[mi]) % p
                rows.append(row)
        self.B = np.zeros((0, len(self.cols))); self.piv = []
        self.add(np.array(rows))
    def add(self, X):
        X = np.mod(np.asarray(X, dtype=np.float64), p)
        if self.B.shape[0]: X = np.mod(X - mm(X[:, self.piv], self.B), p)
        X = X.astype(np.int64); r = 0; piv = []
        for c in range(X.shape[1]):                                 # RREF of residual batch
            nz = np.flatnonzero(X[r:, c])
            if len(nz) == 0: continue
            pr = r + nz[0]; X[[r, pr]] = X[[pr, r]]; X[r] = (X[r] * pow(int(X[r, c]), p - 2, p)) % p
            oth = np.flatnonzero(X[:, c]); oth = oth[oth != r]
            if len(oth): X[oth] = (X[oth] - (X[oth, c][:, None] * X[r][None, :]) % p) % p
            piv.append(c); r += 1
            if r == X.shape[0]: break
        if r == 0: return 0
        Nw = X[:r].astype(np.float64)
        if self.B.shape[0]: self.B = np.mod(self.B - mm(self.B[:, piv], Nw), p)
        self.B = np.vstack([self.B, Nw]); self.piv += piv
        return r
    @property
    def gap(self): return len(self.cols) - len(self.piv)
t = time.time(); BL = [Block(psi) for psi in blocks]
print('initial gaps:', {b.psi: b.gap for b in BL}, f'({time.time()-t:.0f}s)', flush=True)
E = np.load('Exxxxxxxx.npy'); W = np.load('Wmodxxxxxxxx.npy')
Wy = np.mod(np.mod(W @ (E & 0x3FF), p) + np.mod(np.mod(W @ (E >> 10), p) * 1024, p), p).astype(np.float64)
crk = crossing_parts(k); Lc, nbc = labels(crk, k)
allP = sorted(parts(k)); Lall, nball = labels(allP, k); allP_index = {P: i for i, P in enumerate(allP)}
iu = np.triu_indices(d)
mp = np.array([[cubidx[tuple(sorted((a, b, e)))] for e in range(d)] for a, b in zip(*iu)])
mon_index = {'list': cub, 'map': mp}
def confs(target):
    return [(s, t_, j, o1, o2) for s in range(k) for t_ in range(k) for j in range(1, k + 1) for o1 in (0, 1) for o2 in (0, 1)
            if not (o2 and j == 1) and target(2 * k - 2 * j + o1 + o2)]
import random
firsts = confs(lambda L: L == k)                              # all positions s, t
seconds = [c for c in confs(lambda L: 4 <= L <= k - 1) if c[0] == 0 and c[2] == 5]
rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
maxcombos = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
order = []
while len(order) < min(maxcombos, len(firsts) * len(seconds)):
    f = rng.choice(firsts)
    for s2 in rng.sample(seconds, 8): order.append((f, s2))
cache = {}; Ycache = {}
for n_, (c1, c2) in enumerate(order):
    if all(b.gap == 0 for b in BL): break
    t = time.time()
    if c1 not in Ycache:
        Ycache.clear(); Ycache[c1] = first_forms(Lc, nbc, k, c1, Wy, N, allP_index)
    rows = second_cubics(Lall, nball, Lc, nbc, k, c2, Ycache[c1], Wy, N, mon_index, cache=cache)
    if rows:
        R = np.array(rows)
        for b in BL: b.add(R[:, b.colsel])
    print(f'{n_+1}: first {c1} second {c2}: {len(rows)} cubics; gaps {[b.gap for b in BL]} ({time.time()-t:.0f}s)', flush=True)
print('final gaps:', {b.psi: b.gap for b in BL}, flush=True)
