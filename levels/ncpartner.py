# Extra linear constraints on K_k(N): ribbon contractions x (x) eta, with eta a noncrossing partition of level m,
# landing at level <= k-1, must have zero crossing part. Do they cut K_k(N) further? (mod p)
import sys, itertools, numpy as np
from engine import kernel_basis, crossing_parts, parts, noncrossing, canon, PR
p = PR
k = int(sys.argv[1]); N = int(sys.argv[2]); mmax = int(sys.argv[3])
crk, W, free = kernel_basis(k, N, p); W = W.astype(np.int64); d = W.shape[1]
print(f'level {k}, N={N}: dim K = {d}', flush=True)
def contract(P, Q, kP, kQ, s, t, j, o1, o2):
    """Ribbon of partition P (kP legs) and Q (kQ legs): x_{s+i} -- y_{t+j-1-i}."""
    xs = [(s + i) % kP for i in range(j)]; ys = [(t + j - 1 - i) % kQ for i in range(j)]
    outs = [('x', (s + j + i) % kP) for i in range(kP - j)]
    if o1: outs.append(('x', xs[0]))
    outs += [('y', (t + j + i) % kQ) for i in range(kQ - j)]
    if o2 and j > 1: outs.append(('x', xs[-1]))
    par = {}
    def f(a):
        par.setdefault(a, a)
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    def u(a, b): par[f(a)] = f(b)
    for B in P:
        for a in B[1:]: u(('x', a), ('x', B[0]))
    for B in Q:
        for a in B[1:]: u(('y', a), ('y', B[0]))
    for a, b in zip(xs, ys): u(('x', a), ('y', b))
    nodes = [('x', i) for i in range(kP)] + [('y', i) for i in range(kQ)]
    roots = {f(v) for v in nodes}; oroots = [f(v) for v in outs]
    closed = len(roots - set(oroots)); bl = {}
    for pos, r in enumerate(oroots): bl.setdefault(r, set()).add(pos)
    return canon(list(bl.values())), closed, len(outs)
rows = []
for m in range(1, mmax + 1):
    for eta in [q for q in parts(m) if noncrossing(q)]:
        for s in range(k):
            for t in range(m):
                for j in range(1, min(k, m) + 1):
                    for o1 in (0, 1):
                        for o2 in (0, 1):
                            L = k + m - 2 * j + o1 + o2
                            if L > k - 1 or (o2 and j == 1): continue
                            out = {}
                            for a, P in enumerate(crk):
                                Pout, closed, nl = contract(P, eta, k, m, s, t, j, o1, o2)
                                if nl >= 4 and not noncrossing(Pout):
                                    out.setdefault(Pout, np.zeros(len(crk), dtype=np.int64))[a] += pow(N, closed, p)
                            for v in out.values():
                                rows.append((v % p) @ W % p)
    print(f'  after partners of level <= {m}: {len(rows)} constraint rows', flush=True)
M = np.array(rows) % p if rows else np.zeros((0, d), dtype=np.int64)
# rank mod p
r = 0; M = M.copy()
for c in range(d):
    nz = np.flatnonzero(M[r:, c])
    if len(nz) == 0: continue
    pr = r + nz[0]; M[[r, pr]] = M[[pr, r]]
    M[r] = (M[r] * pow(int(M[r, c]), p - 2, p)) % p
    oth = np.flatnonzero(M[:, c]); oth = oth[oth != r]
    if len(oth): M[oth] = (M[oth] - (M[oth, c][:, None] * M[r][None, :]) % p) % p
    r += 1
    if r == M.shape[0]: break
print(f'extra linear constraints on K: rank {r} -> K shrinks from {d} to {d - r}', flush=True)
