# Triple contractions at level k (small k, plain Python): y = R1(x (x) x) at level k (full formal vector),
# then R2(y (x) x) landing at level <= k-1 must be noncrossing: cubic equations in the coordinates c of x in K_k.
import sys, itertools, numpy as np
from engine import kernel_basis, crossing_parts, parts, noncrossing, canon, PR
p = PR
k = int(sys.argv[1]); N = int(sys.argv[2]); nfirst = int(sys.argv[3]); nsecond = int(sys.argv[4])
crk, W, free = kernel_basis(k, N, p); W = W.astype(np.int64); d = W.shape[1]
X = [{crk[a]: int(W[a, b]) for a in range(len(crk)) if W[a, b]} for b in range(d)]   # basis vectors as dicts
def contract(P, Q, kP, kQ, s, t, j, o1, o2):
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
    for B in P:
        for a in B[1:]: par[f(('x', a))] = f(('x', B[0]))
    for B in Q:
        for a in B[1:]: par[f(('y', a))] = f(('y', B[0]))
    for a, b in zip(xs, ys): par[f(('x', a))] = f(('y', b))
    nodes = [('x', i) for i in range(kP)] + [('y', i) for i in range(kQ)]
    roots = {f(v) for v in nodes}; oroots = [f(v) for v in outs]
    bl = {}
    for pos, r in enumerate(oroots): bl.setdefault(r, set()).add(pos)
    return canon(list(bl.values())), len(roots - set(oroots)), len(outs)
def confs(target):
    out = []
    for s in range(k):
        for t in range(k):
            for j in range(1, k + 1):
                for o1 in (0, 1):
                    for o2 in (0, 1):
                        if o2 and j == 1: continue
                        L = 2 * k - 2 * j + o1 + o2
                        if target(L): out.append((s, t, j, o1, o2))
    return out
first = [c for c in confs(lambda L: L == k) if c[0] == 0 and c[1] == 0][:nfirst]
second = [c for c in confs(lambda L: 4 <= L <= k - 1) if c[0] == 0][:nsecond]
print(f'level {k}, N={N}: dim K {d}; first contractions {len(first)}, second {len(second)}', flush=True)
pairs = [(a, b) for a in range(d) for b in range(a, d)]
mon3 = list(itertools.combinations_with_replacement(range(d), 3)); m3 = {m: i for i, m in enumerate(mon3)}
rows = []
for c1 in first:
    # y[pi] as quadratic form: dict pi -> {(a,b): coeff}
    Y = {}
    for a in range(d):
        for b in range(d):
            for Pa, va in X[a].items():
                for Pb, vb in X[b].items():
                    out, closed, nl = contract(Pa, Pb, k, k, *c1)
                    key = (min(a, b), max(a, b))
                    dd = Y.setdefault(out, {}); dd[key] = (dd.get(key, 0) + va * vb * pow(N, closed, p)) % p
    for c2 in second:
        Z = {}
        for pi, form in Y.items():
            for e in range(d):
                for Pe, ve in X[e].items():
                    out, closed, nl = contract(pi, Pe, k, k, *c2)
                    if nl < 4 or noncrossing(out): continue
                    row = Z.setdefault(out, np.zeros(len(mon3), dtype=np.int64))
                    w = ve * pow(N, closed, p) % p
                    for (a, b), cf in form.items():
                        row[m3[tuple(sorted((a, b, e)))]] = (row[m3[tuple(sorted((a, b, e)))]] + cf * w) % p
        rows += list(Z.values())
    print(f'  first {c1}: {len(rows)} cubic rows so far', flush=True)
def rank_mod(M):
    M = M.copy() % p; r = 0
    for c in range(M.shape[1]):
        nz = np.flatnonzero(M[r:, c])
        if len(nz) == 0: continue
        pr = r + nz[0]; M[[r, pr]] = M[[pr, r]]
        M[r] = (M[r] * pow(int(M[r, c]), p - 2, p)) % p
        oth = np.flatnonzero(M[:, c]); oth = oth[oth != r]
        if len(oth): M[oth] = (M[oth] - (M[oth, c][:, None] * M[r][None, :]) % p) % p
        r += 1
        if r == M.shape[0]: break
    return r
print(f'triple-contraction cubics alone: rank {rank_mod(np.array(rows))} of {len(mon3)} cubic monomials', flush=True)
