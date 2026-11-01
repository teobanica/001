#!/usr/bin/env python3
# twists.py -- cocycle twists of S_N and of its subgroups as seeds, for "On the maximality conjecture for quantum
# permutation groups" (Claude and Teo), Sections 4.3-4.4 (Propositions 4.4, 4.5, 4.8, and Remark 4.7).
#
#   python3 twists.py cyclic      Proposition 4.4: A = Z_m^n on n disjoint m-cycles, all cocycle classes
#   python3 twists.py mixed       Proposition 4.5: non-cyclic and mixed orbits (Klein, Z_4, ...)
#   python3 twists.py sub         Proposition 4.8: twists L^sigma of L = A x| <g>, plus the classical controls of Remark 4.7
#   python3 twists.py all         everything (about half an hour on a laptop)
#
# Setting (Proposition 4.3).  A = H_1 x ... x H_r, each H_q acting regularly on its own orbit, and a bicharacter
# coupling different orbits: Theta(a) = prod_{s<t, q_s<q_t} beta_{q_s q_t}(h_s, h_t) on the Fourier coordinates
# a = ((q_s, h_s))_s.  The partition vector xi_pi has coefficient prod_b |H_{q_b}|^{1-|b|} on the tuples balanced for
# pi (q constant and labels summing to 0 on each block), and Fix_{S_N^sigma}(k) = Theta(V), V = span{xi_pi}.  So
#     dim Fix_{<S_N, S_N^sigma>}(k) = 2 dim V - rank [V | Theta V],
# and, for L = A x| <g> with g normalizing A (monomial on the Fourier coordinates),
#     dim Fix_{<S_N, L^sigma>}(k)  = dim V - rank (g - 1) Theta^{-1} V.
# Ranks are computed exactly modulo the prime p = 1048261 = 1 mod 60 (so that the roots of unity of order
# 2, 3, 4, 5 exist), after a random compression; the rank can only drop, so the output is an upper bound, and a
# Catalan output is a proof (the noncrossing vectors are always there).
import sys, itertools, time
import numpy as np

P = 1048261

def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))
assert is_prime(P) and P % 60 == 1

def factor(n):
    f, d = set(), 2
    while d * d <= n:
        while n % d == 0: f.add(d); n //= d
        d += 1
    if n > 1: f.add(n)
    return f

GEN = next(g for g in range(2, P) if all(pow(g, (P - 1) // q, P) != 1 for q in factor(P - 1)))
def omega(m): return pow(GEN, (P - 1) // m, P)

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

def stirling2(k, j):
    S = [[0] * (j + 1) for _ in range(k + 1)]; S[0][0] = 1
    for n in range(1, k + 1):
        for m in range(1, j + 1): S[n][m] = m * S[n - 1][m] + S[n - 1][m - 1]
    return S[k][j]

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

# ----------------------------------------------------------------------------------------- the setting
class Twist:
    """orbits: list of tuples of cyclic orders, e.g. [(2,), (2, 2), (4,)]; coup: dict (q, q') -> integer matrix B with
    beta_{qq'}(h, h') = exp(2 pi i * sum_{x,y} B[x][y] h_x h'_y / M), M = lcm of the orders involved, for q < q'."""
    def __init__(self, orbits, coup, M):
        self.orb = orbits; self.r = len(orbits); self.coup = coup; self.M = M; self.w = omega(M)
        self.H = [list(itertools.product(*[range(n) for n in o])) for o in orbits]
        self.N = sum(len(h) for h in self.H)
    def beta(self, q, h, q2, h2):
        if q >= q2 or (q, q2) not in self.coup: return 0
        B = self.coup[(q, q2)]
        return sum(B[x][y] * h[x] * h2[y] for x in range(len(h)) for y in range(len(h2))) % self.M

def coordinates(tw, qs):
    """All label tuples on the orbit sequence qs with zero sum on every orbit."""
    k = len(qs); per = {}
    for q in set(qs): per[q] = [l for l in range(k) if qs[l] == q]
    lists = []
    for q, legs in per.items():
        o = tw.orb[q]; t = len(legs); opts = []
        for labs in itertools.product(tw.H[q], repeat=t):
            if all(sum(lab[x] for lab in labs) % o[x] == 0 for x in range(len(o))): opts.append(labs)
        lists.append((legs, opts))
    out = []
    for combo in itertools.product(*[o for _, o in lists]):
        lab = [None] * k
        for (legs, _), labs in zip(lists, combo):
            for l, x in zip(legs, labs): lab[l] = x
        out.append(tuple(lab))
    return out

class Level:
    def __init__(self, k):
        self.k = k; self.parts = set_partitions(k); self.B = len(self.parts)
        self.maxb = max(len(q) for q in self.parts)
        self.Bm = np.zeros((self.B, self.maxb), dtype=np.int64)
        for i, q in enumerate(self.parts):
            for j, b in enumerate(q): self.Bm[i, j] = sum(1 << l for l in b)
        self.pc = np.array([bin(x).count('1') for x in range(1 << k)])
        self.ncat = sum(1 for q in self.parts if noncrossing(q))

def columns(tw, lev, qs, tuples):
    """Matrix (tuples x partitions) of the coefficients of the xi_pi (mod P)."""
    k = lev.k; T = len(tuples)
    good = np.zeros((T, 1 << k), dtype=np.int64); good[:, 0] = 1
    inv = {q: pow(len(tw.H[q]), P - 2, P) for q in set(qs)}
    dmax = max(len(o) for o in tw.orb)
    arr = np.zeros((T, k, dmax), dtype=np.int64)
    for i, t in enumerate(tuples):
        for l in range(k): arr[i, l, :len(t[l])] = t[l]
    within = []
    for q in set(qs):
        L = [l for l in range(k) if qs[l] == q]
        for r in range(1, len(L) + 1):
            for S in itertools.combinations(L, r): within.append(list(S))
    for legs in within:
        m = sum(1 << l for l in legs)
        q = qs[legs[0]]; o = np.array(list(tw.orb[q]) + [1] * (dmax - len(tw.orb[q])))
        ok = ((arr[:, legs, :].sum(axis=1) % o) == 0).all(axis=1)
        good[ok, m] = pow(inv[q], len(legs) - 1, P)
    vals = np.ones((T, lev.B), dtype=np.int64)
    for j in range(lev.maxb): vals = (vals * good[:, lev.Bm[:, j]]) % P
    return vals.astype(np.float64)

def theta(tw, qs, tuples):
    k = len(qs)
    e = np.array([sum(tw.beta(qs[s], t[s], qs[u], t[u]) for s in range(k) for u in range(s + 1, k)) % tw.M for t in tuples])
    return np.array([pow(tw.w, int(x), P) for x in e], dtype=np.float64)

def joint_twist(tw, k, seed=1):
    t0 = time.time(); lev = Level(k)
    dimspan = sum(stirling2(k, j) for j in range(1, min(k, tw.N) + 1))
    R_ = 2 * dimspan - lev.ncat + 20; rng = np.random.default_rng(seed); M = np.zeros((R_, 2 * lev.B)); buf = []; nbuf = 0
    for qs in itertools.product(range(tw.r), repeat=k):
        tup = coordinates(tw, qs)
        if not tup: continue
        C = columns(tw, lev, qs, tup); Th = theta(tw, qs, tup)
        buf.append(np.hstack([C, np.mod(Th[:, None] * C, P)])); nbuf += len(tup)
        if nbuf >= 4096: M = flush(M, buf, R_, rng); buf = []; nbuf = 0
    if buf: M = flush(M, buf, R_, rng)
    return 2 * dimspan - rank_mod_p(M), time.time() - t0

def flush(M, buf, R_, rng):
    X = np.vstack(buf); Rq = rng.integers(0, 1 << 12, size=(R_, X.shape[0])).astype(np.float64)
    return np.mod(M + Rq @ X, P)

def joint_sub(tw, k, g, seed=1):
    """g = (orbit permutation as a tuple, label map: function (q, h) -> h'), normalizing A and preserving the
    Fourier labelings; Fix_{L^sigma} = Theta(Fix_L), L = A x| <g>."""
    t0 = time.time(); lev = Level(k); perm, lab = g
    dimspan = sum(stirling2(k, j) for j in range(1, min(k, tw.N) + 1))
    R_ = dimspan - lev.ncat + 20; rng = np.random.default_rng(seed); M = np.zeros((R_, lev.B)); buf = []; nbuf = 0
    for qs in itertools.product(range(tw.r), repeat=k):
        tup = coordinates(tw, qs)
        if not tup: continue
        qs2 = tuple(perm[q] for q in qs)
        tup2 = [tuple(lab(qs[l], t[l]) for l in range(k)) for t in tup]
        D1 = np.mod(columns(tw, lev, qs, tup) * np.array([pow(int(x), P - 2, P) for x in theta(tw, qs, tup)])[:, None], P)
        D2 = np.mod(columns(tw, lev, qs2, tup2) * np.array([pow(int(x), P - 2, P) for x in theta(tw, qs2, tup2)])[:, None], P)
        buf.append(np.mod(D2 - D1, P)); nbuf += len(tup)
        if nbuf >= 4096: M = flush(M, buf, R_, rng); buf = []; nbuf = 0
    if buf: M = flush(M, buf, R_, rng)
    return dimspan - rank_mod_p(M), time.time() - t0

# ----------------------------------------------------------------------------------------- the cases
def sign_graph_classes(n):
    """Graphs on n vertices up to isomorphism (non-empty), as edge sets."""
    pairs = list(itertools.combinations(range(n), 2)); seen = set(); out = []
    for bits in range(1, 1 << len(pairs)):
        E = [pairs[i] for i in range(len(pairs)) if bits >> i & 1]
        can = min(tuple(sorted(tuple(sorted((p[a], p[b]))) for a, b in E)) for p in itertools.permutations(range(n)))
        if can not in seen: seen.add(can); out.append(E)
    return out

def z3_classes(n):
    """Alternating forms on Z_3^n (strictly upper b with entries in Z_3), nonzero, up to orbit permutations."""
    pairs = list(itertools.combinations(range(n), 2)); seen = set(); out = []
    for vals in itertools.product(range(3), repeat=len(pairs)):
        if not any(vals): continue
        b = dict(zip(pairs, vals)); best = None
        for p in itertools.permutations(range(n)):
            for eps in [(1,) * n]:      # orbit permutations only: a superset of the classes up to all symmetries
                img = {}
                for (i, j), v in b.items():
                    a, c = p[i], p[j]; v2 = v * eps[i] * eps[j] % 3
                    if a > c: a, c, v2 = c, a, (-v2) % 3
                    img[(a, c)] = v2
                key = tuple(img.get(pr, 0) for pr in pairs)
                best = key if best is None or key < best else best
        if best not in seen: seen.add(best); out.append(dict(zip(pairs, best)))
    return out

def cyc(n, m, b):
    return Twist([(m,)] * n, {pr: [[v]] for pr, v in b.items() if v}, m)

def cases_cyclic():
    C = [('N=4, Z2^2, S_4^tau', cyc(2, 2, {(0, 1): 1}), range(4, 7))]
    for E in sign_graph_classes(3): C.append((f'N=6, Z2^3, sign graph {E}', cyc(3, 2, {e: 1 for e in E}), range(4, 8)))
    for E in sign_graph_classes(4): C.append((f'N=8, Z2^4, sign graph {E}', cyc(4, 2, {e: 1 for e in E}), range(4, 8)))
    C.append(('N=6, Z3^2, b=1', cyc(2, 3, {(0, 1): 1}), range(4, 8)))
    for bb in (1, 2): C.append((f'N=8, Z4^2, b={bb}', cyc(2, 4, {(0, 1): bb}), range(4, 8)))
    for b in z3_classes(3): C.append((f'N=9, Z3^3, b={b}', cyc(3, 3, b), range(4, 7)))
    C.append(('N=10, Z5^2, b=1', cyc(2, 5, {(0, 1): 1}), range(4, 7)))
    return C

def cases_mixed():
    K = (2, 2)
    return [('N=6, Klein + Z2', Twist([K, (2,)], {(0, 1): [[1], [0]]}, 2), range(4, 8)),
            ('N=6, Z4 + Z2', Twist([(4,), (2,)], {(0, 1): [[2]]}, 4), range(4, 8)),
            ('N=8, Klein + Klein, rank 1', Twist([K, K], {(0, 1): [[1, 0], [0, 0]]}, 2), range(4, 8)),
            ('N=8, Klein + Klein, rank 2', Twist([K, K], {(0, 1): [[1, 0], [0, 1]]}, 2), range(4, 8)),
            ('N=8, Klein + Z4', Twist([K, (4,)], {(0, 1): [[2], [0]]}, 4), range(4, 8)),
            ('N=8, Klein + Z2 + Z2, chain', Twist([K, (2,), (2,)], {(0, 1): [[1], [0]], (1, 2): [[1]]}, 2), range(4, 8)),
            ('N=8, Klein + Z2 + Z2, star', Twist([K, (2,), (2,)], {(0, 1): [[1], [0]], (0, 2): [[0], [1]]}, 2), range(4, 8))]

def swap(n, i, j):
    p = list(range(n)); p[i], p[j] = j, i; return (tuple(p), lambda q, h: h)

def cases_sub():
    neg = lambda m, q0: (tuple(range(2)), lambda q, h: tuple((-x) % m for x in h) if q == q0 else h)
    return [('N=6, Z2^3, edge {1,2}, swap 2,3', cyc(3, 2, {(0, 1): 1}), swap(3, 1, 2), 'Catalan'),
            ('N=6, Z2^3, path 1-2-3, swap 1,2', cyc(3, 2, {(0, 1): 1, (1, 2): 1}), swap(3, 0, 1), 'Catalan'),
            ('N=6, Z3^2, b=1, swap', cyc(2, 3, {(0, 1): 1}), swap(2, 0, 1), 'Catalan'),
            ('N=6, Z3^2, b=1, a -> -a on one orbit', cyc(2, 3, {(0, 1): 1}), neg(3, 1), 'Catalan'),
            ('N=8, Z2^4, edge {1,2}, swap 2,3', cyc(4, 2, {(0, 1): 1}), swap(4, 1, 2), 'Catalan'),
            ('N=8, Z2^4, edges {1,2},{3,4}, swap 2,3', cyc(4, 2, {(0, 1): 1, (2, 3): 1}), swap(4, 1, 2), 'Catalan'),
            ('N=8, Z2^4, path 1-2-3-4, swap 1,2', cyc(4, 2, {(0, 1): 1, (1, 2): 1, (2, 3): 1}), swap(4, 0, 1), 'Catalan'),
            ('N=8, Z4^2, b=1, swap', cyc(2, 4, {(0, 1): 1}), swap(2, 0, 1), 'Catalan'),
            # classical controls (Remark 4.7): the twist has no effect, Bell numbers expected
            ('control: N=4, D4 = Z2^2 x| swap', cyc(2, 2, {(0, 1): 1}), swap(2, 0, 1), 'Bell'),
            ('control: N=6, triangle, swap 1,2', cyc(3, 2, {(0, 1): 1, (1, 2): 1, (0, 2): 1}), swap(3, 0, 1), 'Bell')]

CAT = {4: 14, 5: 42, 6: 132, 7: 429}
BELL = {4: 15, 5: 52, 6: 203, 7: 877}

def run(which, kmax=7):
    if which in ('cyclic', 'mixed'):
        cases = cases_cyclic() if which == 'cyclic' else cases_mixed()
        for name, tw, ks in cases:
            res = []
            for k in ks:
                if k > kmax: continue
                d, s = joint_twist(tw, k); res.append(d)
                print(f'  {name}: k={k} dim {d} (Catalan {CAT[k]})  [{s:.0f} s]', flush=True)
    else:
        for name, tw, g, exp in cases_sub():
            for k in range(4, kmax + 1):
                d, s = joint_sub(tw, k, g)
                ref = CAT[k] if exp == 'Catalan' else BELL[k] if tw.N >= k else None
                print(f'  {name}: k={k} dim {d} (expected {exp}{" " + str(ref) if ref else ""})  [{s:.0f} s]', flush=True)

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'cyclic'
    kmax = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    for w in (['cyclic', 'mixed', 'sub'] if what == 'all' else [what]): print(w, flush=True); run(w, kmax)
