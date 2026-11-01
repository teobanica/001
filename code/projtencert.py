#!/usr/bin/env python3
# projtencert.py -- projective level 10, half-liberated case, exact for all N >= 5: the merged block
# ker(rho - 1) + ker(rho + 1) of U(10) (dimension 4 + 1, merged since W is only rho^2-invariant), Subsection 12.4 of
# "On the maximality conjecture for quantum permutation groups" (Claude and Teo).  Uses pairlevels.py (same folder).
#
#   python3 projtencert.py        about 1 minute; log in projtencertout.txt
#
# 1. Basis with entries in Z[N]: ker cap_0 on the 17 rotation-orbit sums (rho = 1) and on the 11 signed orbit sums of the
#    even orbits (rho = -1); for a rho-eigenvector, cap_0 x = 0 implies cap_i x = 0 for all i.  The kernel vectors are the
#    adjugate (Cramer) vectors of a pivot minor D(N), interpolated exactly and divided by their polynomial content.
#    All cap identities are checked as identities of polynomials.  Dimensions 4 and 1 hold for every N with D(N) != 0.
# 2. The 11 real forms Q_k(N) on the merged block: 2 Sym coeff_tau(x o_6 Jy) for the 10 crossing pairings tau of
#    P_2^*(8), in the order of pairlevels.pairings, then 2 Sym coeff_kappa(x o_7 Jy); integer polynomial matrices.
#    Only even rotations are allowed in the projective case, and rho^2 is the identity on the merged block, so these
#    unrotated forms are all the forms of the real-parts lemma.
# 3. Certificate: c(N) of Proposition 11.x(3) (the rho = 1 certificate of Section 11), unchanged.  The paper lists the
#    forms in another order: paper order = code order with the forms 2, 3 swapped and 6, 7 swapped.  The 5 leading
#    principal minors of A(N) = sum_k c_k(N) Q_k(N) are computed as exact polynomials (Newton interpolation beyond the
#    degree bound, checked at further points), and after N -> N + 7 all their coefficients are positive (Descartes):
#    A(N) is positive definite for all real N >= 7.  For N = 5, 6, exact integer certificates (LDL^t, exact).
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction as Fr
from math import gcd
import pairlevels as pl

log = open('projtencertout.txt', 'a')
def out(s):
    print(s, flush=True); log.write(s + '\n'); log.flush()

# ---------------------------------------------------------------- polynomials (coefficient lists, low -> high)
def ptrim(a):
    a = list(a)
    while a and a[-1] == 0: a.pop()
    return a
def padd(a, b):
    m = max(len(a), len(b)); return ptrim([(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(m)])
def pmul(a, b):
    if not a or not b: return []
    r = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b): r[i + j] += x * y
    return ptrim(r)
def pscale(a, c): return ptrim([x * c for x in a])
def pdivmod(a, b):
    a = [Fr(x) for x in ptrim(a)]; b = ptrim(b); q = [Fr(0)] * max(0, len(a) - len(b) + 1)
    while a and len(a) >= len(b):
        c = a[-1] / b[-1]; d = len(a) - len(b); q[d] = c
        for i, y in enumerate(b): a[i + d] -= c * y
        a = ptrim(a)
    return ptrim(q), a
def pgcd(a, b):
    a, b = ptrim(a), ptrim(b)
    while b: a, b = b, pdivmod(a, b)[1]
    return [Fr(x) / a[-1] for x in a]
def peval(a, x):
    r = 0
    for c in reversed(a): r = r * x + c
    return r
def interp(xs, ys):
    m = len(xs); co = [Fr(y) for y in ys]
    for j in range(1, m):
        for i in range(m - 1, j - 1, -1): co[i] = (co[i] - co[i - 1]) / (xs[i] - xs[i - j])
    p = []
    for i in range(m - 1, -1, -1): p = padd(pmul(p, [-xs[i], 1]), [co[i]])
    return p
def fitpoly(f, deg, start=5, extra=8):
    xs = list(range(start, start + deg + 1)); p = interp(xs, [f(x) for x in xs])
    for x in range(start + deg + 1, start + deg + 1 + extra):
        assert peval(p, x) == f(x), 'degree bound violated'
    return p
def shift(a, s):
    """coefficients of a(N + s)"""
    r = []
    for c in reversed(a): r = padd(pmul(r, [s, 1]), [c])
    return r
def primitive(vec):
    den = 1
    for p in vec:
        for c in p: den = den * Fr(c).denominator // gcd(den, Fr(c).denominator)
    v = [[int(Fr(c) * den) for c in p] for p in vec]; g = 0
    for p in v:
        for c in p: g = gcd(g, c)
    return [[c // g for c in p] for p in v]
def pstr(a): return pl.poly_str([int(x) if Fr(x).denominator == 1 else x for x in a])

# ---------------------------------------------------------------- exact linear algebra over Q
def det(M):
    M = [r[:] for r in M]; s = len(M); d = Fr(1)
    for c in range(s):
        p = next((i for i in range(c, s) if M[i][c] != 0), None)
        if p is None: return Fr(0)
        if p != c: M[c], M[p] = M[p], M[c]; d = -d
        d *= M[c][c]
        for i in range(c + 1, s):
            f = M[i][c] / M[c][c]
            if f: M[i] = [a - f * b for a, b in zip(M[i], M[c])]
    return d
def solve(A, b):
    s = len(A); M = [A[i][:] + [b[i]] for i in range(s)]
    for c in range(s):
        p = next(i for i in range(c, s) if M[i][c] != 0); M[c], M[p] = M[p], M[c]
        M[c] = [x / M[c][c] for x in M[c]]
        for i in range(s):
            if i != c and M[i][c] != 0: f = M[i][c]; M[i] = [x - f * y for x, y in zip(M[i], M[c])]
    return [M[i][s] for i in range(s)]
def pivot_rows_cols(M):
    rows = []; red = []
    for i, r in enumerate(M):
        v = [Fr(x) for x in r]
        for (pc, b) in red:
            if v[pc]: f = v[pc] / b[pc]; v = [x - f * y for x, y in zip(v, b)]
        nz = next((j for j in range(len(v)) if v[j] != 0), None)
        if nz is not None: red.append((nz, v)); rows.append(i)
    B = [[Fr(x) for x in M[i]] for i in rows]; cols = []; r = 0
    for c in range(len(M[0])):
        p = next((i for i in range(r, len(B)) if B[i][c] != 0), None)
        if p is None: continue
        B[r], B[p] = B[p], B[r]; B[r] = [x / B[r][c] for x in B[r]]
        for i in range(len(B)):
            if i != r and B[i][c] != 0: f = B[i][c]; B[i] = [x - f * y for x, y in zip(B[i], B[r])]
        cols.append(c); r += 1
    return rows, cols
def ldl_pivots(A):
    n = len(A); M = [[Fr(x) for x in r] for r in A]; piv = []
    for i in range(n):
        p = M[i][i]; piv.append(p)
        if p <= 0: return piv
        for r in range(i + 1, n):
            f = M[r][i] / p; M[r] = [a - f * b for a, b in zip(M[r], M[i])]
    return piv

t00 = time.time()
out(time.strftime('%d %b %H:%M') + '  projtencert.py: projective level 10, half-liberated, merged block rho = +-1, exact for all N >= 5')
k = 10; P = pl.pairings(k, 'HL'); idx = {p: a for a, p in enumerate(P)}
P8 = pl.pairings(8, 'HL'); il = {q: a for a, q in enumerate(P8)}; cr8 = [q for q in P8 if not pl.noncrossing(q)]
P6 = pl.pairings(6, 'HL'); kap = pl.canon(((0, 3), (1, 4), (2, 5)))
out(f'  |P_2^*(10)| = {len(P)}; crossing pairings of P_2^*(8), code order: ' + '; '.join(str(q) for q in cr8))

# ---------------------------------------------------------------- 1. polynomial basis
orbits = []; seen = set()
for p in P:
    if p in seen: continue
    orb = [p]; q = pl.rot(p, k)
    while q != p: orb.append(q); q = pl.rot(q, k)
    seen.update(orb); orbits.append(orb)
S1 = [{idx[q]: 1 for q in orb} for orb in orbits]
Sm = [{idx[q]: (-1) ** r for r, q in enumerate(orb)} for orb in orbits if len(orb) % 2 == 0]
out(f'  rotation orbits: {len(orbits)}, sizes {sorted(len(o) for o in orbits)}; rho = 1 sums: {len(S1)}, rho = -1 signed sums: {len(Sm)}')
cap0 = [(il[pl.cap(p, 0, k)[1]], pl.cap(p, 0, k)[0]) for p in P]
def capmat(S, N):
    M = [[Fr(0)] * len(S) for _ in P8]
    for c, vec in enumerate(S):
        for a, s in vec.items(): r, l = cap0[a]; M[r][c] += s * N ** l
    return M
def poly_kernel(S, name):
    rows, cols = pivot_rows_cols(capmat(S, 37)); free = [c for c in range(len(S)) if c not in cols]; r = len(rows)
    D = fitpoly(lambda N: det([[capmat(S, N)[i][j] for j in cols] for i in rows]), r)
    vecs = []
    for f in free:
        cache = {}
        def comp(N):
            if N not in cache:
                M = capmat(S, N); A = [[M[i][j] for j in cols] for i in rows]; d = det(A)
                sol = solve(A, [-M[i][f] for i in rows]); v = [Fr(0)] * len(S); v[f] = d
                for j, c in enumerate(cols): v[c] = d * sol[j]
                cache[N] = v
            return cache[N]
        vec = [fitpoly(lambda N, c=c: comp(N)[c], r) for c in range(len(S))]
        g = []
        for p in vec:
            if p: g = p if not g else pgcd(g, p)
        if len(g) > 1:
            assert all(not pdivmod(p, g)[1] for p in vec); vec = [pdivmod(p, g)[0] for p in vec]
        vecs.append(primitive(vec))
    # independence: the minor on the free coordinates is diagonal, with entries dividing D(N) (up to constants)
    for a, f in enumerate(free):
        for b, g in enumerate(free): assert (a == b) == bool(vecs[a][g]) and (a != b or not pdivmod(D, vecs[a][g])[1])
    Dsh = shift(D, 5)
    out(f'  {name}: rank of cap_0 on the sums = {r} (pivot minor D(N) = {pstr(D)}; D(N + 5) has coefficients of one sign: '
        f'{all(c > 0 for c in Dsh) or all(c < 0 for c in Dsh)}, so D(N) != 0 for N >= 5), kernel dimension {len(free)}, basis degrees {[max(len(p) - 1 for p in v) for v in vecs]}, independent wherever D(N) != 0')
    return vecs
V1 = poly_kernel(S1, 'rho = 1'); Vm = poly_kernel(Sm, 'rho = -1')
def expand(vecs, S):
    res = []
    for v in vecs:
        e = {}
        for c, p in enumerate(v):
            for a, s in S[c].items(): e[a] = padd(e.get(a, []), pscale(p, s))
        res.append({a: p for a, p in e.items() if p})
    return res
B = expand(V1, S1) + expand(Vm, Sm); d = len(B)
# identities in Z[N]: all ten caps vanish, rho acts by +-1
for b in B:
    for i in range(k):
        v = {}
        for a, p in b.items():
            l, q = pl.cap(P[a], i, k); v[il[q]] = padd(v.get(il[q], []), pmul(p, [0] * l + [1]))
        assert all(not p for p in v.values())
for s, bs in ((1, B[:len(V1)]), (-1, B[len(V1):])):
    for b in bs:
        rb = {idx[pl.rot(P[a], k)]: p for a, p in b.items()}
        assert all(padd(rb.get(a, []), pscale(b.get(a, []), -s)) == [] for a in set(rb) | set(b))
out(f'  merged block: dimension {d}; the ten caps vanish and rho acts by +1 (first {len(V1)}) and -1 (last {len(Vm)}), identically in N')

# ---------------------------------------------------------------- 2. the forms
tables = [[] for _ in range(11)]; t8 = {il[t]: s for s, t in enumerate(cr8)}
for i, p in enumerate(P):
    for j, q in enumerate(P):
        l, r = pl.glue(p, q, k, 6)
        if il[r] in t8: tables[t8[il[r]]].append((i, j, l))
        l, r = pl.glue(p, q, k, 7)
        if r == kap: tables[10].append((i, j, l))
JB = [{idx[pl.rev(P[a], k)]: p for a, p in b.items()} for b in B]
Q = []
for t in range(11):
    M = [[[] for _ in range(d)] for _ in range(d)]
    for (i, j, l) in tables[t]:
        for a in range(d):
            pa = B[a].get(i)
            if not pa: continue
            for b in range(d):
                pb = JB[b].get(j)
                if pb: M[a][b] = padd(M[a][b], pmul(pmul(pa, pb), [0] * l + [1]))
    Q.append([[padd(M[a][b], M[b][a]) for b in range(d)] for a in range(d)])
qdeg = max(len(Q[t][a][b]) - 1 for t in range(11) for a in range(d) for b in range(d))
out(f'  11 forms (10 crossings at level 8, then kappa): integer polynomial matrices of degree <= {qdeg}')

# ---------------------------------------------------------------- 3. certificate
paper = lambda N: [-3 * N * N, -3 * N * N, 30 * N, 20 * N, -3 * N * N, 20 * N, -3 * N * N, 30, -18 * N, -18 * N, -18]
perm = [0, 1, 3, 2, 4, 5, 7, 6, 8, 9, 10]            # code index -> paper index
cN = lambda N: [paper(N)[perm[t]] for t in range(11)]
out('  c(N) of Proposition 11.x(3), paper order: (-3N^2, -3N^2, 30N, 20N, -3N^2, 20N, -3N^2, 30, -18N, -18N, -18); '
    'code order: (' + ', '.join(pstr(fitpoly(lambda N, t=t: cN(N)[t], 2)) for t in range(11)) + ')')
cpoly = [fitpoly(lambda N, t=t: cN(N)[t], 2) for t in range(11)]
A = [[[] for _ in range(d)] for _ in range(d)]
for t in range(11):
    for a in range(d):
        for b in range(d): A[a][b] = padd(A[a][b], pmul(cpoly[t], Q[t][a][b]))
adeg = max(len(A[a][b]) - 1 for a in range(d) for b in range(d))
ok_all = True
for s in range(1, d + 1):
    bound = s * adeg
    m = fitpoly(lambda N: det([[peval(A[a][b], N) for b in range(s)] for a in range(s)]), bound, start=0, extra=8)
    sh = shift(m, 7); pos = all(c > 0 for c in sh) and len(sh) > 0
    neg = [x for x in sh if x <= 0]
    ok_all &= pos
    out(f'    leading minor {s}: degree {len(m) - 1}, leading coefficient {int(m[-1])}, all coefficients of m(N + 7) positive: {pos}'
        + ('' if pos else f'  ({len(neg)} nonpositive)'))
out(f'  => sum_k c_k(N) Q_k(N) is positive definite on the merged block for all real N >= 7: {ok_all}')
small = {5: [-5, -5, 10, 1, -5, 10, 6, -5, -3, -3, -2], 6: [-4, -4, 10, 1, -4, 10, 4, -4, -2, -2, -1]}
for N, c in small.items():
    AN = [[sum(c[t] * peval(Q[t][a][b], N) for t in range(11)) for b in range(d)] for a in range(d)]
    pv = ldl_pivots(AN); ok = all(p > 0 for p in pv) and len(pv) == d
    ok_all &= ok
    out(f'    N = {N}: c = {tuple(c)} (code order), exact LDL^t pivots all positive: {ok}')
out(f'  conclusion: the merged rho = +-1 block of U(10) is excluded for every N >= 5, exactly: {ok_all}   ({time.time() - t00:.0f}s)')
out('  (paper: Subsection 12.4, the table row "10 | 13 | 5,4,4 | all N >= 5 | excluded, exact"; the Phi_5 blocks keep the '
    'certificate of Proposition 11.x(4), even rotations sufficing there)')
