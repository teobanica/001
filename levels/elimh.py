# Exact elimination for the level-6 quadrics (task 1a), pure Python over Q.
# Input : quad_N{N}.pkl from levquadrics.py (exact Fractions).
# Method: Macaulay matrices -> Hilbert function H(d) of R/I, R = Q[c0..c4];
#         Gotzmann persistence certifies the Hilbert polynomial;
#         multiplication matrices on (R/I)_d give an exact characteristic
#         polynomial whose squarefreeness certifies distinct, simple points.
import sys, pickle, itertools, random
from fractions import Fraction as Fr
from math import comb

N = int(sys.argv[1]); dmax = int(sys.argv[2]) if len(sys.argv) > 2 else 6
Q, sym, vecs = pickle.load(open(f'quad_N{N}.pkl', 'rb'))
n = 5
def unit(i): return tuple(1 if k == i else 0 for k in range(n))
def add(a, b): return tuple(x + y for x, y in zip(a, b))
F = []
for row in Q:
    p = {}
    for x, (a, b) in zip(row, sym):
        if x != 0:
            m = add(unit(a), unit(b)); p[m] = p.get(m, 0) + Fr(x)
    F.append(p)
def monos(d):
    return sorted((m for m in itertools.product(range(d + 1), repeat=n) if sum(m) == d), reverse=True)

def rref(rows, cols):
    """rows: list of dicts mono->Fr. Returns pivot dict col->row (row normalised, fully reduced)."""
    order = {m: i for i, m in enumerate(cols)}
    piv = {}
    for r in rows:
        r = dict(r)
        # reduce by existing pivots
        changed = True
        while True:
            hits = [m for m in r if m in piv]
            if not hits: break
            m = min(hits, key=order.get); f = r[m]
            for k, v in piv[m].items():
                r[k] = r.get(k, 0) - f * v
                if r[k] == 0: del r[k]
        if not r: continue
        m = min(r, key=order.get); f = r[m]
        r = {k: v / f for k, v in r.items()}
        # back-reduce older pivots
        for pm, pr in piv.items():
            if m in pr:
                g = pr[m]
                for k, v in r.items():
                    pr[k] = pr.get(k, 0) - g * v
                    if pr[k] == 0: del pr[k]
        piv[m] = r
    return piv

def ideal_part(d):
    rows = [{add(m, k): v for k, v in f.items()} for f in F for m in monos(d - 2)]
    return rref(rows, monos(d))

H = {}; P = {}
for d in range(2, dmax + 1):
    P[d] = ideal_part(d); H[d] = comb(d + 4, 4) - len(P[d])
    print(f'N={N}: H({d}) = {H[d]}   (dim R_d = {comb(d+4,4)}, dim I_d = {len(P[d])})', flush=True)

h = H[dmax]; d0 = next((d for d in range(max(4, h), dmax) if H[d] == h and H[d + 1] == h), None)
if d0 is None:
    print('  Hilbert function not yet stable at 4; increase dmax'); sys.exit()
print(f'  Gotzmann: H({d0}) = H({d0+1}) = 4 with 4^<{d0}> = 4  =>  Hilbert polynomial = 4:'
      f' the zero set in P^4 is finite, of degree exactly 4 (with multiplicity).', flush=True)

def NF(p, d):
    p = dict(p)
    for m in [m for m in p if m in P[d]]:
        f = p.pop(m)
        if f == 0: continue
        for k, v in P[d][m].items():
            if k == m: continue
            p[k] = p.get(k, 0) - f * v
    return {k: v for k, v in p.items() if v != 0}
B0 = [m for m in monos(d0) if m not in P[d0]]
B1 = [m for m in monos(d0 + 1) if m not in P[d0 + 1]]
print('  standard monomials, degree', d0, ':', B0, flush=True)
def mult_matrix(lin):           # lin: list of 5 Fr
    A = [[Fr(0)] * h for _ in range(h)]
    for j, m in enumerate(B0):
        p = {}
        for i, a in enumerate(lin):
            if a: k = add(m, unit(i)); p[k] = p.get(k, 0) + a
        nf = NF(p, d0 + 1)
        for i, b in enumerate(B1): A[i][j] = nf.get(b, Fr(0))
    return A
def matmul(A, B): return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
def inv(A):
    m = len(A); M = [list(r) + [Fr(int(i == j)) for j in range(m)] for i, r in enumerate(A)]
    for c in range(m):
        p = next(r for r in range(c, m) if M[r][c] != 0); M[c], M[p] = M[p], M[c]
        v = M[c][c]; M[c] = [x / v for x in M[c]]
        for r in range(m):
            if r != c and M[r][c] != 0:
                f = M[r][c]; M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [r[m:] for r in M]
def det(A):
    m = len(A); M = [list(r) for r in A]; D = Fr(1)
    for c in range(m):
        p = next((r for r in range(c, m) if M[r][c] != 0), None)
        if p is None: return Fr(0)
        if p != c: M[c], M[p] = M[p], M[c]; D = -D
        D *= M[c][c]
        for r in range(c + 1, m):
            f = M[r][c] / M[c][c]; M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return D
def charpoly(T):                # Faddeev-LeVerrier, returns coeffs [1, a1, ..., a4]
    m = len(T); I = [[Fr(int(i == j)) for j in range(m)] for i in range(m)]
    Mk = [[Fr(0)] * m for _ in range(m)]; cs = [Fr(1)]
    for k in range(1, m + 1):
        Mk = [[x + (cs[-1] if i == j else 0) for j, x in enumerate(r)] for i, r in enumerate(matmul(T, Mk))]
        TM = matmul(T, Mk); cs.append(-sum(TM[i][i] for i in range(m)) / k)
    return cs
random.seed(N)
ell = [Fr(random.randint(1, 9)) for _ in range(n)]
Al = mult_matrix(ell); dl = det(Al)
print(f'  generic linear form ell = {[int(x) for x in ell]}: det(mult by ell) = {dl}  ->',
      'non-zero-divisor, no zeros on ell = 0' if dl != 0 else 'ZERO, pick another', flush=True)
Ali = inv(Al)
T = [matmul(Ali, mult_matrix([Fr(int(i == k)) for k in range(n)])) for i in range(n)]
comm = all(matmul(T[i], T[j]) == matmul(T[j], T[i]) for i in range(n) for j in range(n))
print('  multiplication operators commute:', comm, flush=True)
g = [Fr(random.randint(-9, 9)) for _ in range(n)]
Tg = [[sum(g[i] * T[i][r][s] for i in range(n)) for s in range(h)] for r in range(h)]
cp = charpoly(Tg)
# squarefree test via discriminant of quartic: gcd(p, p') over Q
def pdiv(a, b):
    a = list(a)
    while len(a) >= len(b) and any(a):
        f = a[0] / b[0]
        for i in range(len(b)): a[i] -= f * b[i]
        a.pop(0)
    while a and a[0] == 0: a.pop(0)
    return a
def pgcd(a, b):
    while b: a, b = b, pdiv(a, b)
    return a
dp = [cp[i] * (h - i) for i in range(h)]
gd = pgcd(cp, dp)
print('  char poly of generic coordinate:', [str(x) for x in cp], flush=True)
print('  squarefree (gcd with derivative is constant):', len(gd) == 1, flush=True)
import numpy as np
roots = np.roots([float(x) for x in cp])
print('  its roots:', np.round(roots, 6), flush=True)
# factor over Q: test rational roots of the monic quartic via all pairings of numeric roots
def polymul(a, b):
    r = [Fr(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b): r[i + j] += x * y
    return r
facts = []
for k in (1, 2):
    for S in itertools.combinations(range(4), k):
        q = np.poly(roots[list(S)])
        if np.max(np.abs(q.imag)) > 1e-8: continue
        qq = [Fr(float(x)).limit_denominator(10 ** 12) for x in q.real]
        if not pdiv(cp, qq): facts.append((S, [str(x) for x in qq]))
print('  factors over Q found (root subsets, coefficients):', facts if facts else 'none: irreducible over Q', flush=True)
# exact coordinates of each point (c0:...:c4) via eigenvectors of the transposed operators, numerically, for comparison
ev, V = np.linalg.eig(np.array([[float(x) for x in r] for r in Tg]).T)
pts = []
for k in range(h):
    w = V[:, k]                       # left eigenvector of Tg: w T_i = (c_i/ell)(p) w
    coords = np.array([(w @ np.array([[float(x) for x in r] for r in T[i]]) @ w.conj()) / (w @ w.conj()) for i in range(n)])
    coords = coords / np.linalg.norm(coords); j = np.argmax(abs(coords)); coords *= abs(coords[j]) / coords[j]
    pts.append(coords)
    print(f'  point {k}: real={np.max(abs(coords.imag)) < 1e-9}', np.round(coords, 5), flush=True)
pickle.dump({'H': H, 'd0': d0, 'charpoly': [str(x) for x in cp], 'ell': ell, 'g': g,
             'T': [[[str(x) for x in r] for r in t] for t in T]}, open(f'elim_N{N}.pkl', 'wb'))
