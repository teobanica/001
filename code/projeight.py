#!/usr/bin/env python3
# projeight.py -- projective level 8, half-liberated case, exact: the merged block span{x_+, x_-} of Theorem 11.x
# under rho^2, the real quadric on it, and its definiteness as a function of N.  Uses pairlevels.py (same folder).
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction as Fr
import pairlevels as pl
out = pl.out
k = 8; P = pl.pairings(k, 'HL'); idx = {p: a for a, p in enumerate(P)}
kap = pl.canon(((0, 3), (1, 4), (2, 5))); P6 = pl.pairings(6, 'HL'); ik = P6.index(kap)
R = [[Fr(0)] * len(P) for _ in P]
for a, p in enumerate(P): R[idx[pl.rot(p, k)]][a] = Fr(1)
def apply(Mat, v): return [sum(Mat[i][j] * v[j] for j in range(len(v))) for i in range(len(Mat))]
def J(x):
    y = [Fr(0)] * len(P)
    for a, p in enumerate(P): y[idx[pl.rev(p, k)]] = x[a]
    return y
out('=== projective level 8, half-liberated: the block span{x_+, x_-} under rho^2 ===')
rows = {'++': [], '--': [], '+-': [], '-+': [], 'det': []}; Ns = list(range(3, 21))
for N in Ns:
    M, _ = pl.cap_matrix(P, k, N, 'HL'); null, rank = pl.nullspace(M)
    xs = {}
    for sign, s in (('+', Fr(1)), ('-', Fr(-1))):
        RK = [apply(R, null[b]) for b in range(len(null))]
        A = [[RK[b][i] - s * null[b][i] for b in range(len(null))] for i in range(len(P))]
        cs, _ = pl.nullspace(A); x = [sum(cs[0][b] * null[b][i] for b in range(len(null))) for i in range(len(P))]
        x0 = x[idx[pl.canon(((0, 1), (2, 3), (4, 5), (6, 7)))]]; xs[sign] = [v / x0 for v in x]
    # J on x_+-: rev
    jsym = {s: ('J x = x' if J(xs[s]) == xs[s] else 'J x = -x' if J(xs[s]) == [-v for v in xs[s]] else 'other') for s in xs}
    h = {}
    for s in '+-':
        for t in '+-':
            h[s + t] = pl.glue_vector(xs[s], J(xs[t]), P, k, 5, N, P6)[ik]
    # real J-real vectors: alpha x_+ + beta x_- (both J-real), q(alpha, beta) = alpha^2 h++ + alpha beta (h+- + h-+) + beta^2 h--
    det = h['++'] * h['--'] - ((h['+-'] + h['-+']) / 2) ** 2
    for key in ('++', '--', '+-', '-+'): rows[key].append(h[key])
    rows['det'].append(det)
    if N <= 8 or N in (12, 16, 20):
        out(f'N={N}: dim U = {len(null)}, {jsym}, h++ = {h["++"]}, h-- = {h["--"]}, h+- = {h["+-"]}, h-+ = {h["-+"]}, '
            f'det = {det} ({"definite" if det > 0 else "indefinite"}), trace sign {"-" if h["++"] + h["--"] < 0 else "+"}')
# closed forms: (N-2)^2 (N+2)^2 h+- should be a polynomial
xs4 = [N for N in Ns if N >= 4]
for key, fac in (('+-', lambda N: Fr((N - 2) * (N + 2))), ('-+', lambda N: Fr((N - 2) * (N + 2)))):
    ys = [v * fac(N) for N, v in zip(Ns, rows[key]) if N >= 4]
    c = pl.poly_fit(xs4, ys, 10)
    out(f'(N-2)(N+2) h_{key}(N) = {pl.poly_str(c) if c else "no polynomial of degree <= 10"}')
ys = [v * Fr((N - 2) ** 2 * (N + 2) ** 2) for N, v in zip(Ns, rows['det']) if N >= 4]
c = pl.poly_fit(xs4, ys, 16)
out(f'(N-2)^2 (N+2)^2 det(N) = {pl.poly_str(c) if c else "no polynomial of degree <= 16"}')
