#!/usr/bin/env python3
# projten.py -- projective level 10, half-liberated case, exact: the merged block ker(rho-1) + ker(rho+1) of U(10),
# the paper's certificate c(N) of Proposition 11.x(3) evaluated on the whole merged block.  Uses pairlevels.py.
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction as Fr
import numpy as np
import pairlevels as pl
out = pl.out
k = 10; P = pl.pairings(k, 'HL'); idx = {p: a for a, p in enumerate(P)}
P8 = pl.pairings(8, 'HL'); cr8 = [q for q in P8 if not pl.noncrossing(q)]; P6 = pl.pairings(6, 'HL'); kap = pl.canon(((0, 3), (1, 4), (2, 5)))
out('=== projective level 10, half-liberated: merged block rho = +-1, exact ===')
out('crossing pairings of P_2^*(8), in the order used: ' + '; '.join(str(q) for q in cr8))
def cN(N): return [-3*N*N, -3*N*N, 30*N, 20*N, -3*N*N, 20*N, -3*N*N, 30, -18*N, -18*N, -18]
R = [[Fr(0)] * len(P) for _ in P]
for a, p in enumerate(P): R[idx[pl.rot(p, k)]][a] = Fr(1)
def apply(M, v): return [sum(M[i][j] * v[j] for j in range(len(v)) if v[j] != 0) for i in range(len(M))]
def J(x):
    y = [Fr(0)] * len(P)
    for a, p in enumerate(P): y[idx[pl.rev(p, k)]] = x[a]
    return y
def eig_block(null, s):
    RK = [apply(R, b) for b in null]
    A = [[RK[b][i] - s * null[b][i] for b in range(len(null))] for i in range(len(P))]
    cs, _ = pl.nullspace(A)
    return [[sum(c[b] * null[b][i] for b in range(len(null))) for i in range(len(P))] for c in cs]
def forms(basis, N):
    """11 exact symmetric matrices Q_k on the given real basis: coeff_{tau_k}(x o_6 Jy) for the 10 level-8 crossings,
    then coeff_kappa(x o_7 Jy); symmetrized."""
    d = len(basis); Qs = [[[Fr(0)] * d for _ in range(d)] for _ in range(11)]
    Jb = [J(b) for b in basis]
    for i in range(d):
        for j in range(d):
            g8 = pl.glue_vector(basis[i], Jb[j], P, k, 6, N, P8)
            g6 = pl.glue_vector(basis[i], Jb[j], P, k, 7, N, P6)
            for t, tau in enumerate(cr8): Qs[t][i][j] = g8[P8.index(tau)]
            Qs[10][i][j] = g6[P6.index(kap)]
    return [[[ (Q[i][j] + Q[j][i]) / 2 for j in range(d)] for i in range(d)] for Q in Qs]
def ldl_positive(A):
    n = len(A); L = [row[:] for row in A]
    for i in range(n):
        for j in range(i): L[i][i] -= L[i][j] ** 2 * L[j][j]
        if L[i][i] <= 0: return False
        for r in range(i + 1, n):
            for j in range(i): L[r][i] -= L[r][j] * L[i][j] * L[j][j]
            L[r][i] /= L[i][i]
    return True
def minors(A):
    n = len(A); res = []
    for s in range(1, n + 1):
        M = [row[:s] for row in A[:s]]
        # determinant by Gaussian elimination
        det = Fr(1); M = [r[:] for r in M]
        for c in range(s):
            p = next((i for i in range(c, s) if M[i][c] != 0), None)
            if p is None: det = Fr(0); break
            if p != c: M[c], M[p] = M[p], M[c]; det = -det
            det *= M[c][c]
            for i in range(c + 1, s):
                f = M[i][c] / M[c][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[c])]
        res.append(det)
    return res
for N in list(range(5, 13)) + [16, 20]:
    t0 = time.time()
    M, _ = pl.cap_matrix(P, k, N, 'HL'); null, rank = pl.nullspace(M)
    B1 = eig_block(null, Fr(1)); Bm = eig_block(null, Fr(-1))
    basis = B1 + Bm
    Qs = forms(basis, N)
    c = cN(N)
    A = [[sum(c[t] * Qs[t][i][j] for t in range(11)) for j in range(len(basis))] for i in range(len(basis))]
    d1 = len(B1)
    cross = max(abs(A[i][j]) for i in range(d1) for j in range(d1, len(basis)))
    val_m = A[d1][d1] if len(Bm) == 1 else None
    pos1 = ldl_positive([row[:d1] for row in A[:d1]])
    posall = ldl_positive(A)
    out(f'N={N}: dim U = {len(null)}, rho=1 block dim {d1}, rho=-1 block dim {len(Bm)}; with the paper c(N): PD on rho=1: {pos1}; '
        f'cross terms max {cross}; value on the rho=-1 vector: {val_m}; PD on the merged block: {posall}  ({time.time()-t0:.0f}s)')
    if N in (5, 6, 7, 8):
        out('   leading principal minors on the merged block: ' + ', '.join(str(m) for m in minors(A)))
    # the individual values Q_k(x_-) for the record (x_- normalized by its first nonzero coordinate)
    if len(Bm) == 1:
        out('   Q_k(x_-), k = 0..10: ' + ', '.join(str(Qs[t][d1][d1]) for t in range(11)))
