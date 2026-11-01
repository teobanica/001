#!/usr/bin/env python3
# numerical optimal certificates on the merged rho=+-1 block at level 10 (half-liberated), several N, to guess c(N)
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import minimize
import pairlevels as pl
exec(open("projten.py").read().split("for N in list(range(5, 13))")[0])
def best_cert(Qs, tries=20, seed=0):
    F = np.array([[[float(v) for v in row] for row in Q] for Q in Qs])
    # normalize the basis: Gram = identity already? no: use Cholesky of the identity in the given basis is not available;
    # we measure PD-ness via lambda_min relative to the spectral norm, basis fixed (same echelon basis for all N)
    m = len(F); rng = np.random.default_rng(seed); best = (-np.inf, None)
    def negf(y, beta):
        c = y / np.linalg.norm(y); A = np.tensordot(c, F, 1); ev, vec = np.linalg.eigh(A)
        sc = np.abs(ev).max()
        f = (ev[0] - np.log(np.sum(np.exp(-beta * (ev - ev[0])))) / beta) / sc
        wgt = np.exp(-beta * (ev - ev[0])); wgt /= wgt.sum()
        g = np.array([np.sum(wgt * np.einsum('ij,jk,ki->i', vec.T, Fk, vec)) for Fk in F]) / sc
        return -f, -(g - c * (g @ c)) / np.linalg.norm(y)
    for tr in range(tries):
        y = rng.standard_normal(m)
        for beta in (5, 50, 500, 5000): y = minimize(negf, y, args=(beta,), jac=True, method='L-BFGS-B').x
        c = y / np.linalg.norm(y); A = np.tensordot(c, F, 1); ev = np.linalg.eigvalsh(A); lm = ev[0] / np.abs(ev).max()
        if lm > best[0]: best = (lm, c)
    return best
for N in [5, 6, 7, 8, 10, 12, 16, 20, 30]:
    M, _ = pl.cap_matrix(P, k, N, 'HL'); null, rank = pl.nullspace(M)
    B1 = eig_block(null, Fr(1)); Bm = eig_block(null, Fr(-1)); basis = B1 + Bm
    # orthonormalize the basis (float) so that lambda_min is meaningful; forms are exact then projected
    Qs = forms(basis, N)
    Bf = np.array([[float(v) for v in b] for b in basis]).T
    G = Bf.T @ Bf; Li = np.linalg.inv(np.linalg.cholesky(G))
    Qn = [Li @ np.array([[float(v) for v in row] for row in Q]) @ Li.T for Q in Qs]
    Qn = [[[Fr(x) for x in row] for row in Q] for Q in Qn]
    lm, c = best_cert(Qn)
    # scale c so that the largest |entry| is 1, print
    c = c / np.abs(c).max()
    out(f'N={N}: merged block dim {len(basis)}, best relative lambda_min {lm:.3e}; c = ' + ', '.join(f'{v:+.4f}' for v in c))
