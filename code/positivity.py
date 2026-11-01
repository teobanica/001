#!/usr/bin/env python3
# positivity.py -- positivity certificates for the S_N problem at level k (test at level 8, N = 11).
#
# The argument (transported from Section 11 of the paper):
#   * if C first differs from NC at level k, then W = C(k) (-) NC(k) is nonzero, invariant under the rotation R of the
#     legs and under the antilinear involution J = conj o rev; since J R J = R^{-1}, each eigenspace W_w (R = w) is
#     J-invariant and contains a nonzero J-real vector x = Jx;
#   * on such an x, every quadric q_tau(x) = coeff_tau(x o x) (ribbon contraction landing at level < k, crossing
#     output tau) vanishes; q_tau(x) is a complex number, and Re q_tau, Im q_tau are REAL quadratic forms on the
#     real vector space U_w^J of J-real vectors of the w-eigenspace of K_k;
#   * so if some real combination of these forms is positive definite on U_w^J, then W_w = 0.
# If this happens for every eigenvalue w, the level is excluded.  Positivity is sufficient, not necessary.
#
# Input: W8_real.npy (orthonormal basis of K_8 in crossing coordinates, from realbasis.py) and span8_*.npy
# (orthonormal basis of the span of the quadratic forms, in the coordinates c of that basis, from realforms.py).
import sys, glob, numpy as np
from scipy.optimize import minimize
from engine import crossing_parts, canon

k = 8
crk = crossing_parts(k); idx = {P: a for a, P in enumerate(crk)}; n = len(crk)
Wd = np.load('W8_real.npy'); d = Wd.shape[1]
files = sorted(glob.glob('span8_*.npy')); S = np.vstack([np.load(f) for f in files])
U_, s_, Vt_ = np.linalg.svd(S, full_matrices=False); S = Vt_[s_ > 1e-6 * s_[0]]
print(f'kernel dim {d}; quadric span from {files}: dimension {S.shape[0]} (of {d*(d+1)//2})', flush=True)
iu = np.triu_indices(d)

def perm_matrix(f):
    P = np.zeros((n, n))
    for a, Pt in enumerate(crk): P[idx[canon([set(f(x) for x in B) for B in Pt])], a] = 1
    return P
PR = perm_matrix(lambda x: (x + 1) % k); PJ = perm_matrix(lambda x: k - 1 - x)
Rc = Wd.T @ PR @ Wd; Jc = Wd.T @ PJ @ Wd
print('invariance of K under R, J:', np.abs(PR @ Wd - Wd @ Rc).max(), np.abs(PJ @ Wd - Wd @ Jc).max(), flush=True)
print('J R J = R^-1:', np.abs(Jc @ Rc @ Jc - Rc.T).max(), flush=True)

def sym(v):
    F = np.zeros((d, d)); F[iu] = v; F = F + F.T - np.diag(np.diag(F)); return F
forms = [sym(v) for v in S]

def real_basis_Jreal(j):
    """Real basis of U_j^J = {x in E_j : J conj(x) = x}, E_j the w^j-eigenspace of Rc."""
    w = np.exp(2j * np.pi * j / k)
    Pj = sum(w ** (-r) * np.linalg.matrix_power(Rc, r) for r in range(k)) / k        # projector onto E_j
    Uu, ss, _ = np.linalg.svd(Pj); V = Uu[:, ss > 0.5]; dj = V.shape[1]
    if dj == 0: return np.zeros((d, 0))
    cand = []
    for y in list(V.T) + list((1j * V).T):
        x = y + Jc @ np.conj(y); cand.append(x)
    C = np.array(cand)                                                               # candidates, J-real
    Cr = np.hstack([C.real, C.imag])                                                 # as real vectors
    Uc, sc, Vtc = np.linalg.svd(Cr, full_matrices=False); keep = sc > 1e-8 * sc[0]
    B = Vtc[keep]; B = B[:, :d] + 1j * B[:, d:]                                        # real-orthonormal basis of U_j^J
    assert B.shape[0] == dj, (B.shape, dj)
    assert np.abs(Jc @ np.conj(B.T) - B.T).max() < 1e-9
    return B.T                                                                       # d x dj complex

def restricted_forms(V):
    out = []
    for F in forms:
        M = V.T @ F @ V
        for A in (M.real, M.imag):
            A = (A + A.T) / 2
            if np.abs(A).max() > 1e-10: out.append(A)
    if not out: return []
    X = np.array([A[np.triu_indices(V.shape[1])] * np.where(np.triu_indices(V.shape[1])[0] == np.triu_indices(V.shape[1])[1], 1, np.sqrt(2)) for A in out])
    Ux, sx, Vtx = np.linalg.svd(X, full_matrices=False); Vtx = Vtx[sx > 1e-8 * sx[0]]
    dj = V.shape[1]; scale = np.where(np.triu_indices(dj)[0] == np.triu_indices(dj)[1], 1, 1 / np.sqrt(2))
    basis = []
    for v in Vtx:
        A = np.zeros((dj, dj)); A[np.triu_indices(dj)] = v * scale; A = A + A.T - np.diag(np.diag(A)); basis.append(A)
    return basis

def best_certificate(B, tries=20, seed=0):
    """maximize lambda_min(sum c_k B_k) over ||c|| = 1 (smooth soft-min surrogate, several beta, random restarts)."""
    m = len(B); rng = np.random.default_rng(seed); Bt = np.array(B)
    def lam_min(c):
        A = np.tensordot(c, Bt, 1); return np.linalg.eigvalsh(A)[0]
    def negf(y, beta):
        c = y / np.linalg.norm(y); A = np.tensordot(c, Bt, 1); ev, vec = np.linalg.eigh(A)
        wgt = np.exp(-beta * (ev - ev[0])); wgt /= wgt.sum()
        f = ev[0] - np.log(np.sum(np.exp(-beta * (ev - ev[0])))) / beta
        g = np.array([np.sum(wgt * np.einsum('ij,jk,ki->i', vec.T, Bk, vec)) for Bk in Bt])     # d f / d c
        ny = np.linalg.norm(y); gy = (g - c * (g @ c)) / ny                                  # chain rule through normalization
        return -f, -gy
    best = (-np.inf, None)
    for t in range(tries):
        y = rng.standard_normal(m)
        for beta in (2, 10, 50, 200, 1000):
            res = minimize(negf, y, args=(beta,), jac=True, method='L-BFGS-B', options={'maxiter': 500})
            y = res.x
        c = y / np.linalg.norm(y); lm = lam_min(c)
        if lm > best[0]: best = (lm, c)
    return best

print()
total_ok = True
for j in range(k):
    V = real_basis_Jreal(j); dj = V.shape[1]
    if dj == 0: print(f'character {j}: empty'); continue
    B = restricted_forms(V); m = len(B)
    if m == 0: print(f'character {j}: dim {dj}, NO forms'); total_ok = False; continue
    lm, c = best_certificate(B)
    A = np.tensordot(c, np.array(B), 1); ev = np.linalg.eigvalsh(A)
    status = 'POSITIVE DEFINITE: block excluded' if lm > 1e-9 else 'no positive combination found'
    if lm <= 1e-9:
        # search for real common zeros of the restricted forms on the unit sphere (least squares, many starts)
        from scipy.optimize import least_squares
        Bt = np.array(B); rng = np.random.default_rng(1); bestres = np.inf; bestu = None
        for t in range(200):
            u0 = rng.standard_normal(dj); u0 /= np.linalg.norm(u0)
            fun = lambda u: np.concatenate([np.einsum('i,kij,j->k', u, Bt, u), [u @ u - 1]])
            r = least_squares(fun, u0, xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=2000)
            if r.cost < bestres: bestres, bestu = r.cost, r.x
        res = np.linalg.norm(np.einsum('i,kij,j->k', bestu, Bt, bestu))
        print(f'   real-zero search on block {j}: smallest residual of the {m} forms on the unit sphere = {res:.2e}', flush=True)
        if res < 1e-8:
            np.save('realzero_block.npy', bestu)
            # dimension of the tangent space of the real zero set at this point (rank of the Jacobian)
            Jm = 2 * np.einsum('kij,j->ki', Bt, bestu); rk = np.linalg.matrix_rank(Jm, tol=1e-8)
            print(f'   a real common zero exists; Jacobian rank there {rk} of {dj-1} needed for isolation', flush=True)
    total_ok &= lm > 1e-9
    print(f'character {j}: dim {dj}, restricted span {m} of {dj*(dj+1)//2} forms; best min eigenvalue {lm:.3e} '
          f'(max {ev[-1]:.3e}) -> {status}', flush=True)
    np.save(f'cert_char{"x"*j if j else "zero"}.npy', {'c': c, 'V': V, 'lam': lm}, allow_pickle=True)
print('\nALL BLOCKS EXCLUDED BY POSITIVITY' if total_ok else '\nsome blocks not excluded by positivity (with these forms)')
