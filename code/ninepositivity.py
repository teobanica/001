#!/usr/bin/env python3
# ninepositivity.py -- positivity certificates at level 9, block by block, from nineforms_*.npz (restricted real
# quadratic forms on the J-real slices of the rotation blocks of K_9).  Works on partial data too (fewer forms:
# a certificate found is still valid, a failure may disappear with more forms).
import sys, os, glob, numpy as np
OUT = sys.argv[1] if len(sys.argv) > 1 else '.'
from scipy.optimize import minimize, least_squares

files = sorted(glob.glob(os.path.join(OUT, 'nineforms_*.npz'))); Zs = [np.load(f, allow_pickle=True) for f in files]
nblk = 9; dims = [int(np.sqrt(2 * Zs[0][f'block{"x"*j}'].shape[1] + 0.25) - 0.5) for j in range(nblk)]
ndone = sum(len(Z['done']) for Z in Zs)
print(f'files {files}, configurations done {ndone} of 90; blocks {dims}', flush=True)

def untri(v, d):
    i, j = np.triu_indices(d); M = np.zeros((d, d)); M[i, j] = v / np.where(i == j, 1.0, np.sqrt(2)); return M + M.T - np.diag(np.diag(M))

def best_certificate(B, tries=12, seed=0):
    m = len(B); rng = np.random.default_rng(seed); Bt = np.array(B)
    def lam_min(c): return np.linalg.eigvalsh(np.tensordot(c, Bt, 1))[0]
    def negf(y, beta):
        c = y / np.linalg.norm(y); ev, vec = np.linalg.eigh(np.tensordot(c, Bt, 1))
        wgt = np.exp(-beta * (ev - ev[0])); wgt /= wgt.sum()
        f = ev[0] - np.log(np.sum(np.exp(-beta * (ev - ev[0])))) / beta
        g = np.array([np.sum(wgt * np.einsum('ij,jk,ki->i', vec.T, Bk, vec)) for Bk in Bt])
        ny = np.linalg.norm(y); gy = (g - c * (g @ c)) / ny
        return -f, -gy
    best = (-np.inf, None)
    for t in range(tries):
        y = rng.standard_normal(m)
        for beta in (2, 10, 50, 200, 1000, 5000):
            res = minimize(negf, y, args=(beta,), jac=True, method='L-BFGS-B', options={'maxiter': 800})
            y = res.x
        c = y / np.linalg.norm(y); lm = lam_min(c)
        if lm > best[0]: best = (lm, c)
    return best

allok = True
for j, d in enumerate(dims):
    X = np.vstack([Z[f'block{"x"*j}'] for Z in Zs])
    U_, s_, Vt_ = np.linalg.svd(X, full_matrices=False); X = Vt_[s_ > 1e-6 * s_[0]]
    B = [untri(v, d) for v in X]; m = len(B)
    if m == 0: print(f'character {j}: dim {d}, no forms yet'); allok = False; continue
    lm, c = best_certificate(B); ev = np.linalg.eigvalsh(np.tensordot(c, np.array(B), 1))
    ok = lm > 1e-9; allok &= ok
    print(f'character {j}: dim {d}, restricted span {m} of {d*(d+1)//2}; best min eigenvalue {lm:.3e} (max {ev[-1]:.3e}) '
          f'-> {"POSITIVE DEFINITE: block excluded" if ok else "no positive combination found"}', flush=True)
    np.save(os.path.join(OUT, f'ninecert_{"abcdefghi"[j]}.npy'), {'c': c, 'lam': lm, 'forms': np.array(B)}, allow_pickle=True)
    if not ok:
        Bt = np.array(B); rng = np.random.default_rng(1); bestres = np.inf
        for t in range(60):
            u0 = rng.standard_normal(d); u0 /= np.linalg.norm(u0)
            r = least_squares(lambda u: np.concatenate([np.einsum('i,kij,j->k', u, Bt, u), [u @ u - 1]]), u0, xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=3000)
            bestres = min(bestres, np.linalg.norm(np.einsum('i,kij,j->k', r.x, Bt, r.x)))
        print(f'   real-zero search: smallest residual on the unit sphere {bestres:.2e}', flush=True)
print('\nALL BLOCKS EXCLUDED BY POSITIVITY' if allok else '\nsome blocks not excluded (yet)')
