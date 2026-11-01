# Search for common zeros of the level-8 quadrics inside symmetry subspaces of K_8 (N=11, real arithmetic).
import numpy as np
from engine import crossing_parts, canon
k = 8
W = np.load('W8_real.npy'); span = np.load('span8.npy'); d = W.shape[1]; iu = np.triu_indices(d)
def sym_of(v):
    S = np.zeros((d, d)); S[iu] = v; return S + S.T - np.diag(np.diag(S))
S_all = np.array([sym_of(v) for v in span])
crk = crossing_parts(k); idx = {P: a for a, P in enumerate(crk)}
def opK(f):
    Pm = np.zeros((len(crk), len(crk)))
    for a, P in enumerate(crk): Pm[idx[canon([set(f(x) for x in B) for B in P])], a] = 1
    return W.T @ Pm @ W
R = opK(lambda x: (x + 1) % k); F = opK(lambda x: (-x) % k)
def fixed(ops):
    M = np.vstack([o - np.eye(d) for o in ops]); U, s, Vt = np.linalg.svd(M)
    return Vt[np.sum(s > 1e-9):].T
subs = {
    'D8-invariant': fixed([R, F]),
    'rotation-invariant': fixed([R]),
    'invariant under R^2 and F': fixed([R @ R, F]),
    'invariant under R^2': fixed([R @ R]),
    'invariant under R^4 and F': fixed([np.linalg.matrix_power(R, 4), F]),
    'invariant under R^4': fixed([np.linalg.matrix_power(R, 4)]),
    'invariant under F': fixed([F]),
}
rng = np.random.default_rng(1)
for name, V in subs.items():
    m = V.shape[1]
    Sr = np.einsum('ia,rij,jb->rab', V, S_all, V)
    # rank of restricted quadric system
    vecs = np.array([S[np.triu_indices(m)] for S in Sr]) if m else np.zeros((0, 0))
    rk = np.linalg.matrix_rank(vecs, tol=1e-8) if m else 0
    best = []; found = 0
    for trial in range(30):
        y = rng.normal(size=m) + 1j * rng.normal(size=m); y /= np.linalg.norm(y)
        for it in range(400):
            f = np.einsum('a,rab,b->r', y, Sr, y); Jm = 2 * np.einsum('rab,b->ra', Sr, y)
            A = np.vstack([Jm, np.conj(y)[None, :]]); b = np.concatenate([-f, [0]])
            dy = np.linalg.lstsq(A, b, rcond=None)[0]; y = y + dy; y /= np.linalg.norm(y)
            if np.linalg.norm(dy) < 1e-14: break
        res = np.linalg.norm(np.einsum('a,rab,b->r', y, Sr, y)); best.append(res)
        if res < 1e-10: found += 1
    print(f'{name}: dim {m}, restricted quadric rank {rk} of {m*(m+1)//2}; zeros found {found}/30, best residual {min(best):.1e}', flush=True)
