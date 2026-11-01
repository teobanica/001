# Structure of Z = common zeros of the level-8 quadrics (N=11) in P^65.
import numpy as np
from engine import crossing_parts, canon
k = 8
W = np.load('W8_real.npy'); span = np.load('span8.npy')
d = W.shape[1]; iu = np.triu_indices(d)
def sym_of(v):
    S = np.zeros((d, d)); S[iu] = v; return S + S.T - np.diag(np.diag(S))
S_all = np.array([sym_of(v) for v in span])               # r x d x d
r = len(S_all); print('forms:', r, flush=True)
crk = crossing_parts(k); idx = {P: a for a, P in enumerate(crk)}
def perm_matrix(f):
    Pm = np.zeros((len(crk), len(crk)))
    for a, P in enumerate(crk): Pm[idx[canon([set(f(x) for x in B) for B in P])], a] = 1
    return Pm
Rot = W.T @ perm_matrix(lambda x: (x + 1) % k) @ W
Ref = W.T @ perm_matrix(lambda x: (-x) % k) @ W
print('rotation preserves K:', np.allclose(Rot @ Rot.T, np.eye(d)), ' order 8:', np.allclose(np.linalg.matrix_power(Rot, 8), np.eye(d)))
ev, V = np.linalg.eig(Rot)
roots = np.exp(2j * np.pi * np.arange(8) / 8)
groups = {j: V[:, np.abs(ev - roots[j]) < 1e-6] for j in range(8)}
print('rotation eigenspace dimensions (eigenvalue e^{2 pi i j/8}, j=0..7):', [groups[j].shape[1] for j in range(8)])
# do the forms vanish on an eigenspace (pair) ?
def vanishes_on(B1, B2):
    # all forms: B1^T S B2 + B2^T S B1 = 0  <=> c^T S c = 0 on span(B1) + span(B2) cross terms
    m = 0
    for S in S_all:
        m = max(m, np.abs(B1.T @ S @ B2).max() if B1.size and B2.size else 0)
    return m
for j in range(8):
    if groups[j].shape[1]:
        print(f'  max |form| on eigenspace j={j} (self): {vanishes_on(groups[j], groups[j]):.2e}')
# Gauss-Newton multistart for points of Z
rng = np.random.default_rng(0)
def F(c): return np.einsum('a,rab,b->r', c, S_all, c)
def J(c): return 2 * np.einsum('rab,b->ra', S_all, c)
pts = []
for trial in range(40):
    c = rng.normal(size=d) + 1j * rng.normal(size=d); c /= np.linalg.norm(c)
    lam = 1e-3
    for it in range(300):
        f = F(c); Jm = J(c)
        A = np.vstack([Jm, np.conj(c)[None, :]]); b = np.concatenate([-f, [0]])
        dc = np.linalg.lstsq(A, b, rcond=None)[0]
        c = c + dc; c /= np.linalg.norm(c)
        if np.linalg.norm(dc) < 1e-13: break
    res = np.linalg.norm(F(c))
    if res < 1e-9:
        sv = np.linalg.svd(J(c), compute_uv=False)
        rk = int((sv > 1e-7 * sv[0]).sum())
        pts.append((c, rk))
        print(f'  trial {trial}: converged, residual {res:.1e}, Jacobian rank {rk} -> local dim of Z (projective) <= {d - 1 - rk}', flush=True)
    else:
        print(f'  trial {trial}: no convergence (residual {res:.1e})', flush=True)
np.save('zpoints.npy', np.array([p for p, _ in pts]) if pts else np.zeros((0, d)))
