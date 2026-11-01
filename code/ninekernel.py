#!/usr/bin/env python3
# ninekernel.py -- K_9(N) in floating point, split by rotation characters (the rotation trick, Prop. 9.9), with the
# J-real structure of each block, for the positivity test at level 9.   Usage: python3 ninekernel.py [N]   (default 11)
#
# Output: ninekernel.npz with
#   V      : complex (16285 x 307) orthonormal columns, the kernel vectors, block j = columns blocks[j]:blocks[j+1]
#   blocks : the 10 block boundaries (character j = 0..8)
#   A      : object array of 9 complex (d_j x d_j) matrices: columns = real basis of the J-real vectors of block j,
#            in the coordinates of V's block j (x = V_j a, a = A_j u, u real)
import sys, os, time, numpy as np, scipy.sparse as sp
from engine import crossing_parts, canon, delete_leg, merge

k = 9; N = int(sys.argv[1]) if len(sys.argv) > 1 else 11
OUT = sys.argv[2] if len(sys.argv) > 2 else '.'; os.makedirs(OUT, exist_ok=True)
t0 = time.time()
crk = crossing_parts(k); crl = crossing_parts(k - 1); idx = {P: a for a, P in enumerate(crk)}; il = {q: a for a, q in enumerate(crl)}
n, nl = len(crk), len(crl); print(f'level {k}, N={N}: {n} crossing partitions, {nl} at level {k-1}', flush=True)
rot = np.array([idx[canon([set((x + 1) % k for x in B) for B in P])] for P in crk])
rev = np.array([idx[canon([set(k - 1 - x for x in B) for B in P])] for P in crk])

def ops_matrix(indices):
    """sparse real matrix of the leg-sums d_i and merges m_i, i in indices, crossing outputs only"""
    rows, cols, vals = [], [], []; r0 = 0
    for typ, i in [('d', i) for i in indices] + [('m', i) for i in indices]:
        for a, P in enumerate(crk):
            if typ == 'd':
                sing, q = delete_leg(P, i, k)
                if q in il: rows.append(r0 + il[q]); cols.append(a); vals.append(float(N) if sing else 1.0)
            else:
                q = merge(P, i, (i + 1) % k, k)
                if q in il: rows.append(r0 + il[q]); cols.append(a); vals.append(1.0)
        r0 += nl
    return sp.csr_matrix((vals, (rows, cols)), shape=(r0, n))
D0 = ops_matrix([0]); Dall = ops_matrix(range(k))
print(f'operator matrices built ({time.time()-t0:.0f}s)', flush=True)

# rotation orbits and the eigenbasis of R on the crossing coordinates
seen = np.zeros(n, bool); orbits = []
for a in range(n):
    if seen[a]: continue
    o = [a]; b = rot[a]
    while b != a: o.append(b); b = rot[b]
    seen[o] = True; orbits.append(o)
w = np.exp(2j * np.pi / k)
Vblocks, Ablocks, dims = [], [], []
for j in range(k):
    cols = []
    for o in orbits:
        L = len(o)
        if (j * L) % k: continue
        v = np.zeros(n, complex); v[o] = [w ** (-j * r) for r in range(L)]; v /= np.sqrt(L); cols.append(v)
    E = np.array(cols).T                                   # orthonormal basis of the w^j eigenspace of R
    Aj = D0 @ E                                            # d_0, m_0 on this eigenspace
    U_, s_, Vt_ = np.linalg.svd(Aj, full_matrices=True)
    rank = int(np.sum(s_ > 1e-9 * s_[0])); null = Vt_[rank:].conj().T    # orthonormal null vectors (columns)
    Kj = E @ null; dj = Kj.shape[1]
    err = np.abs(Dall @ Kj).max()                           # all leg-sums and merges must vanish (Prop. 9.9)
    err_rot = np.abs(Kj[np.argsort(rot)] - w ** j * Kj).max()   # (R x)_q = x_{rot^{-1} q}
    # J = conj o rev maps E_j to E_j; its antilinear action on coordinates a: a -> T conj(a)
    JKj = np.conj(Kj)[rev]                                  # (J x)_{rev pi} = conj(x_pi), rev an involution
    T = Kj.conj().T @ JKj                                   # coordinates of J(basis) in the basis
    err_J = np.abs(Kj @ T - JKj).max()
    # real basis of the J-real vectors: fixed points of a -> T conj(a)
    cand = []
    for y in list(np.eye(dj)) + list(1j * np.eye(dj)): cand.append(y + T @ np.conj(y))
    C = np.array(cand); Cr = np.hstack([C.real, C.imag])
    Uc, sc, Vtc = np.linalg.svd(Cr, full_matrices=False); B = Vtc[sc > 1e-8 * sc[0]]
    A = (B[:, :dj] + 1j * B[:, dj:]).T                      # dj x dj complex, columns = real basis of U_j^J
    assert A.shape[1] == dj, (A.shape, dj)
    err_real = np.abs(T @ np.conj(A) - A).max()
    print(f'character {j}: dim E_j = {E.shape[1]}, dim K_j = {dj}; checks: all ops {err:.1e}, rotation {err_rot:.1e}, '
          f'J-invariance {err_J:.1e}, J-real basis {err_real:.1e}  ({time.time()-t0:.0f}s)', flush=True)
    Vblocks.append(Kj); Ablocks.append(A); dims.append(dj)
V = np.hstack(Vblocks); blocks = np.concatenate([[0], np.cumsum(dims)])
print('dim K_9 =', V.shape[1], 'by character', dims, flush=True)
np.savez(os.path.join(OUT, 'ninekernel.npz'), V=V, blocks=blocks, A=np.array(Ablocks, dtype=object), N=N)
print(f'saved {OUT}/ninekernel.npz ({time.time()-t0:.0f}s)')
