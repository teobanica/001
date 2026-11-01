#!/usr/bin/env python3
# nineforms.py -- level-9 quadrics restricted to the J-real slices of the rotation blocks of K_9 (float64).
#   python3 nineforms.py part nparts        (e.g. 0 2 and 1 2 on two cores)
# For every ribbon configuration (0, t, j, o1, o2) landing at level <= 8 with >= 4 output legs, and every crossing
# output partition tau, the quadric q_tau(x) = coeff_tau(x o x) is restricted to each rotation block E_j of K_9 and
# then to its J-real slice, giving two real quadratic forms Re q, Im q in d_j real variables.  Only the configurations
# with s = 0 are needed: a rotation of the ribbon multiplies the restricted complex form on E_j by a unimodular scalar,
# which does not change the real span of {Re q, Im q}.  For each block a running orthonormal basis of the real span is
# kept (dimension at most d_j (d_j+1)/2).  Resumable: the state is saved after each configuration.
import sys, os, time, numpy as np, scipy.sparse as sp
from engine import labels, pair_keys, decode, rgs_partition, noncrossing, crossing_parts, config_outputs

k = 9; part, nparts = int(sys.argv[1]), int(sys.argv[2])
OUT = sys.argv[3] if len(sys.argv) > 3 else '.'
Z = np.load(os.path.join(OUT, 'ninekernel.npz'), allow_pickle=True); V = Z['V']; bl = Z['blocks']; A = Z['A']; N = int(Z['N'])
nb_ = len(bl) - 1; dims = [bl[j + 1] - bl[j] for j in range(nb_)]
crk = crossing_parts(k); L, nb = labels(crk, k); n = V.shape[0]
confs = [(0, t, j, o1, o2) for t in range(k) for j in range(4, k + 1) for o1 in (0, 1) for o2 in (0, 1)
         if 2 * k - 2 * j + o1 + o2 <= k - 1 and len(config_outputs(k, (0, t, j, o1, o2))[2]) >= 4]
confs = confs[part::nparts]
Npow = np.array([float(N) ** e for e in range(2 * k + 1)])
state = os.path.join(OUT, f'nineforms_{"abcdefgh"[part]}.npz')
if os.path.exists(state):
    S0 = np.load(state, allow_pickle=True); bases = [S0[f'block{"x"*j}'] for j in range(nb_)]; done = set(map(tuple, S0['done']))
else:
    bases = [np.zeros((0, d * (d + 1) // 2)) for d in dims]; done = set()
print(f'{len(confs)} configurations in this part, {len(done)} done; blocks {dims}', flush=True)

def tri(M):            # symmetric real d x d -> vector with Frobenius-compatible weights
    d = M.shape[0]; i, j = np.triu_indices(d); return M[i, j] * np.where(i == j, 1.0, np.sqrt(2))

def add_forms(j, X):
    """add the rows X (vectors tri(form)) to the running orthonormal basis of block j"""
    B = bases[j]
    if B.shape[0] == B.shape[1]: return
    X = X - (X @ B.T) @ B
    nrm = np.linalg.norm(X, axis=1); X = X[nrm > 1e-9]
    if len(X) == 0: return
    U_, s_, Vt_ = np.linalg.svd(X, full_matrices=False); new = Vt_[s_ > 1e-8 * s_[0]]
    new = new - (new @ B.T) @ B; q, _ = np.linalg.qr(new.T); bases[j] = np.vstack([B, q.T])

CH = 40
for ci, conf in enumerate(confs):
    if conf in done: continue
    t = time.time(); acc = {}; cache = {}
    for c0 in range(0, n, CH):
        prow = np.arange(c0, min(n, c0 + CH)); m = len(prow)
        code, closed, nout = pair_keys(L, nb, k, conf, prow)
        flat = code.ravel(); uc, inv = np.unique(flat, return_inverse=True)
        keep = [i for i, u in enumerate(uc) if cache.setdefault(int(u), not noncrossing(rgs_partition(decode(int(u), nout))))]
        if not keep: continue
        keep = np.array(keep); kmap = -np.ones(len(uc), dtype=np.int64); kmap[keep] = np.arange(len(keep))
        kidx = kmap[inv.ravel()]; sel = kidx >= 0
        P_ = np.repeat(prow, n)[sel]; Q_ = np.tile(np.arange(n), m)[sel]; KI = kidx[sel]; val = Npow[closed.ravel()[sel]]
        Vp = V[prow]                                                               # m x 307
        for g0 in range(0, len(keep), 300):
            g1 = min(len(keep), g0 + 300); gsel = (KI >= g0) & (KI < g1)
            S = sp.csr_matrix((val[gsel], ((KI[gsel] - g0) * m + (P_[gsel] - c0), Q_[gsel])), shape=((g1 - g0) * m, n))
            SV = (S @ V).reshape(g1 - g0, m, V.shape[1])                           # complex
            for gi in range(g1 - g0):
                cd = int(uc[keep[g0 + gi]])
                if cd not in acc: acc[cd] = [np.zeros((d, d), complex) for d in dims]
                for j in range(nb_):
                    sl = slice(bl[j], bl[j + 1]); acc[cd][j] += Vp[:, sl].T @ SV[gi][:, sl]
    for j in range(nb_):
        rows = []
        for cd, G in acc.items():
            Bm = G[j] + G[j].T; M = A[j].T @ Bm @ A[j]
            rows.append(tri(M.real)); rows.append(tri(M.imag))
        if rows: add_forms(j, np.array(rows))
    done.add(conf)
    np.savez(state, done=np.array(sorted(done)), **{f'block{"x"*j}': bases[j] for j in range(nb_)})
    print(f'{len(done)}/{len(confs)} {conf}: {len(acc)} crossing outputs; restricted spans {[b.shape[0] for b in bases]} '
          f'({time.time()-t:.0f}s)', flush=True)
print('finished', flush=True)
