# Level-8 quadrics at N = 11 in real (float64) arithmetic; keep an orthonormal basis of their span.
import sys, time, numpy as np, scipy.sparse as sp
from engine import labels, pair_keys, decode, rgs_partition, noncrossing, crossing_parts, config_outputs
k = 8; N = int(sys.argv[1]); part, nparts = int(sys.argv[2]), int(sys.argv[3])
Wd = np.load(f'W8_real.npy'); crk = crossing_parts(k); L, nb = labels(crk, k)
n, d = Wd.shape; iu = np.triu_indices(d)
confs = [(s, t, j, o1, o2) for s in range(k) for t in range(k) for j in range(4, k + 1)
         for o1 in (0, 1) for o2 in (0, 1) if 2 * k - 2 * j + o1 + o2 <= k - 1 and len(config_outputs(k, (s, t, j, o1, o2))[2]) >= 4]
confs = confs[part::nparts]
Npow = np.array([float(N) ** e for e in range(17)])
basis = np.zeros((0, len(iu[0]))); cache = {}
for ci, conf in enumerate(confs):
    t = time.time(); rows = []
    for c0 in range(0, n, 200):
        prow = np.arange(c0, min(n, c0 + 200))
        code, closed, nout = pair_keys(L, nb, k, conf, prow)
        flat = code.ravel(); uc, inv = np.unique(flat, return_inverse=True)
        keep = [i for i, u in enumerate(uc) if cache.setdefault((int(u), nout), not noncrossing(rgs_partition(decode(int(u), nout))))]
        if not keep: continue
        keep = np.array(keep); kmap = -np.ones(len(uc), dtype=np.int64); kmap[keep] = np.arange(len(keep))
        kidx = kmap[inv.ravel()]; sel = kidx >= 0
        P_ = np.repeat(prow, n)[sel]; Q_ = np.tile(np.arange(n), len(prow))[sel]
        S = sp.csr_matrix((Npow[closed.ravel()[sel]], (kidx[sel] * len(prow) + (P_ - c0), Q_)), shape=(len(keep) * len(prow), n))
        SW = (S @ Wd).reshape(len(keep), len(prow), d)
        for ki in range(len(keep)):
            rows.append((int(uc[keep[ki]]), Wd[prow].T @ SW[ki]))
    # sum over chunks per output code
    acc = {}
    for code_, Q in rows: acc[code_] = acc.get(code_, 0) + Q
    if acc:
        X = np.array([(Q + Q.T)[iu] for Q in acc.values()])
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        X = X - (X @ basis.T) @ basis
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        new = Vt[s > 1e-8 * max(1.0, s.max() if len(s) else 1)]
        if len(new):
            new = new - (new @ basis.T) @ basis; q, _ = np.linalg.qr(new.T); basis = np.vstack([basis, q.T])
    print(f'{ci+1}/{len(confs)} {conf} rank {basis.shape[0]} ({time.time()-t:.1f}s)', flush=True)
np.save(f'span8_{"abcdefgh"[part]}.npy', basis)
