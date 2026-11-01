# Vectorised triple contractions at level k, modulo p = 1048433, in the rotation eigenbasis y of K_k.
# Step A: y-forms Y[pi] (quadratic, all output partitions pi of P(k)) of a first ribbon landing at level k.
# Step B: second ribbon of (pi, sigma), pi in P(k), sigma crossing, landing at level <= k-1 with crossing output:
#         cubic forms sum_pi Y[pi](y) * (sum_sigma W'[sigma] N^closed) -> rows over cubic monomials.
import sys, time, itertools, json, numpy as np, scipy.sparse as sp
from engine import parts, crossing_parts, labels, config_outputs, decode, rgs_partition, noncrossing, canon
p = 1048433

def pair_keys2(Lp, nbp, Lq, nbq, k, conf, prow):
    xs, ys, outs = config_outputs(k, conf)
    m = len(prow); n = Lq.shape[0]
    lab = np.empty((m, n, 16), dtype=np.int8); lab[:] = np.arange(16, dtype=np.int8)
    lx = Lp[prow]; ly = Lq + 8
    for x, y in zip(xs, ys):
        u = np.broadcast_to(lx[:, x][:, None], (m, n)); v = np.broadcast_to(ly[:, y][None, :], (m, n))
        ru = np.take_along_axis(lab, u[:, :, None].astype(np.int64), 2)[:, :, 0]
        rv = np.take_along_axis(lab, v[:, :, None].astype(np.int64), 2)[:, :, 0]
        lab = np.where(lab == rv[:, :, None], ru[:, :, None], lab)
    oc = []
    for side, v in outs:
        node = np.broadcast_to(lx[:, v][:, None], (m, n)) if side == 'x' else np.broadcast_to(ly[:, v][None, :], (m, n))
        oc.append(np.take_along_axis(lab, node[:, :, None].astype(np.int64), 2)[:, :, 0])
    oc = np.stack(oc, axis=2); nout = oc.shape[2]
    rgs = np.zeros_like(oc); seen = np.full((m, n, 16), -1, dtype=np.int8); cnt = np.zeros((m, n), dtype=np.int8)
    ii, jj = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
    for o in range(nout):
        lbl = oc[:, :, o].astype(np.int64); cur = seen[ii, jj, lbl]; new = cur < 0
        seen[ii[new], jj[new], lbl[new]] = cnt[new]; rgs[:, :, o] = np.where(new, cnt, cur); cnt = cnt + new
    code = np.zeros((m, n), dtype=np.int64)
    for o in range(nout): code = code * 16 + rgs[:, :, o]
    valid = np.zeros((m, n, 16), dtype=bool)
    valid[:, :, :8] = (np.arange(8)[None, :] < nbp[prow][:, None])[:, None, :]
    valid[:, :, 8:] = (np.arange(8)[None, :] < nbq[:, None])[None, :, :]
    isroot = valid & (lab == np.arange(16, dtype=np.int8)[None, None, :])
    closed = (isroot.sum(axis=2) - cnt).astype(np.int8)
    return code, closed, nout

def setup(k, N):
    W = np.load('W8_mod.npy') if k == 8 else None
    return W

def first_forms(Lc, nbc, k, conf, Wy, N, allP_index, chunk=150):
    """Y: array (|P(k)|, d, d) of symmetric quadratic forms (mod p) of the first ribbon (output level k)."""
    n, d = Wy.shape
    Npow = np.array([pow(N, e, p) for e in range(17)], dtype=np.float64)
    Y = np.zeros((len(allP_index), d, d))
    for c0 in range(0, n, chunk):
        prow = np.arange(c0, min(n, c0 + chunk))
        code, closed, nout = pair_keys2(Lc, nbc, Lc, nbc, k, conf, prow)
        assert nout == k
        flat = code.ravel(); uc, inv = np.unique(flat, return_inverse=True)
        pidx = np.array([allP_index[rgs_partition(decode(int(u), nout))] for u in uc])
        P_ = np.repeat(prow, n); Q_ = np.tile(np.arange(n), len(prow))
        rowsS = inv.ravel() * len(prow) + (P_ - c0)
        S = sp.csr_matrix((Npow[closed.ravel()], (rowsS, Q_)), shape=(len(uc) * len(prow), n))
        SW = np.mod(S @ Wy, p).reshape(len(uc), len(prow), d)
        Wp = Wy[prow]
        for ui in range(len(uc)):
            Y[pidx[ui]] = np.mod(Y[pidx[ui]] + Wp.T @ SW[ui], p)
    Y = np.mod(Y + np.transpose(Y, (0, 2, 1)), p)
    return Y

def second_cubics(Lall, nball, Lc, nbc, k, conf, Y, Wy, N, mon_index, chunk=100, cache=None):
    """Cubic constraint rows (in cubic monomials of y), one per crossing output partition."""
    nP, d, _ = Y.shape; n = Lc.shape[0]
    if cache is None: cache = {}
    iu = np.triu_indices(d); wgt = np.where(iu[0] == iu[1], 1.0, 2.0)
    Yflat = Y[:, iu[0], iu[1]] * wgt[None, :]              # nP x 2211, coefficient of y_a y_b (a<=b)
    nz = np.flatnonzero(np.abs(Yflat).sum(axis=1) > 0)
    Npow = np.array([pow(N, e, p) for e in range(17)], dtype=np.float64)
    acc = {}
    for c0 in range(0, len(nz), chunk):
        prow = nz[c0:c0 + chunk]
        code, closed, nout = pair_keys2(Lall, nball, Lc, nbc, k, conf, prow)
        flat = code.ravel(); uc, inv = np.unique(flat, return_inverse=True)
        keep = [i for i, u in enumerate(uc) if cache.setdefault((int(u), nout), nout >= 4 and not noncrossing(rgs_partition(decode(int(u), nout))))]
        if not keep: continue
        keep = np.array(keep); kmap = -np.ones(len(uc), dtype=np.int64); kmap[keep] = np.arange(len(keep))
        kidx = kmap[inv.ravel()]; sel = kidx >= 0
        P_ = np.repeat(np.arange(len(prow)), n)[sel]; Q_ = np.tile(np.arange(n), len(prow))[sel]
        S = sp.csr_matrix((Npow[closed.ravel()[sel]], (kidx[sel] * len(prow) + P_, Q_)), shape=(len(keep) * len(prow), n))
        SW = np.mod(S @ Wy, p).reshape(len(keep), len(prow), d)
        Yc = Yflat[prow]                                    # m x 2211
        for ki, ui in enumerate(keep):
            C = np.mod(Yc.T @ SW[ki], p)                    # 2211 x d : coefficient of (y_a y_b) * y_e
            u = int(uc[ui]); acc[u] = np.mod(acc[u] + C, p) if u in acc else C
    rows = []
    for u, C in acc.items():
        v = np.zeros(len(mon_index['list']))
        np.add.at(v, mon_index['map'].ravel(), C.ravel())
        rows.append(np.mod(v, p))
    return rows

if __name__ == '__main__':
    k = int(sys.argv[1]); N = int(sys.argv[2]); nfirst = int(sys.argv[3]); nsecond = int(sys.argv[4]); tag = sys.argv[5]
    E = np.load(f'E{"x" * k}.npy'); chars = np.load(f'chars{"x" * k}.npy')
    W = np.load(f'Wmod{"x" * k}.npy')
    Wy = np.mod(W @ E, p).astype(np.float64) if W.shape[1] < 100 else None
    Wy = np.zeros(W.shape)
    El = E & 0x3FF; Eh = E >> 10
    Wy = np.mod(np.mod(W @ El, p) + np.mod(np.mod(W @ Eh, p) * 1024, p), p).astype(np.float64)
    d = Wy.shape[1]
    crk = crossing_parts(k); Lc, nbc = labels(crk, k)
    allP = sorted(parts(k)); Lall, nball = labels(allP, k); allP_index = {P: i for i, P in enumerate(allP)}
    iu = np.triu_indices(d)
    mon3 = list(itertools.combinations_with_replacement(range(d), 3)); m3 = {m: i for i, m in enumerate(mon3)}
    mp = np.array([[m3[tuple(sorted((a, b, e)))] for e in range(d)] for a, b in zip(*iu)])
    mon_index = {'list': mon3, 'map': mp}
    def confs(target):
        return [(s, t, j, o1, o2) for s in range(k) for t in range(k) for j in range(1, k + 1) for o1 in (0, 1) for o2 in (0, 1)
                if not (o2 and j == 1) and target(2 * k - 2 * j + o1 + o2)]
    first = [c for c in confs(lambda L: L == k) if c[0] == 0 and c[1] == 0][:nfirst]
    second = [c for c in confs(lambda L: 4 <= L <= k - 1) if c[0] == 0][:nsecond]
    print(f'level {k}, N={N}: d={d}, first {first}, {len(second)} second configs', flush=True)
    allrows = []; cache = {}
    for c1 in first:
        t = time.time(); Y = first_forms(Lc, nbc, k, c1, Wy, N, allP_index); print(f'  first {c1}: {time.time()-t:.0f}s', flush=True)
        for c2 in second:
            t = time.time(); rows = second_cubics(Lall, nball, Lc, nbc, k, c2, Y, Wy, N, mon_index, cache=cache)
            allrows += rows; print(f'    second {c2}: {len(rows)} cubics ({time.time()-t:.0f}s)', flush=True)
    np.save(f'trip_{tag}.npy', np.array(allrows, dtype=np.int64) if allrows else np.zeros((0, len(mon3)), dtype=np.int64))
