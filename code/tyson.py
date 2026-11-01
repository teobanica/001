#!/usr/bin/env python3
# Overnight computations for "On the maximality conjecture for quantum permutation groups" (Claude and Teo).
#
#   python3 tyson.py test     quick self-test (a few minutes); run this first on each laptop
#   python3 tyson.py a        on the first laptop
#   python3 tyson.py b        on the second laptop
# Both run level 8, N by N (two N at a time), on disjoint lists of N: a = 11, 9, 13, 15, ..., 37 and b = 12, 8, 10, ..., 36.
# For each N: start from the half-turn-invariant part of K_8 and remove rotation characters one class at a time
# (46 -> 26 -> 14 -> ... coordinates), using quadrics, triple contractions, degree 4 and "triangle" contractions.
# If the list is finished before the time is up, the laptop goes on with the level-9 quadric step.
#
# Needs numpy and scipy (as for halfturn.py).  Everything is written to the folder  tysonout/  next to this file.
# The file to keep is  tysonout/progress.txt  (the rest can stay on the laptop).
# The job stops by itself after HOURS hours.  If it is interrupted (lid closed, Ctrl-C, crash), just run the same
# command again: finished pieces are kept and it continues where it was.
#
# All arithmetic is exact, modulo primes p < 2^20 with p = 1 mod k (so F_p contains the k-th roots of unity),
# in float64 with products < 2^40 (exact).  Rank statements mod p are lower bounds for ranks over Q(omega).
import sys, os, time, json, itertools, random, traceback
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '3')
import numpy as np
import scipy.sparse as sp
import multiprocessing as mp

HOURS = 12.0
TRIANGLE_MINUTES = 75     # level 8: time allowed for the triangle contractions in one ring
WORKERS_EIGHT = 2          # level 8: number of N computed in parallel
WORKERS_NINE = 2           # level 9: number of ribbon configurations computed in parallel
PRIMES = {8: [1048433, 1048361, 1048273], 9: [1048573, 1048447, 1048357]}
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'tysonout')

# ======================================================================= partitions
def set_partitions(k):
    def rec(l):
        if l == 0:
            yield []; return
        for q in rec(l - 1):
            for i in range(len(q)): yield q[:i] + [q[i] + [l - 1]] + q[i + 1:]
            yield q + [[l - 1]]
    return list(rec(k))
def canon(blocks): return tuple(sorted(tuple(sorted(b)) for b in blocks if b))
def noncrossing(P):
    for A, B in itertools.permutations(P, 2):
        for a1, a2 in itertools.combinations(A, 2):
            for b1, b2 in itertools.combinations(B, 2):
                if a1 < b1 < a2 < b2: return False
    return True
_PARTS = {}
def parts(k):
    if k not in _PARTS: _PARTS[k] = sorted(canon([set(b) for b in q]) for q in set_partitions(k))
    return _PARTS[k]
def crossing_parts(k): return [q for q in parts(k) if not noncrossing(q)]
def delete_leg(P, i, k):
    sing = any(b == (i,) for b in P)
    return sing, canon([set(x if x < i else x - 1 for x in b if x != i) for b in P])
def merge(P, i, j, k):
    bi = [b for b in P if i in b][0]; bj = [b for b in P if j in b][0]
    blocks = [set(b) for b in P if b not in (bi, bj)] + [set(bi) | set(bj)]
    if j == i + 1:
        f = lambda x: x if x < j else x - 1
        return canon([set(f(x) for x in b if x != j) for b in blocks])
    return canon([set(x for x in b if x != i) for b in blocks])
def labels(P_list, k):
    L = np.zeros((len(P_list), k), dtype=np.int8); nb = np.zeros(len(P_list), dtype=np.int8)
    for a, P in enumerate(P_list):
        for bi, B in enumerate(sorted(P)):
            for x in B: L[a, x] = bi
        nb[a] = len(P)
    return L, nb
def decode(code, nout):
    r = []
    for _ in range(nout): r.append(int(code % 16)); code //= 16
    return tuple(reversed(r))
def rgs_partition(rgs):
    bl = {}
    for i, l in enumerate(rgs): bl.setdefault(l, set()).add(i)
    return canon(list(bl.values()))

# ======================================================================= linear algebra mod p
class BatchRank:
    """Row space mod p (p < 2^20), kept in reduced row echelon form; rows added in batches (float64 GEMMs)."""
    def __init__(self, dim, p):
        self.p = p; self.dim = dim; self.B = np.zeros((0, dim)); self.piv = []
    def reduce(self, X):
        p = self.p; X = np.mod(np.asarray(X, dtype=np.float64), p)
        if self.B.shape[0]:
            for c0 in range(0, len(self.piv), 4096):          # chunks of <= 4096 terms keep sums < 2^53
                pv = self.piv[c0:c0 + 4096]
                X = np.mod(X - np.mod(X[:, pv] @ self.B[c0:c0 + 4096], p), p)
        return X
    def add(self, X):
        p = self.p
        if len(X) == 0 or len(self.piv) == self.dim: return 0
        X = self.reduce(X)
        X = X[np.any(X != 0, axis=1)]
        if len(X) == 0: return 0
        X = X.astype(np.int64); newrows = []; newpiv = []
        for i in range(X.shape[0]):
            v = X[i]
            for c, r in zip(newpiv, newrows):
                if v[c]: v = (v - v[c] * r) % p
            nz = np.flatnonzero(v)
            if len(nz) == 0: continue
            c = nz[0]; v = (v * pow(int(v[c]), p - 2, p)) % p
            for t in range(len(newrows)):
                if newrows[t][c]: newrows[t] = (newrows[t] - newrows[t][c] * v) % p
            newrows.append(v); newpiv.append(c)
            if len(newpiv) + len(self.piv) == self.dim: break
        if not newrows: return 0
        Nw = np.array(newrows, dtype=np.float64)
        if self.B.shape[0]: self.B = np.mod(self.B - np.mod(self.B[:, newpiv] @ Nw, p), p)
        self.B = np.vstack([self.B, Nw]); self.piv += newpiv
        return len(newrows)
    def contains(self, v):
        return not np.any(self.reduce(np.asarray(v, dtype=np.float64)[None, :]))
    @property
    def rank(self): return len(self.piv)
    @property
    def gap(self): return self.dim - len(self.piv)

def nullspace_mod(M, p):
    M = np.mod(M, p).astype(np.float64); rows, cols = M.shape; r = 0; piv = []
    for c in range(cols):
        if r == rows: break
        nz = np.flatnonzero(M[r:, c])
        if len(nz) == 0: continue
        pr = r + nz[0]
        if pr != r: M[[r, pr]] = M[[pr, r]]
        M[r] = np.mod(M[r] * pow(int(M[r, c]), p - 2, p), p)
        oth = np.flatnonzero(M[:, c]); oth = oth[oth != r]
        if len(oth): M[oth] = np.mod(M[oth] - np.mod(np.outer(M[oth, c], M[r]), p), p)
        piv.append(c); r += 1
    ps = set(piv); fr = [c for c in range(cols) if c not in ps]
    Nm = np.zeros((cols, len(fr)), dtype=np.int64)
    for b, f in enumerate(fr):
        Nm[f, b] = 1
        for i, pc in enumerate(piv): Nm[pc, b] = int(-M[i, f]) % p
    return Nm

# ======================================================================= K_k(N) in the rotation eigenbasis
def eigenbasis(k, N, p):
    """Basis of K_k(N) mod p by eigenvectors of the rotation of the k legs: R w_b = omega^chars[b] w_b.
    On the omega^j eigenspace the leg-sums and merges at all legs reduce to those at leg 0."""
    crk = crossing_parts(k); crl = crossing_parts(k - 1)
    idx = {P: a for a, P in enumerate(crk)}; il = {q: a for a, q in enumerate(crl)}
    rot = lambda P: canon([set((x + 1) % k for x in B) for B in P])
    seen = set(); orbits = []
    for P in crk:
        if P in seen: continue
        orb = [P]; Q = rot(P)
        while Q != P: orb.append(Q); Q = rot(Q)
        seen.update(orb); orbits.append(orb)
    img = {}
    for P in crk:
        s, q = delete_leg(P, 0, k); m = merge(P, 0, 1, k); img[P] = (il.get(q, -1), s, il.get(m, -1))
    primes = [r for r in range(2, k + 1) if k % r == 0 and all(r % s for s in range(2, r))]
    g = next(x for x in range(2, p) if all(pow(x, (p - 1) // r, p) != 1 for r in set(primes + [2])))
    om = pow(g, (p - 1) // k, p)
    cols_all = []; chars = []; dims = []
    for j in range(k):
        cols = [o for o in orbits if (j * len(o)) % k == 0]
        V = []; M = np.zeros((2 * len(crl), len(cols)), dtype=np.int64)
        for c, orb in enumerate(cols):
            ent = []
            for t, Q in enumerate(orb):
                w = sum(pow(om, (-j * (t + u * len(orb))) % k, p) for u in range(k // len(orb))) % p
                ent.append((idx[Q], w)); a, s, b = img[Q]
                if a >= 0: M[a, c] = (M[a, c] + w * (N if s else 1)) % p
                if b >= 0: M[len(crl) + b, c] = (M[len(crl) + b, c] + w) % p
            V.append(ent)
        Nm = nullspace_mod(M, p); dims.append(Nm.shape[1])
        for b in range(Nm.shape[1]):
            v = np.zeros(len(crk), dtype=np.int64)
            for c in np.flatnonzero(Nm[:, b]):
                for a, w in V[c]: v[a] = (v[a] + int(Nm[c, b]) * w) % p
            cols_all.append(v); chars.append(j)
    W = np.array(cols_all, dtype=np.float64).T if cols_all else np.zeros((len(crk), 0))
    return crk, W, np.array(chars, dtype=int), dims

def check_kernel(crk, W, k, N, p):
    """Independent check: all leg-sums and merges of every basis vector have zero crossing part mod p."""
    crl = crossing_parts(k - 1); il = {q: a for a, q in enumerate(crl)}
    for i in range(k):
        for typ in 'dm':
            rows, cols, vals = [], [], []
            for a, P in enumerate(crk):
                if typ == 'd':
                    s, q = delete_leg(P, i, k); c = N if s else 1
                else:
                    q = merge(P, i, (i + 1) % k, k); c = 1
                if q in il: rows.append(il[q]); cols.append(a); vals.append(c)
            A = sp.csr_matrix((np.array(vals, dtype=np.float64), (rows, cols)), shape=(len(crl), len(crk)))
            if np.any(np.mod(A @ W, p)): return False
    return True

# ======================================================================= ribbons (asymmetric)
def outs_g(kx, ky, conf):
    s, t, j, o1, o2 = conf
    xs = [(s + i) % kx for i in range(j)]; ys = [(t + j - 1 - i) % ky for i in range(j)]
    outs = [('x', (s + j + i) % kx) for i in range(kx - j)]
    if o1: outs.append(('x', xs[0]))
    outs += [('y', (t + j + i) % ky) for i in range(ky - j)]
    if o2 and j > 1: outs.append(('x', xs[-1]))
    return xs, ys, outs

def pair_keys_g(Lp, nbp, kx, Lq, nbq, ky, conf, prow):
    """For partitions prow of the x side against all partitions of the y side: output partition code, closed loops."""
    xs, ys, outs = outs_g(kx, ky, conf)
    m = len(prow); n = Lq.shape[0]; T = kx + ky
    lab = np.empty((m, n, T), dtype=np.int8); lab[:] = np.arange(T, dtype=np.int8)
    lx = Lp[prow]; ly = Lq + kx
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
    rgs = np.zeros_like(oc); seen = np.full((m, n, T), -1, dtype=np.int8); cnt = np.zeros((m, n), dtype=np.int8)
    ii, jj = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
    for o in range(nout):
        lbl = oc[:, :, o].astype(np.int64); cur = seen[ii, jj, lbl]; new = cur < 0
        seen[ii[new], jj[new], lbl[new]] = cnt[new]; rgs[:, :, o] = np.where(new, cnt, cur); cnt = cnt + new
    code = np.zeros((m, n), dtype=np.int64)
    for o in range(nout): code = code * 16 + rgs[:, :, o]
    valid = np.zeros((m, n, T), dtype=bool)
    valid[:, :, :kx] = (np.arange(kx)[None, :] < nbp[prow][:, None])[:, None, :]
    valid[:, :, kx:] = (np.arange(ky)[None, :] < nbq[:, None])[None, :, :]
    isroot = valid & (lab == np.arange(T, dtype=np.int8)[None, None, :])
    closed = (isroot.sum(axis=2) - cnt).astype(np.int8)
    return code, closed, nout

def is_crossing(cache, u, nout):
    key = (int(u), nout)
    if key not in cache: cache[key] = nout >= 4 and not noncrossing(rgs_partition(decode(int(u), nout)))
    return cache[key]

def config_quadrics(L, nb, k, conf, Wy, N, p, cache, chunk=50, kbatch=64):
    """Ribbon conf on x (x) x (x with k legs): for every crossing output partition, the coefficients of
    y_a y_b (a <= b) of its coordinate, mod p.  Returns dict code -> float32 row."""
    n, d = Wy.shape
    iu0, iu1 = np.triu_indices(d); offd = (iu0 != iu1).astype(np.float64)
    Npow = np.array([pow(N, e, p) for e in range(2 * k + 2)], dtype=np.float64)
    acc = {}
    for c0 in range(0, n, chunk):
        prow = np.arange(c0, min(n, c0 + chunk)); m = len(prow)
        code, closed, nout = pair_keys_g(L, nb, k, L, nb, k, conf, prow)
        if nout < 4: return {}
        uc, inv = np.unique(code.ravel(), return_inverse=True)
        keep = np.array([i for i, u in enumerate(uc) if is_crossing(cache, u, nout)], dtype=np.int64)
        if len(keep) == 0: continue
        kmap = -np.ones(len(uc), dtype=np.int64); kmap[keep] = np.arange(len(keep))
        kidx = kmap[inv.ravel()]; sel = kidx >= 0
        P_ = np.repeat(np.arange(m), n)[sel]; Q_ = np.tile(np.arange(n), m)[sel]
        S = sp.csr_matrix((Npow[closed.ravel()[sel]], (kidx[sel] * m + P_, Q_)), shape=(len(keep) * m, n))
        SW = np.mod(S @ Wy, p).reshape(len(keep), m, d)
        WpT = Wy[prow].T
        for b0 in range(0, len(keep), kbatch):
            blk = SW[b0:b0 + kbatch]; nbk = blk.shape[0]
            G = np.mod(WpT @ blk.transpose(1, 0, 2).reshape(m, nbk * d), p).reshape(d, nbk, d).transpose(1, 0, 2)
            Tm = np.mod(G[:, iu0, iu1] + G[:, iu1, iu0] * offd[None, :], p).astype(np.float32)
            for t in range(nbk):
                u = int(uc[keep[b0 + t]])
                if u in acc: acc[u] = np.mod(acc[u] + Tm[t], p)
                else: acc[u] = Tm[t]
    return acc

def first_forms_g(Lc, nbc, k, conf, Wy, N, outP_index, p, chunk=100):
    """rho1(x (x) x) for x with k legs, all output partitions: Y[pi] = quadratic form (d x d, mod p)."""
    n, d = Wy.shape
    Npow = np.array([pow(N, e, p) for e in range(2 * k + 2)], dtype=np.float64)
    Y = np.zeros((len(outP_index), d, d))
    for c0 in range(0, n, chunk):
        prow = np.arange(c0, min(n, c0 + chunk))
        code, closed, nout = pair_keys_g(Lc, nbc, k, Lc, nbc, k, conf, prow)
        uc, inv = np.unique(code.ravel(), return_inverse=True)
        pidx = np.array([outP_index[rgs_partition(decode(int(u), nout))] for u in uc])
        P_ = np.repeat(prow, n); Q_ = np.tile(np.arange(n), len(prow))
        S = sp.csr_matrix((Npow[closed.ravel()], (inv.ravel() * len(prow) + (P_ - c0), Q_)), shape=(len(uc) * len(prow), n))
        SW = np.mod(S @ Wy, p).reshape(len(uc), len(prow), d); Wp = Wy[prow]
        for ui in range(len(uc)):
            Y[pidx[ui]] = np.mod(Y[pidx[ui]] + Wp.T @ SW[ui], p)
    return np.mod(Y + np.transpose(Y, (0, 2, 1)), p)

def second_cubics_g(Lout, nbout, k1, Lc, nbc, k, conf, Yflat, Wy, N, mmap, ncub, p, cache, chunk=60):
    """rho2 joining rho1(x (x) x) (k1 legs) with x (k legs): crossing coordinates, as rows over cubic monomials."""
    n, d = Wy.shape
    nz = np.flatnonzero(np.abs(Yflat).sum(axis=1) > 0)
    Npow = np.array([pow(N, e, p) for e in range(2 * k1 + 2)], dtype=np.float64)
    acc = {}
    for c0 in range(0, len(nz), chunk):
        prow = nz[c0:c0 + chunk]
        code, closed, nout = pair_keys_g(Lout, nbout, k1, Lc, nbc, k, conf, prow)
        if nout < 4: return []
        uc, inv = np.unique(code.ravel(), return_inverse=True)
        keep = np.array([i for i, u in enumerate(uc) if is_crossing(cache, u, nout)], dtype=np.int64)
        if len(keep) == 0: continue
        kmap = -np.ones(len(uc), dtype=np.int64); kmap[keep] = np.arange(len(keep))
        kidx = kmap[inv.ravel()]; sel = kidx >= 0
        P_ = np.repeat(np.arange(len(prow)), n)[sel]; Q_ = np.tile(np.arange(n), len(prow))[sel]
        S = sp.csr_matrix((Npow[closed.ravel()[sel]], (kidx[sel] * len(prow) + P_, Q_)), shape=(len(keep) * len(prow), n))
        SW = np.mod(S @ Wy, p).reshape(len(keep), len(prow), d)
        Yc = Yflat[prow]
        for ki in range(len(keep)):
            C = np.mod(Yc.T @ SW[ki], p); u = int(uc[keep[ki]])
            acc[u] = np.mod(acc[u] + C, p) if u in acc else C
    rows = []
    for u, C in acc.items():
        v = np.zeros(ncub); np.add.at(v, mmap.ravel(), C.ravel()); rows.append(np.mod(v, p))
    return rows

# ======================================================================= bookkeeping
class Log:
    def __init__(self, name):
        os.makedirs(OUT, exist_ok=True)
        self.f = os.path.join(OUT, name); self.t0 = time.time()
    def __call__(self, *a):
        s = time.strftime('%H:%M:%S') + '  ' + ' '.join(str(x) for x in a)
        print(s, flush=True)
        with open(self.f, 'a') as fh: fh.write(s + '\n')
def progress(line):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'progress.txt'), 'a') as fh: fh.write(time.strftime('%d %b %H:%M') + '  ' + line + '\n')
def load_json(path, default):
    try:
        with open(path) as fh: return json.load(fh)
    except Exception: return default
def save_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh: json.dump(obj, fh, indent=1)
    os.replace(tmp, path)

# ======================================================================= level 8
K8 = 8
def confs8(target):
    k = K8
    return [(s, t, j, o1, o2) for s in range(k) for t in range(k) for j in range(1, k + 1) for o1 in (0, 1) for o2 in (0, 1)
            if not (o2 and j == 1) and target(2 * k - 2 * j + o1 + o2)]

class Ring:
    """Polynomial constraints on the coordinates vars_ (rotation eigen-coordinates of K_8), split by character."""
    def __init__(self, ctx, vars_):
        self.c = ctx; self.vars = np.array(vars_); self.e = e = len(vars_)
        self.chars = ctx['chars'][self.vars]; self.Wy = ctx['Wy'][:, self.vars].copy()
        ch = self.chars
        self.period = next(m for m in (1, 2, 4, 8) if all((c * m) % 8 == 0 for c in ch))   # R^period x = x
        self.mon = {dg: list(itertools.combinations_with_replacement(range(e), dg)) for dg in (1, 2, 3, 4)}
        self.mchar = {dg: np.array([int(sum(ch[list(m)]) % 8) for m in self.mon[dg]]) for dg in (2, 3)}
        self.idx = {dg: {m: i for i, m in enumerate(self.mon[dg])} for dg in (2, 3)}
        self.sec3 = sorted(set(self.mchar[3].tolist()))
        self.col3 = {s: np.flatnonzero(self.mchar[3] == s) for s in self.sec3}
        self.loc3 = np.zeros(len(self.mon[3]), dtype=np.int64)
        for s in self.sec3: self.loc3[self.col3[s]] = np.arange(len(self.col3[s]))
        self.mul23 = np.array([[self.idx[3][tuple(sorted((a, b, i)))] for i in range(e)] for a, b in self.mon[2]])
        self.qspan = {}; self.blocks = {}; self.cubic_count = 0
    def quadrics(self):
        c = self.c; p = c['p']; e = self.e
        self.qspan = {chi: BatchRank(len(self.mon[2]), p) for chi in range(8)}
        qconfs = [(0, t, j, o1, o2) for t in range(self.period) for j in range(1, K8 + 1) for o1 in (0, 1) for o2 in (0, 1)
                  if not (o2 and j == 1) and 4 <= 2 * K8 - 2 * j + o1 + o2 <= K8 - 1]
        for conf in qconfs:
            acc = config_quadrics(c['L'], c['nb'], K8, conf, self.Wy, c['N'], p, c['cache'])
            if not acc: continue
            X = np.array(list(acc.values()), dtype=np.float64)
            for chi in range(8):
                Xc = np.where(self.mchar[2][None, :] == chi, X, 0.0)
                Xc = Xc[np.any(Xc != 0, axis=1)]
                if len(Xc): self.qspan[chi].add(Xc)
        self.blocks = {s: BatchRank(len(self.col3[s]), p) for s in self.sec3}
        for s in self.sec3:
            for i in range(e):
                chi = (s - self.chars[i]) % 8; Q = self.qspan[chi].B
                if Q.shape[0] == 0: continue
                pm = np.flatnonzero(self.mchar[2] == chi)
                R = np.zeros((Q.shape[0], len(self.col3[s]))); R[:, self.loc3[self.mul23[pm, i]]] = Q[:, pm]
                self.blocks[s].add(R)
    def add_cubics(self, rows):
        if not len(rows): return
        R = np.array(rows); self.cubic_count += len(rows)
        for s in self.sec3:
            Rc = R[:, self.col3[s]]; Rc = Rc[np.any(Rc != 0, axis=1)]
            if len(Rc): self.blocks[s].add(Rc)
    def gaps(self): return {int(s): self.blocks[s].gap for s in self.sec3}
    def closed(self): return all(b.gap == 0 for b in self.blocks.values())
    def cubes(self):
        out = []
        for a in range(self.e):
            s = (3 * self.chars[a]) % 8; v = np.zeros(len(self.col3[s])); v[self.loc3[self.idx[3][(a, a, a)]]] = 1
            if self.blocks[s].contains(v): out.append(a)
        return out
    def quartic_test(self):
        """Degree-4 Macaulay test: quadrics x quadratic monomials + cubic span x linear forms, by character."""
        p = self.c['p']; e = self.e; ch = self.chars
        m4 = self.mon[4]; c4 = np.array([int(sum(ch[list(m)]) % 8) for m in m4]); i4 = {m: i for i, m in enumerate(m4)}
        res = {}; fourth = set()
        for s in sorted(set(c4.tolist())):
            cols = np.flatnonzero(c4 == s); loc = -np.ones(len(m4), dtype=np.int64); loc[cols] = np.arange(len(cols))
            BR = BatchRank(len(cols), p)
            for m2i, m2 in enumerate(self.mon[2]):            # quadric (char chi) times monomial m2
                chi = (s - int(self.mchar[2][m2i])) % 8; Q = self.qspan[chi].B
                if Q.shape[0] == 0: continue
                pm = np.flatnonzero(self.mchar[2] == chi)
                tgt = loc[[i4[tuple(sorted(self.mon[2][q] + m2))] for q in pm]]
                R = np.zeros((Q.shape[0], len(cols))); R[:, tgt] = Q[:, pm]; BR.add(R)
                if BR.gap == 0: break
            for i in range(e):                                 # cubic span (char s - chi_i) times y_i
                if BR.gap == 0: break
                s3 = (s - ch[i]) % 8
                if s3 not in self.blocks: continue
                Bc = self.blocks[s3].B
                if Bc.shape[0] == 0: continue
                cm = self.col3[s3]
                tgt = loc[[i4[tuple(sorted(self.mon[3][q] + (i,)))] for q in cm]]
                for r0 in range(0, Bc.shape[0], 2000):
                    R = np.zeros((min(2000, Bc.shape[0] - r0), len(cols))); R[:, tgt] = Bc[r0:r0 + 2000]; BR.add(R)
            res[int(s)] = BR.gap
            for a in range(e):
                if (4 * ch[a]) % 8 == s:
                    v = np.zeros(len(cols)); v[loc[i4[(a, a, a, a)]]] = 1
                    if BR.contains(v): fourth.add(a)
            del BR
        return res, sorted(fourth)

def triple_family(ring, triangles=False):
    """All inequivalent triple contractions for the ring's symmetry (first ribbons anchored at leg 0, t mod period),
    in a fixed shuffled order, grouped by first ribbon: list of (first, [seconds])."""
    m = ring.period
    if not triangles:
        firsts = [c for c in confs8(lambda l: l == K8) if c[0] == 0 and c[1] < m]; k1 = K8
    else:
        firsts = [(0, t, j, o1, o2) for t in range(m) for (j, o1, o2) in [(4, 1, 0), (4, 0, 1), (3, 0, 0)]]
    fam = []
    for f in firsts:
        k1 = 2 * K8 - 2 * f[2] + f[3] + f[4]
        sec = [(s2, t2, j, o1, o2) for s2 in range(k1) for t2 in range(m) for j in range(1, K8 + 1) for o1 in (0, 1) for o2 in (0, 1)
               if not (o2 and j == 1) and 4 <= (k1 - j) + (K8 - j) + o1 + (o2 if j > 1 else 0) <= K8 - 1]
        fam.append((f, k1, sec))
    rng = random.Random(8)
    for f in fam: rng.shuffle(f[2])
    rng.shuffle(fam)
    return fam

def level8_job(N, p, deadline, logname):
    """Full level-8 chain at one (N, p).  Returns a summary dict (also written to tysonout/eight_N{N}_p{p}.json)."""
    log = Log(logname); t0 = time.time()
    out = os.path.join(OUT, f'eight_N{N}_p{p}.json')
    res = load_json(out, None)
    if res and res.get('status') in ('CLOSED', 'OPEN'): return res
    log(f'level 8, N={N}, p={p}: start')
    crk, Wy, chars, dims = eigenbasis(K8, N, p)
    ok = check_kernel(crk, Wy, K8, N, p)
    log(f'  K_8 eigenspace dims {dims} (total {sum(dims)}), kernel check {"OK" if ok else "FAILED"}  ({time.time()-t0:.0f}s)')
    allP = parts(K8)
    ctx = {'N': N, 'p': p, 'Wy': Wy, 'chars': chars, 'cache': {}}
    ctx['L'], ctx['nb'] = labels(crk, K8)
    ctx['Lall'], ctx['nball'] = labels(allP, K8); ctx['allP_index'] = {P: i for i, P in enumerate(allP)}
    res = {'N': N, 'p': p, 'dims': dims, 'kernel_check': ok, 'chain': [], 'status': 'RUNNING'}
    # odd rotation characters vanish by the half-turn theorem (Theorem 9.x, proved for 7 <= N <= 37): start in the even ring
    vars_ = [i for i in range(Wy.shape[1]) if chars[i] % 2 == 0]
    res['start'] = 'even ring (half-turn theorem)'
    big = {}
    while True:
        if time.time() > deadline: res['status'] = 'STOPPED (time)'; break
        ring = Ring(ctx, vars_); step = {'vars': len(vars_), 'chars': np.bincount(ring.chars, minlength=8).tolist(), 'period': ring.period}
        ring.quadrics()
        step['quadrics'] = {int(c): ring.qspan[c].rank for c in range(8) if ring.qspan[c].rank}
        step['gaps_quadrics'] = ring.gaps()
        log(f'  ring of {ring.e} coordinates (chars {step["chars"]}, invariant under R^{ring.period}): quadrics {step["quadrics"]}, degree-3 gaps {step["gaps_quadrics"]}  ({time.time()-t0:.0f}s)')
        killed = ring.cubes(); how = 'quadrics'; step['combos'] = 0; step['triangle_combos'] = 0
        for stage in ('triple', 'quartic', 'triangle'):
            if killed or ring.closed() or time.time() > deadline: break
            if stage in ('quartic', 'triangle') and ring.e > 26: break
            if stage == 'quartic':
                g4, fourth = ring.quartic_test(); step['quartic_gaps'] = g4; step['fourth_powers'] = len(fourth)
                log(f'  degree 4: gaps {g4}; 4th powers in ideal: {len(fourth)}  ({time.time()-t0:.0f}s)')
                if all(v == 0 for v in g4.values()): killed = list(range(ring.e)); how = 'degree 4 (full)'
                elif fourth: killed = fourth; how = 'degree 4'
                continue
            triangles = stage == 'triangle'
            stage_end = min(deadline, time.time() + TRIANGLE_MINUTES * 60) if triangles else deadline
            if triangles and K8 + 2 not in big:
                for kk in (K8 + 1, K8 + 2):
                    PP = parts(kk); big[kk] = (labels(PP, kk), {P: i for i, P in enumerate(PP)})
            fam = triple_family(ring, triangles)
            iu = np.triu_indices(ring.e); wgt = np.where(iu[0] == iu[1], 1.0, 2.0)
            mmap = np.array([[ring.idx[3][tuple(sorted((a, b, c)))] for c in range(ring.e)] for a, b in zip(*iu)])
            pos = [0] * len(fam); ncomb = 0
            while not killed and not ring.closed() and time.time() < stage_end and any(pos[i] < len(fam[i][2]) for i in range(len(fam))):
                for fi, (f, k1, sec) in enumerate(fam):          # round robin over first ribbons, 8 seconds at a time
                    if pos[fi] >= len(sec) or time.time() > stage_end: continue
                    if k1 == K8: Lout, nbout, oidx = ctx['Lall'], ctx['nball'], ctx['allP_index']
                    else: (Lout, nbout), oidx = big[k1]
                    Y = first_forms_g(ctx['L'], ctx['nb'], K8, f, ring.Wy, N, oidx, p)
                    Yflat = Y[:, iu[0], iu[1]] * wgt[None, :]; del Y
                    for c2 in sec[pos[fi]:pos[fi] + 8]:
                        ring.add_cubics(second_cubics_g(Lout, nbout, k1, ctx['L'], ctx['nb'], K8, c2, Yflat, ring.Wy, N, mmap, len(ring.mon[3]), p, ctx['cache']))
                        ncomb += 1
                    pos[fi] += 8; del Yflat
                    killed = ring.cubes()
                    if killed or ring.closed(): how = 'triangle contractions' if triangles else 'triple contractions'; break
            step['triangle_combos' if triangles else 'combos'] = ncomb
            log(f'  {"triangle" if triangles else "triple"} contractions: {ncomb} of {sum(len(f[2]) for f in fam)} used; gaps {ring.gaps()}; cubes in ideal: {len(killed)}  ({time.time()-t0:.0f}s)')
        step['gaps_final'] = ring.gaps()
        if ring.closed() and not killed: killed = list(range(ring.e)); how += ' (degree 3 full)'
        if killed:
            kc = sorted(set(int(ring.chars[a]) for a in killed))
            step['killed'] = len(killed); step['killed_chars'] = kc; step['how'] = how
            res['chain'].append(step)
            log(f'  -> {len(killed)} coordinates vanish (characters {kc}), by {how}')
            vars_ = [v for i, v in enumerate(vars_) if i not in set(killed)]
            if not vars_: res['status'] = 'CLOSED'; break
            continue
        res['chain'].append(step); res['status'] = 'OPEN' if time.time() < deadline else 'STOPPED (time)'; break
    res['minutes'] = round((time.time() - t0) / 60, 1)
    if res['status'] != 'STOPPED (time)': save_json(out, res)
    return res

def eight_worker(args):
    N, p, deadline = args
    try:
        r = level8_job(N, p, deadline, f'eight_N{N}.log')
    except Exception:
        Log(f'eight_N{N}.log')(traceback.format_exc()); return {'N': N, 'p': p, 'status': 'ERROR'}
    return r

def summary8(r):
    ch = ' -> '.join(str(s['vars']) for s in r.get('chain', []))
    if r['status'] == 'CLOSED': ch += ' -> 0'
    last = r['chain'][-1] if r.get('chain') else {}
    extra = '' if r['status'] == 'CLOSED' else f"  last gaps {last.get('gaps_final')}, degree 4 {last.get('quartic_gaps')}"
    return f"level 8  N={r['N']:<3d} p={r['p']}: {r['status']:<8s} coordinates {ch}  ({r.get('minutes', '?')} min){extra}"

LISTS = {'a': [11, 9, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37],
         'b': [12, 8, 10, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]}
def run_laptop(which):
    """Laptop 'a' or 'b': level 8 for its list of N (two at a time), then the other primes, then level 9."""
    t_end = time.time() + HOURS * 3600
    progress(f'=== laptop {which}: job started (stops after {HOURS:g} h) ===')
    Ns = LISTS[which]
    jobs = [(N, PRIMES[8][0]) for N in Ns] + [(N, PRIMES[8][1]) for N in Ns[:4]]
    done = set()
    if os.path.isdir(OUT):
        for f in os.listdir(OUT):
            if f.startswith('eight_N') and f.endswith('.json'):
                r = load_json(os.path.join(OUT, f), {})
                if r.get('status') in ('CLOSED', 'OPEN'): done.add((r['N'], r['p']))
    todo = [(N, p, t_end) for N, p in jobs if (N, p) not in done]
    if todo:
        with mp.Pool(WORKERS_EIGHT, maxtasksperchild=1) as pool:
            it = pool.imap_unordered(eight_worker, todo); left = len(todo)
            while left and time.time() < t_end + 300:
                try: r = it.next(timeout=30)
                except mp.TimeoutError: continue
                except StopIteration: break
                left -= 1
                if r.get('status') in ('CLOSED', 'OPEN'): progress(summary8(r))
                elif r.get('status') == 'ERROR': progress(f"level 8 N={r['N']}: ERROR, see tysonout/eight_N{r['N']}.log")
            pool.terminate()
    if time.time() < t_end:
        log = Log('nine.log')
        for N in ([11, 13] if which == 'a' else [13, 11]):
            if time.time() > t_end: break
            try: level9_N(N, PRIMES[9][0], t_end, log)
            except Exception: log(traceback.format_exc()); progress(f'level 9 N={N}: ERROR, see tysonout/nine.log')
    progress(f'=== laptop {which}: time is up ===' if time.time() > t_end else f'=== laptop {which}: everything done ===')

# ======================================================================= level 9
K9 = 9
def nine_configs():
    k = K9
    c = [(0, t, j, o1, o2) for t in range(k) for j in range(1, k + 1) for o1 in (0, 1) for o2 in (0, 1)
         if not (o2 and j == 1) and 4 <= 2 * k - 2 * j + o1 + o2 <= k - 1]
    c.sort(key=lambda x: (-(2 * k - 2 * x[2] + x[3] + x[4]), x))   # biggest outputs first
    return c

def nine_worker(args):
    N, p, conf, basisfile, tmpdir, deadline = args
    try:
        if time.time() > deadline: return conf, None
        z = np.load(basisfile); Wy = z['W']
        crk = crossing_parts(K9); L, nb = labels(crk, K9)
        acc = config_quadrics(L, nb, K9, conf, Wy, N, p, {})
        f = os.path.join(tmpdir, 'conf_' + '_'.join('abcdefghij'[x] for x in conf) + '.npy')
        D = Wy.shape[1] * (Wy.shape[1] + 1) // 2; X = np.empty((len(acc), D), dtype=np.float32)
        for i, kk in enumerate(list(acc)): X[i] = acc.pop(kk)
        np.save(f + '.part.npy', X); os.replace(f + '.part.npy', f)
        return conf, f
    except Exception:
        Log(f'nine_N{N}.log')(traceback.format_exc()); return conf, None

def level9_N(N, p, t_end, log):
    tag = f'N{N}_p{p}'
    stfile = os.path.join(OUT, f'nine_{tag}.json')
    st = load_json(stfile, {'N': N, 'p': p, 'merged': [], 'status': 'RUNNING'})
    if st['status'] in ('CLOSED', 'OPEN'): return st
    basisfile = os.path.join(OUT, f'nine_basis_{tag}.npz')
    t0 = time.time()
    if not os.path.exists(basisfile):
        log(f'level 9, N={N}, p={p}: computing K_9 (linear step)')
        crk, W, chars, dims = eigenbasis(K9, N, p)
        ok = check_kernel(crk, W, K9, N, p)
        np.savez(basisfile + '.part.npz', W=W, chars=chars); os.replace(basisfile + '.part.npz', basisfile)
        st['dims'] = dims; st['kernel_check'] = ok; save_json(stfile, st)
        progress(f'level 9  N={N:<3d} p={p}: dim K_9 = {sum(dims)}, by rotation character {dims}, kernel check {"OK" if ok else "FAILED"}  ({(time.time()-t0)/60:.0f} min)')
    z = np.load(basisfile); chars = z['chars']; d = len(chars)
    iu0, iu1 = np.triu_indices(d); mch = (chars[iu0] + chars[iu1]) % K9
    secs = {s: np.flatnonzero(mch == s) for s in range(K9)}
    statefile = os.path.join(OUT, f'nine_state_{tag}.npz')
    BR = {s: BatchRank(len(secs[s]), p) for s in range(K9)}
    if os.path.exists(statefile) and st['merged']:
        zz = np.load(statefile)
        for s in range(K9):
            BR[s].B = zz[f'B{s}'].astype(np.float64); BR[s].piv = [int(x) for x in zz[f'piv{s}']]
    else: st['merged'] = []
    confs = [c for c in nine_configs() if list(c) not in st['merged']]
    log(f'level 9, N={N}, p={p}: dim K_9 = {d}; {len(confs)} ribbon configurations to go; current gaps {[BR[s].gap for s in range(K9)]}')
    tmpdir = os.path.join(OUT, f'nine_tmp_{tag}'); os.makedirs(tmpdir, exist_ok=True)
    def savestate():
        np.savez(statefile + '.part.npz', **{f'B{s}': BR[s].B.astype(np.float32) for s in range(K9)}, **{f'piv{s}': np.array(BR[s].piv, dtype=np.int64) for s in range(K9)})
        os.replace(statefile + '.part.npz', statefile); save_json(stfile, st)
    nmerged = 0
    with mp.Pool(WORKERS_NINE, maxtasksperchild=4) as pool:
        it = pool.imap(nine_worker, [(N, p, c, basisfile, tmpdir, t_end) for c in confs]); left = len(confs)
        while left and time.time() < t_end + 300:
            try: conf, f = it.next(timeout=30)
            except mp.TimeoutError: continue
            except StopIteration: break
            left -= 1
            if f is None:
                if time.time() > t_end: break
                continue
            X = np.load(f)
            for s in range(K9):
                if BR[s].gap == 0: continue
                Xs = X[:, secs[s]].astype(np.float64); Xs = Xs[np.any(Xs != 0, axis=1)]
                for r0 in range(0, len(Xs), 3000): BR[s].add(Xs[r0:r0 + 3000])
            del X; os.remove(f)
            st['merged'].append(list(conf)); nmerged += 1
            gaps = [BR[s].gap for s in range(K9)]
            log(f'  config {conf}: gaps by character {gaps}  ({(time.time()-t0)/60:.0f} min)')
            if nmerged % 3 == 0 or all(g == 0 for g in gaps): savestate()
            if all(g == 0 for g in gaps): break
            if time.time() > t_end: break
        pool.terminate()
    gaps = [BR[s].gap for s in range(K9)]
    st['gaps'] = gaps; st['ranks'] = [BR[s].rank for s in range(K9)]
    if all(g == 0 for g in gaps): st['status'] = 'CLOSED'
    elif len(st['merged']) == len(nine_configs()): st['status'] = 'OPEN'
    savestate()
    if st['status'] in ('CLOSED', 'OPEN'):
        msg = 'quadrics span ALL quadratic forms: C cannot first differ at level 9' if st['status'] == 'CLOSED' else 'quadrics do not span everything'
        progress(f"level 9  N={N:<3d} p={p}: {st['status']}, {len(st['merged'])} configurations, gaps {gaps}: {msg}")
        for f in os.listdir(tmpdir): os.remove(os.path.join(tmpdir, f))
    return st

def run_nine():
    t_end = time.time() + HOURS * 3600; log = Log('nine.log')
    progress(f'=== level 9 job started (stops after {HOURS:g} h) ===')
    for N, p in [(11, PRIMES[9][0]), (13, PRIMES[9][0]), (11, PRIMES[9][1]), (10, PRIMES[9][0]), (12, PRIMES[9][0]),
                 (14, PRIMES[9][0]), (15, PRIMES[9][0]), (16, PRIMES[9][0]), (17, PRIMES[9][0])]:
        if time.time() > t_end: break
        try: level9_N(N, p, t_end, log)
        except Exception: log(traceback.format_exc()); progress(f'level 9 N={N}: ERROR, see tysonout/nine.log')
    progress('=== level 9 job: time is up ===' if time.time() > t_end else '=== level 9 job: all done ===')

# ======================================================================= self-test
def run_test():
    t0 = time.time(); ok = True
    def say(s): print(f'[{time.time()-t0:5.0f}s] {s}', flush=True)
    say(f'python {sys.version.split()[0]}, numpy {np.__version__}, cores {os.cpu_count()}')
    for k, N, want in [(6, 11, 5), (7, 11, 14), (8, 11, 66)]:
        p = PRIMES[8][0] if k == 8 else 1048573
        crk, W, ch, dims = eigenbasis(k, N, p); good = sum(dims) == want and check_kernel(crk, W, k, N, p)
        ok &= good; say(f'dim K_{k}({N}) = {sum(dims)} {dims}  {"OK" if good else "WRONG"}')
    k = 7; N = 11; p = 1048573
    crk, W, ch, dims = eigenbasis(k, N, p); L, nb = labels(crk, k); d = W.shape[1]
    iu0, iu1 = np.triu_indices(d); mch = (ch[iu0] + ch[iu1]) % k
    BR = {s: BatchRank(int((mch == s).sum()), p) for s in range(k)}; cache = {}
    for c in [(0, t, j, o1, o2) for t in range(k) for j in range(1, k + 1) for o1 in (0, 1) for o2 in (0, 1)
              if not (o2 and j == 1) and 4 <= 2 * k - 2 * j + o1 + o2 <= k - 1]:
        acc = config_quadrics(L, nb, k, c, W, N, p, cache)
        if acc:
            X = np.array(list(acc.values()), dtype=np.float64)
            for s in range(k): BR[s].add(X[:, mch == s])
    r = sum(BR[s].rank for s in range(k)); ok &= (r == 105)
    say(f'level 7 quadrics span {r} of 105  {"OK" if r == 105 else "WRONG"}')
    t1 = time.time(); crk = crossing_parts(9); L9, nb9 = labels(crk, 9)
    code, closed, nout = pair_keys_g(L9, nb9, 9, L9, nb9, 9, (0, 0, 5, 0, 0), np.arange(50))
    say(f'level 9 speed: {time.time()-t1:.1f}s per 50 rows (a configuration has {len(crk)} rows)')
    say('SELF-TEST PASSED' if ok else 'SELF-TEST FAILED -- send me the output')

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else ''
    if what == 'test': run_test()
    elif what in ('a', 'b'): run_laptop(what)
    elif what == 'nine': run_nine()
    else: print(__doc__ if __doc__ else open(__file__).read().split('import sys')[0])
