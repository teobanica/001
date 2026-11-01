#!/usr/bin/env python3
# tiger.py -- level 9, single-character slices attacked to the end, for "On the maximality conjecture for quantum
# permutation groups" (Claude and Teo).  Successor of kitty.py (and tyson.py).
#
#   python3 tiger.py test     self-test at level 8 (about 20-40 minutes); run this first on each laptop
#   python3 tiger.py a        on laptop A (level 9, N = 11): slice j = 1, then j = 0
#   python3 tiger.py b        on laptop B (level 9, N = 13): slice j = 3, then j = 0
#   python3 tiger.py a 14     the same, but stopping after 14 hours instead of 7 (any number of hours works)
#
# Put tiger.py in the SAME folder as tyson.py and kitty.py: it reads the level-9 data that tyson.py saved in
# tysonout/ (basis of K_9 and the span of the quadrics).  Results go to  tigerout/progress.txt  (keep that file,
# and tigerout/tiger_K9_N11.log, resp. N13, if you can).
#
# What is new with respect to kitty.py.
#   * Only one representative per Galois orbit: the slices j and aj (a = 1, 2, 4, 5, 7, 8 mod 9) behave the same,
#     so j = 1 stands for 1, 2, 4, 5, 7, 8, and j = 3 for 3, 6.  kitty.py did the same slice three times.
#   * No time limit per slice: a slice is worked on until it closes, or the family of triple contractions is used up.
#   * Saving: the cubic span of the slice in progress is saved every few minutes in tigerout/.  Closing the terminal
#     (or the lid) at any time is harmless: run the same command again, even another night, and it continues
#     from the last save (at most about 10 minutes are lost).
#   * The triple contractions are no longer taken in a fixed order: tiger keeps track of which first ribbons have
#     been productive, and mostly uses those ("epsilon-greedy"), and it keeps their quadratic forms in memory.
#
# The job stops by itself after HOURS hours.
import sys, os, time, json, itertools, random, traceback
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')
import numpy as np
import scipy.sparse as sp

HOURS = 7.0
SLICE_HOURS = 2.5          # maximal time spent on one slice in step 2
CAP3 = 9500                # largest tracked degree-3 sector (number of cubic monomials) in a slice (memory)
PRIMES = {8: 1048433, 9: 1048573}
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'tigerout')
TYSONOUT = os.path.join(HERE, 'tysonout')

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
    Y = np.zeros((len(outP_index), d, d), dtype=np.float32)
    for c0 in range(0, n, chunk):
        prow = np.arange(c0, min(n, c0 + chunk))
        code, closed, nout = pair_keys_g(Lc, nbc, k, Lc, nbc, k, conf, prow)
        uc, inv = np.unique(code.ravel(), return_inverse=True)
        pidx = np.array([outP_index[rgs_partition(decode(int(u), nout))] for u in uc])
        P_ = np.repeat(prow, n); Q_ = np.tile(np.arange(n), len(prow))
        S = sp.csr_matrix((Npow[closed.ravel()], (inv.ravel() * len(prow) + (P_ - c0), Q_)), shape=(len(uc) * len(prow), n))
        SW = np.mod(S @ Wy, p).reshape(len(uc), len(prow), d); Wp = Wy[prow]
        for ui in range(len(uc)):
            Y[pidx[ui]] = np.mod(Y[pidx[ui]] + np.mod(Wp.T @ SW[ui], p), p)
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
        if nout < 4: return
        uc, inv = np.unique(code.ravel(), return_inverse=True)
        keep = np.array([i for i, u in enumerate(uc) if is_crossing(cache, u, nout)], dtype=np.int64)
        if len(keep) == 0: continue
        kmap = -np.ones(len(uc), dtype=np.int64); kmap[keep] = np.arange(len(keep))
        kidx = kmap[inv.ravel()]; sel = kidx >= 0
        P_ = np.repeat(np.arange(len(prow)), n)[sel]; Q_ = np.tile(np.arange(n), len(prow))[sel]
        S = sp.csr_matrix((Npow[closed.ravel()[sel]], (kidx[sel] * len(prow) + P_, Q_)), shape=(len(keep) * len(prow), n))
        SW = np.mod(S @ Wy, p).reshape(len(keep), len(prow), d)
        Yc = Yflat[prow].astype(np.float64)
        for ki in range(len(keep)):
            C = np.mod(Yc.T @ SW[ki], p).astype(np.float32); u = int(uc[keep[ki]])
            acc[u] = np.mod(acc[u] + C, p) if u in acc else C
    batch = []
    for u in list(acc):
        C = acc.pop(u); v = np.zeros(ncub); np.add.at(v, mmap.ravel(), C.ravel().astype(np.float64)); batch.append(np.mod(v, p))
        if len(batch) == 200: yield batch; batch = []
    if batch: yield batch

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

# ======================================================================= kitty: small helpers
def divisors(K): return [m for m in range(1, K + 1) if K % m == 0]

def rank_mod(X, p):
    """Rank of X mod p (vectorized Gaussian elimination, iterating over the smaller dimension)."""
    M = np.mod(np.asarray(X, dtype=np.float64), p)
    if M.shape[0] > M.shape[1]: M = M.T.copy()
    rows, cols = M.shape; r = 0
    for c in range(cols):
        if r == rows: break
        nz = np.flatnonzero(M[r:, c])
        if len(nz) == 0: continue
        pr = r + nz[0]
        if pr != r: M[[r, pr]] = M[[pr, r]]
        M[r] = np.mod(M[r] * pow(int(M[r, c]), p - 2, p), p)
        bel = r + 1 + np.flatnonzero(M[r + 1:, c])
        if len(bel): M[bel] = np.mod(M[bel] - np.mod(np.outer(M[bel, c], M[r]), p), p)
        r += 1
    return r

def anchored_configs(K):
    c = [(0, t, j, o1, o2) for t in range(K) for j in range(1, K + 1) for o1 in (0, 1) for o2 in (0, 1)
         if not (o2 and j == 1) and 4 <= 2 * K - 2 * j + o1 + o2 <= K - 1]
    c.sort(key=lambda x: (-(2 * K - 2 * x[2] + x[3] + x[4]), x))
    return c

# ======================================================================= the quadric span of the full K_k
def full_state(K, N, p, log, t_end):
    """Basis W of K_K(N) mod p (rotation eigenbasis, characters chars) and, for each character s, a row basis B[s]
    of the span of all quadrics, in the monomials y_a y_b (a <= b) of character s, in numpy.triu_indices order.
    At level 9 this is read from tyson.py's output if present; otherwise it is computed (and saved, resumably)."""
    tag = f'N{N}_p{p}'
    if K == 9:
        bf = os.path.join(TYSONOUT, f'nine_basis_{tag}.npz'); sf = os.path.join(TYSONOUT, f'nine_state_{tag}.npz')
        js = load_json(os.path.join(TYSONOUT, f'nine_{tag}.json'), {})
        if os.path.exists(bf) and os.path.exists(sf) and js.get('status') in ('OPEN', 'CLOSED'):
            z = np.load(bf); zz = np.load(sf)
            log(f'level {K}, N={N}: read basis and quadric span from tyson.py output ({len(js.get("merged", []))} configurations)')
            return z['W'], z['chars'].astype(int), [zz[f'B{s}'].astype(np.float64) for s in range(K)]
    bf = os.path.join(OUT, f'basis_K{K}_{tag}.npz'); sf = os.path.join(OUT, f'state_K{K}_{tag}.npz')
    jf = os.path.join(OUT, f'state_K{K}_{tag}.json')
    if not os.path.exists(bf):
        t0 = time.time(); log(f'level {K}, N={N}: computing K_{K} (linear step)')
        crk, W, chars, dims = eigenbasis(K, N, p); ok = check_kernel(crk, W, K, N, p)
        np.savez(bf + '.part.npz', W=W, chars=chars); os.replace(bf + '.part.npz', bf)
        progress(f'level {K}  N={N}: dim K_{K} = {sum(dims)}, by rotation character {dims}, kernel check {"OK" if ok else "FAILED"}  ({(time.time()-t0)/60:.0f} min)')
    z = np.load(bf); W = z['W']; chars = z['chars'].astype(int); d = len(chars)
    iu0, iu1 = np.triu_indices(d); mch = (chars[iu0] + chars[iu1]) % K
    secs = [np.flatnonzero(mch == s) for s in range(K)]
    st = load_json(jf, {'merged': []}); BR = [BatchRank(len(secs[s]), p) for s in range(K)]
    if os.path.exists(sf) and st['merged']:
        zz = np.load(sf)
        for s in range(K): BR[s].B = zz[f'B{s}'].astype(np.float64); BR[s].piv = [int(x) for x in zz[f'piv{s}']]
    else: st['merged'] = []
    confs = [c for c in anchored_configs(K) if list(c) not in st['merged']]
    if confs: log(f'level {K}, N={N}: quadric step, {len(confs)} ribbon configurations to go')
    crk = crossing_parts(K); L, nb = labels(crk, K); cache = {}; t0 = time.time()
    for conf in confs:
        if time.time() > t_end: return None
        acc = config_quadrics(L, nb, K, conf, W, N, p, cache)
        if acc:
            X = np.array(list(acc.values()), dtype=np.float64); acc = None
            for s in range(K):
                Xs = X[:, secs[s]]; Xs = Xs[np.any(Xs != 0, axis=1)]
                for r0 in range(0, len(Xs), 3000): BR[s].add(Xs[r0:r0 + 3000])
            del X
        st['merged'].append(list(conf))
        np.savez(sf + '.part.npz', **{f'B{s}': BR[s].B.astype(np.float32) for s in range(K)}, **{f'piv{s}': np.array(BR[s].piv, dtype=np.int64) for s in range(K)})
        os.replace(sf + '.part.npz', sf); save_json(jf, st)
        log(f'  config {conf}: quadric gaps by character {[BR[s].gap for s in range(K)]}  ({(time.time()-t0)/60:.0f} min)')
    return W, chars, [BR[s].B for s in range(K)]

def make_ctx(K, N, p, W, chars, B):
    d = len(chars); iu0, iu1 = np.triu_indices(d); mch = (chars[iu0] + chars[iu1]) % K
    secpos = np.full(len(iu0), -1, dtype=np.int64)
    for s in range(K): secpos[np.flatnonzero(mch == s)] = np.arange(int((mch == s).sum()))
    crk = crossing_parts(K); allP = parts(K)
    ctx = {'K': K, 'N': N, 'p': p, 'W': W, 'chars': chars, 'B': B, 'd': d, 'secpos': secpos, 'cache': {}}
    ctx['L'], ctx['nb'] = labels(crk, K)
    ctx['Lall'], ctx['nball'] = labels(allP, K); ctx['allP_index'] = {P: i for i, P in enumerate(allP)}
    return ctx

def gmon(d, a, b):
    """Index of y_a y_b (a <= b, arrays) in numpy.triu_indices(d) order."""
    return a * d - a * (a - 1) // 2 + (b - a)

# ======================================================================= slices
class Slice:
    """Candidates restricted to x = Wy t, where the columns of Wy are either coordinates of K_K (a coordinate slice,
    A = None) or combinations A of coordinates of a single character each (a random rotation-adapted slice).
    Same machinery as the Ring of tyson.py, for any level K, with the quadrics restricted from the full span."""
    def __init__(self, ctx, vars_=None, A=None, achars=None, light=False):
        self.c = ctx; K = ctx['K']; self.K = K
        if A is None:
            self.vars = np.array(sorted(vars_)); self.A = None
            self.chars = ctx['chars'][self.vars]; self.Wy = ctx['W'][:, self.vars].copy()
        else:
            self.vars = None; self.A = np.mod(A, ctx['p']); self.chars = np.array(achars)
            self.Wy = np.mod(ctx['W'] @ self.A, ctx['p'])
        self.e = e = len(self.chars); ch = self.chars
        self.period = next(m for m in divisors(K) if all((c * m) % K == 0 for c in ch))
        if light:
            self.mon = {2: list(itertools.combinations_with_replacement(range(e), 2))}
            self.mchar = {2: np.array([int(sum(ch[list(m)]) % K) for m in self.mon[2]], dtype=np.int64)}
            return
        self.mon = {dg: list(itertools.combinations_with_replacement(range(e), dg)) for dg in (1, 2, 3)}
        self.mchar = {dg: np.array([int(sum(ch[list(m)]) % K) for m in self.mon[dg]], dtype=np.int64) for dg in (2, 3)}
        self.idx = {dg: {m: i for i, m in enumerate(self.mon[dg])} for dg in (2, 3)}
        # only the degree-3 sectors containing cubes y_a^3 are tracked: that is all the cube test needs
        self.sec3 = sorted(set(int((3 * c) % K) for c in ch))
        self.col3 = {s: np.flatnonzero(self.mchar[3] == s) for s in self.sec3}
        self.loc3 = np.zeros(len(self.mon[3]), dtype=np.int64)
        for s in self.sec3: self.loc3[self.col3[s]] = np.arange(len(self.col3[s]))
        self.mul23 = np.array([[self.idx[3][tuple(sorted((a, b, i)))] for i in range(e)] for a, b in self.mon[2]], dtype=np.int64)
        self.qspan = {}; self.blocks = {}; self.cubic_count = 0
    def max_sector3(self): return max(len(v) for v in self.col3.values())
    def restricted_quadrics(self, chi):
        """Rows: the full quadric span of character chi, restricted to the slice, in the slice monomials of character chi."""
        c = self.c; p = c['p']; d = c['d']; B = c['B'][chi]
        loc = np.flatnonzero(self.mchar[2] == chi)
        if B.shape[0] == 0 or len(loc) == 0: return np.zeros((0, len(loc)))
        m2 = np.array([self.mon[2][i] for i in loc])
        if self.A is None:
            g = gmon(d, self.vars[m2[:, 0]], self.vars[m2[:, 1]])
            return B[:, c['secpos'][g]]
        # random slice: q(A t) for q in the span; monomials of the full space of character chi
        iu0, iu1 = np.triu_indices(d); sel = np.flatnonzero((c['chars'][iu0] + c['chars'][iu1]) % self.K == chi)
        a, b = iu0[sel], iu1[sel]; A = self.A
        u, v = m2[:, 0], m2[:, 1]
        # coefficient of t_u t_v in y_a y_b with y = A t:  A[a,u]A[b,v] + A[a,v]A[b,u]  (u < v),  A[a,u]A[b,u]  (u = v)
        KR = np.mod(A[a][:, u] * A[b][:, v] + np.where(u != v, 1, 0)[None, :] * A[a][:, v] * A[b][:, u], p)
        out = np.zeros((B.shape[0], len(loc)))
        for r0 in range(0, len(sel), 4000):
            out = np.mod(out + B[:, r0:r0 + 4000] @ KR[r0:r0 + 4000], p)
        return out
    def quadrics(self):
        p = self.c['p']; e = self.e; K = self.K
        self.qspan = {}
        for chi in sorted(set(self.mchar[2].tolist())):
            R = self.restricted_quadrics(chi); R = R[np.any(R != 0, axis=1)]
            br = BatchRank(int((self.mchar[2] == chi).sum()), p)
            for r0 in range(0, len(R), 2000): br.add(R[r0:r0 + 2000])
            self.qspan[chi] = br
        self.blocks = {s: BatchRank(len(self.col3[s]), p) for s in self.sec3}
        for s in self.sec3:
            for i in range(e):
                chi = (s - self.chars[i]) % K
                if chi not in self.qspan: continue
                Q = self.qspan[chi].B
                if Q.shape[0] == 0: continue
                pm = np.flatnonzero(self.mchar[2] == chi)
                R = np.zeros((Q.shape[0], len(self.col3[s]))); R[:, self.loc3[self.mul23[pm, i]]] = Q
                self.blocks[s].add(R)
    def degree2_closed(self): return all(b.gap == 0 for b in self.qspan.values())
    def degree2_test(self):
        """Do the restricted quadrics span all quadratic forms on the slice?  (Then the slice closes.)"""
        p = self.c['p']
        for chi in sorted(set(self.mchar[2].tolist())):
            n = int((self.mchar[2] == chi).sum()); R = self.restricted_quadrics(chi)
            if R.shape[0] < n or rank_mod(R, p) < n: return False
        return True
    def quartic_test(self):
        """Degree 4: quadrics x quadratic monomials + cubic span x linear forms, by character (small slices only)."""
        p = self.c['p']; e = self.e; ch = self.chars; K = self.K
        m4 = list(itertools.combinations_with_replacement(range(e), 4))
        c4 = np.array([int(sum(ch[list(m)]) % K) for m in m4]); i4 = {m: i for i, m in enumerate(m4)}
        res = {}; fourth = set()
        for s in sorted(set(c4.tolist())):
            cols = np.flatnonzero(c4 == s); loc = -np.ones(len(m4), dtype=np.int64); loc[cols] = np.arange(len(cols))
            BR = BatchRank(len(cols), p)
            for m2i, m2 in enumerate(self.mon[2]):
                chi = (s - int(self.mchar[2][m2i])) % K
                if chi not in self.qspan or self.qspan[chi].B.shape[0] == 0: continue
                Q = self.qspan[chi].B; pm = np.flatnonzero(self.mchar[2] == chi)
                tgt = loc[[i4[tuple(sorted(self.mon[2][q] + m2))] for q in pm]]
                R = np.zeros((Q.shape[0], len(cols))); R[:, tgt] = Q; BR.add(R)
                if BR.gap == 0: break
            for i in range(e):
                if BR.gap == 0: break
                s3 = (s - ch[i]) % K
                if s3 not in self.blocks or self.blocks[s3].B.shape[0] == 0: continue
                Bc = self.blocks[s3].B; cm = self.col3[s3]
                tgt = loc[[i4[tuple(sorted(self.mon[3][q] + (i,)))] for q in cm]]
                for r0 in range(0, Bc.shape[0], 2000):
                    R = np.zeros((min(2000, Bc.shape[0] - r0), len(cols))); R[:, tgt] = Bc[r0:r0 + 2000]; BR.add(R)
            res[int(s)] = BR.gap
            for a in range(e):
                if (4 * ch[a]) % K == s:
                    v = np.zeros(len(cols)); v[loc[i4[(a, a, a, a)]]] = 1
                    if BR.contains(v): fourth.add(a)
            del BR
        return res, sorted(fourth)
    def add_cubics(self, rows):
        if not len(rows): return
        R = np.array(rows); self.cubic_count += len(rows)
        for s in self.sec3:
            if self.blocks[s].gap == 0: continue
            Rc = R[:, self.col3[s]]; Rc = Rc[np.any(Rc != 0, axis=1)]
            if len(Rc): self.blocks[s].add(Rc)
    def gaps(self): return {int(s): self.blocks[s].gap for s in self.sec3}
    def closed(self): return all(b.gap == 0 for b in self.blocks.values())
    def cubes(self):
        out = []
        for a in range(self.e):
            s = (3 * self.chars[a]) % self.K; v = np.zeros(len(self.col3[s])); v[self.loc3[self.idx[3][(a, a, a)]]] = 1
            if self.blocks[s].contains(v): out.append(a)
        return out

def triple_family(sl):
    """Inequivalent triple contractions for the slice's symmetry: first ribbons anchored at leg 0 landing at level K,
    t mod the period; second ribbons landing at levels 4..K-1.  Fixed shuffled order, grouped by first ribbon."""
    K = sl.K; m = sl.period
    firsts = [(0, t, j, o1, o2) for t in range(m) for j in range(1, K + 1) for o1 in (0, 1) for o2 in (0, 1)
              if not (o2 and j == 1) and 2 * K - 2 * j + o1 + o2 == K]
    fam = []
    for f in firsts:
        k1 = K
        sec = [(s2, t2, j, o1, o2) for s2 in range(k1) for t2 in range(m) for j in range(1, K + 1) for o1 in (0, 1) for o2 in (0, 1)
               if not (o2 and j == 1) and 4 <= (k1 - j) + (K - j) + o1 + (o2 if j > 1 else 0) <= K - 1]
        fam.append((f, k1, sec))
    rng = random.Random(9)
    for f in fam: rng.shuffle(f[2])
    rng.shuffle(fam)
    return fam


def support_vars(ctx, S): return np.flatnonzero(np.isin(ctx['chars'], sorted(S)))

# ======================================================================= tiger
QUEUES = {'a': (11, [[1], [0]]), 'b': (13, [[3], [0]])}
EPS = 0.25            # fraction of exploratory picks (least used first ribbon)
PICK = 2              # second ribbons per pick
EMA = 0.5             # weight of the last pick in the score of a first ribbon
YCACHE_MB = 1500      # memory for the cached quadratic forms rho1(x (x) x) of the first ribbons
SAVE_EVERY = 480      # seconds between saves
REPORT_EVERY = 3600   # seconds between status lines in progress.txt
QUARTIC_MAX = 26      # degree-4 test for rings with at most this many coordinates
QUARTIC_EVERY = 8     # ... every so many picks, when the ring has at most 16 coordinates

def total_gap(sl): return sum(b.gap for b in sl.blocks.values())

class Tiger:
    """One slice, attacked as at level 8 (quadrics, cubes in the ideal, triple contractions, degree 4 for small
    rings), with saving and an adaptive choice of the triple contractions."""
    def __init__(self, ctx, S, log, tagbase):
        self.ctx = ctx; self.S = sorted(S); self.log = log; self.name = 'chars ' + ''.join(str(x) for x in self.S)
        self.tag = f"{tagbase}_c{''.join(str(x) for x in self.S)}"
        self.jf = os.path.join(OUT, f'slice_{self.tag}.json'); self.nf = os.path.join(OUT, f'slice_{self.tag}.npz')
        st = load_json(self.jf, None)
        if st is None:
            st = {'slice': self.S, 'vars': support_vars(ctx, S).tolist(), 'chain': [], 'status': 'RUNNING',
                  'pos': None, 'stats': None, 'ncomb': 0, 'ring_ncomb': 0, 'picks': 0, 'seconds': 0.0,
                  'ring_gaps_quadrics': None, 'saved': False, 'gaps': None}
        for key in ('ring_gaps_quadrics', 'gaps'):          # JSON turns the integer keys into strings
            if st.get(key): st[key] = {int(s): v for s, v in st[key].items()}
        self.st = st
        if st['status'] == 'RUNNING': self.new_ring(load=st['saved'])

    def new_ring(self, load):
        st, ctx = self.st, self.ctx
        self.sl = sl = Slice(ctx, vars_=np.array(st['vars'], dtype=np.int64))
        sl.quadrics()
        if st['ring_gaps_quadrics'] is None: st['ring_gaps_quadrics'] = sl.gaps()
        self.fam = triple_family(sl)
        if st['pos'] is None:
            st['pos'] = [0] * len(self.fam); st['stats'] = [[0, None] for _ in self.fam]
        if load and os.path.exists(self.nf):
            z = np.load(self.nf)
            for s in sl.sec3:
                sl.blocks[s].B = z[f'B{s}'].astype(np.float64); sl.blocks[s].piv = [int(x) for x in z[f'piv{s}']]
            self.log(f'  [{self.name}] resumed from the last save: {sl.e} coordinates, gaps {sl.gaps()}, '
                     f'{st["ring_ncomb"]} triple contractions already used in this ring')
        iu = np.triu_indices(sl.e); self.iu = iu
        self.wgt = np.where(iu[0] == iu[1], 1.0, 2.0).astype(np.float32)
        self.mmap = np.array([[sl.idx[3][tuple(sorted((a, b, c)))] for c in range(sl.e)] for a, b in zip(*iu)], dtype=np.int64)
        self.ycache = {}
        self.log(f'  [{self.name}] ring of {sl.e} coordinates (invariant under R^{sl.period}): quadrics give degree-3 gaps '
                 f'{st["ring_gaps_quadrics"]}; family of triple contractions {len(self.fam)} x {len(self.fam[0][2]) if self.fam else 0}')

    def save(self):
        sl, st = self.sl, self.st
        np.savez(self.nf + '.part.npz', **{f'B{s}': sl.blocks[s].B.astype(np.float32) for s in sl.sec3},
                 **{f'piv{s}': np.array(sl.blocks[s].piv, dtype=np.int64) for s in sl.sec3})
        os.replace(self.nf + '.part.npz', self.nf)
        st['saved'] = True; st['gaps'] = sl.gaps(); save_json(self.jf, st)

    def finish_ring(self, killed, how):
        """Record the step; either the slice is closed, or continue on the surviving coordinates."""
        sl, st = self.sl, self.st
        kc = sorted(set(int(sl.chars[a]) for a in killed))
        st['chain'].append({'coords': sl.e, 'gaps_quadrics': st['ring_gaps_quadrics'], 'contractions': st['ring_ncomb'],
                            'killed': len(killed), 'killed_chars': kc, 'how': how})
        self.log(f'  [{self.name}] -> {len(killed)} coordinates vanish (characters {kc}), by {how}, after {st["ring_ncomb"]} triple contractions')
        keep = [i for i in range(sl.e) if i not in set(killed)]
        st['vars'] = [int(x) for x in np.array(st['vars'])[keep]]
        st.update(pos=None, stats=None, ring_ncomb=0, ring_gaps_quadrics=None, saved=False, gaps=None)
        if os.path.exists(self.nf): os.remove(self.nf)
        if not keep:
            st['status'] = 'CLOSED'; save_json(self.jf, st); return True
        save_json(self.jf, st); self.new_ring(load=False); return False

    def pick(self, rng):
        st = self.st
        avail = [i for i in range(len(self.fam)) if st['pos'][i] < len(self.fam[i][2])]
        if not avail: return None
        visited = [i for i in avail if st['stats'][i][1] is not None]
        if not visited or rng.random() < EPS:
            m = min(st['stats'][i][0] for i in avail)
            return rng.choice([i for i in avail if st['stats'][i][0] == m])
        return max(visited, key=lambda i: st['stats'][i][1])

    def forms(self, fi):
        if fi in self.ycache:
            Y = self.ycache.pop(fi); self.ycache[fi] = Y; return Y
        c = self.ctx; sl = self.sl
        Y = first_forms_g(c['L'], c['nb'], c['K'], self.fam[fi][0], sl.Wy, c['N'], c['allP_index'], c['p'])
        Yflat = Y[:, self.iu[0], self.iu[1]] * self.wgt[None, :]; del Y
        while self.ycache and (len(self.ycache) + 1) * Yflat.nbytes > YCACHE_MB * 2 ** 20:
            self.ycache.pop(next(iter(self.ycache)))
        self.ycache[fi] = Yflat
        return Yflat

    def step(self, fi):
        c, sl, st = self.ctx, self.sl, self.st
        g0 = total_gap(sl); t0 = time.time()
        Yflat = self.forms(fi); f, k1, sec = self.fam[fi]; pos = st['pos'][fi]; todo = sec[pos:pos + PICK]
        for c2 in todo:
            for rows in second_cubics_g(c['Lall'], c['nball'], k1, c['L'], c['nb'], c['K'], c2, Yflat, sl.Wy, c['N'],
                                        self.mmap, len(sl.mon[3]), c['p'], c['cache']):
                sl.add_cubics(rows)
                if sl.closed(): break
            st['ncomb'] += 1; st['ring_ncomb'] += 1
        st['pos'][fi] = pos + len(todo); st['picks'] += 1
        gain = (g0 - total_gap(sl)) / max(len(todo), 1)
        s = st['stats'][fi]; s[0] += 1; s[1] = gain if s[1] is None else EMA * gain + (1 - EMA) * s[1]
        return gain, time.time() - t0

    def run(self, t_end, stop_after_picks=None):
        """Work until the slice closes, the family is used up, or time is up.  Returns 'CLOSED', 'OPEN' or 'TIME'."""
        st = self.st
        if st['status'] != 'RUNNING': return st['status'].split()[0]
        rng = random.Random(1000 + st['picks']); t_start = time.time(); last_save = time.time(); last_rep = time.time()
        picks_here = 0
        while True:
            sl = self.sl
            if sl.closed():
                if self.finish_ring(list(range(sl.e)), 'degree 3 (full)'): break
                continue
            killed = sl.cubes()
            if killed:
                if self.finish_ring(killed, 'quadrics' if st['ring_ncomb'] == 0 else 'triple contractions'): break
                continue
            exhausted = all(st['pos'][i] >= len(self.fam[i][2]) for i in range(len(self.fam)))
            if sl.e <= QUARTIC_MAX and (exhausted or (sl.e <= 16 and st['picks'] % QUARTIC_EVERY == 0 and st['ring_ncomb'] > 0)):
                g4, fourth = sl.quartic_test()
                self.log(f'  [{self.name}] degree 4: gaps {g4}; 4th powers in ideal: {len(fourth)}')
                if all(v == 0 for v in g4.values()):
                    if self.finish_ring(list(range(sl.e)), 'degree 4 (full)'): break
                    continue
                if fourth:
                    if self.finish_ring(fourth, 'degree 4'): break
                    continue
            if exhausted:
                st['status'] = f'OPEN (all {st["ring_ncomb"]} triple contractions used, {sl.e} coordinates left)'
                self.save(); break
            if time.time() > t_end or (stop_after_picks is not None and picks_here >= stop_after_picks):
                self.save(); st['seconds'] += time.time() - t_start; save_json(self.jf, st); return 'TIME'
            fi = self.pick(rng)
            gain, dt = self.step(fi); picks_here += 1
            self.log(f'  [{self.name}] pick {st["picks"]}: first ribbon {fi} ({st["stats"][fi][0]} uses), {PICK} contractions, '
                     f'gain {gain:.0f} per contraction, gaps {sl.gaps()}, total {st["ncomb"]} contractions  ({dt/60:.1f} min)')
            if time.time() - last_save > SAVE_EVERY: self.save(); last_save = time.time()
            if time.time() - last_rep > REPORT_EVERY:
                progress(f'level {self.ctx["K"]}  N={self.ctx["N"]}: slice {self.name}: {sl.e} coordinates, gaps {sl.gaps()}, '
                         f'{st["ncomb"]} triple contractions used so far'); last_rep = time.time()
        st['seconds'] += time.time() - t_start; save_json(self.jf, st)
        return st['status'].split()[0]

    def summary(self):
        st = self.st
        chain = ' -> '.join(str(x['coords']) for x in st['chain'])
        if st['status'] == 'CLOSED': chain += ' -> 0'
        elif st['chain']: chain += f' -> {len(st["vars"])}'
        else: chain = str(len(st['vars']))
        hows = '; '.join(f'{x["killed"]} by {x["how"]} after {x["contractions"]} contractions' for x in st['chain'])
        return chain, hows

def report(T, ctx):
    st = T.st; chain, hows = T.summary()
    if st['status'] == 'CLOSED':
        progress(f'level {ctx["K"]}  N={ctx["N"]}: slice {T.name}: CLOSED, coordinates {chain} ({hows}); '
                 f'{st["ncomb"]} triple contractions in all  ({st["seconds"]/60:.0f} min)')
    else:
        progress(f'level {ctx["K"]}  N={ctx["N"]}: slice {T.name}: {st["status"]}, coordinates {chain}, gaps {st["gaps"]}, '
                 f'{st["ncomb"]} triple contractions  ({st["seconds"]/60:.0f} min)' + (f'  [{hows}]' if hows else ''))

# ======================================================================= driver
def run(K, N, p, queue, hours, which):
    os.makedirs(OUT, exist_ok=True)
    t_end = time.time() + hours * 3600; log = Log(f'tiger_K{K}_N{N}.log')
    progress(f'=== tiger {which}: level {K}, N={N}, slices {queue}, job started (stops after {hours:g} h) ===')
    stt = full_state(K, N, p, log, t_end)
    if stt is None: progress(f'=== tiger {which}: time is up (during the quadric step, which will resume) ==='); return
    W, chars, B = stt
    ctx = make_ctx(K, N, p, W, chars, B)
    log(f'dim K_{K} = {ctx["d"]}, characters {np.bincount(chars, minlength=K).tolist()}, quadric span ranks {[b.shape[0] for b in B]}')
    for S in queue:
        T = Tiger(ctx, S, log, f'K{K}_N{N}_p{p}')
        if T.st['status'] != 'RUNNING': continue
        log(f'slice {T.name}: {len(T.st["vars"])} coordinates')
        r = T.run(t_end)
        if r == 'TIME':
            progress(f'level {K}  N={N}: slice {T.name}: paused, gaps {T.st["gaps"]}, {T.st["ncomb"]} triple contractions so far '
                     f'(run the same command again to continue)'); break
        report(T, ctx)
    progress(f'=== tiger {which}: time is up ===' if time.time() >= t_end else f'=== tiger {which}: queue done ===')

def run_test():
    t0 = time.time()
    def say(s): print(f'[{time.time()-t0:5.0f}s] {s}', flush=True)
    say(f'python {sys.version.split()[0]}, numpy {np.__version__}, cores {os.cpu_count()}')
    K, N, p = 8, 11, PRIMES[8]; log = Log('tiger_test.log')
    W, chars, B = full_state(K, N, p, log, time.time() + 3600)
    ok = len(chars) == 66 and sum(b.shape[0] for b in B) == 1154
    say(f'level 8, N=11: dim K_8 = {len(chars)}, quadric span {sum(b.shape[0] for b in B)}  {"OK" if ok else "WRONG (expected 66, 1154)"}')
    ctx = make_ctx(K, N, p, W, chars, B)
    # the even ring at level 8, with an interruption after 2 picks and a restart from the save
    for f in os.listdir(OUT):
        if f.startswith('slice_test'): os.remove(os.path.join(OUT, f))
    T = Tiger(ctx, [0, 2, 4, 6], log, 'test'); r = T.run(time.time() + 3600, stop_after_picks=2)
    g_saved = T.st['gaps']; say(f'even ring at level 8: interrupted on purpose ({r}), saved gaps {g_saved}')
    T = Tiger(ctx, [0, 2, 4, 6], log, 'test'); g_back = T.sl.gaps()
    good = g_back == g_saved; say(f'restart: gaps read back {g_back}  {"OK" if good else "WRONG"}'); ok &= good
    r = T.run(time.time() + 3 * 3600); chain, hows = T.summary()
    good = r == 'CLOSED' and chain == '46 -> 26 -> 14 -> 0'
    say(f'even ring at level 8: {r}, chain {chain} ({hows})  {"OK" if good else "WRONG (expected CLOSED, 46 -> 26 -> 14 -> 0)"}')
    ok &= good
    have = [f for f in (os.listdir(TYSONOUT) if os.path.isdir(TYSONOUT) else []) if f.startswith('nine_')]
    say(f'tyson.py level-9 files found in tysonout/: {sorted(have) if have else "NONE (tiger would redo the quadric step, about 6 hours)"}')
    if have:
        # level 9: the slice j = 1 must give the kitty value 2684 after the quadrics
        for N9 in (11, 13):
            tag = f'N{N9}_p{PRIMES[9]}'
            if not (os.path.exists(os.path.join(TYSONOUT, f'nine_basis_{tag}.npz'))): continue
            js9 = load_json(os.path.join(TYSONOUT, f'nine_{tag}.json'), {})
            if js9.get('status') not in ('OPEN', 'CLOSED'):
                say(f'level 9, N={N9}: tyson.py data in tysonout/ is incomplete ({len(js9.get("merged", []))} of 90 configurations), '
                    f'skipped (not needed unless this laptop runs N={N9})')
                continue
            stt9 = full_state(9, N9, PRIMES[9], log, time.time() + 600)
            if stt9 is None: say(f'level 9, N={N9}: could not read the data, skipped'); continue
            W9, ch9, B9 = stt9
            c9 = make_ctx(9, N9, PRIMES[9], W9, ch9, B9)
            sl = Slice(c9, vars_=support_vars(c9, [1])); sl.quadrics(); g = sl.gaps()
            good = g == {3: 2684}; say(f'level 9, N={N9}: slice j = 1 after the quadrics: gaps {g}  {"OK" if good else "WRONG (expected {3: 2684})"}')
            ok &= good
    say('SELF-TEST PASSED' if ok else 'SELF-TEST FAILED -- send me the output')

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else ''
    if what == 'test': run_test()
    elif what in QUEUES:      # python3 tiger.py a [hours]   (default HOURS)
        hours = float(sys.argv[2]) if len(sys.argv) > 2 else HOURS
        run(9, QUEUES[what][0], PRIMES[9], QUEUES[what][1], hours, what)
    elif what == 'slice':      # python3 tiger.py slice K N j1 j2 ...   (one slice, any level)
        K, N = int(sys.argv[2]), int(sys.argv[3]); S = [int(x) for x in sys.argv[4:]]
        run(K, N, PRIMES[K], [S], HOURS, 'slice')
    else: print(open(__file__).read().split('import sys')[0])
