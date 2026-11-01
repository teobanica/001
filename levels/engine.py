# Vectorised ribbon engine (numpy + scipy.sparse), for levels 7-8 and beyond.
# Arithmetic modulo a prime p < 2^20 in float64 (exact: products < 2^40, sums of < 2^12 terms).
import numpy as np, itertools
import scipy.sparse as sp

PR = 1048573            # prime < 2^20

def set_partitions(k):
    def rec(l):
        if l == 0:
            yield []; return
        for p in rec(l - 1):
            for i in range(len(p)): yield p[:i] + [p[i] + [l - 1]] + p[i + 1:]
            yield p + [[l - 1]]
    return list(rec(k))
def canon(blocks): return tuple(sorted(tuple(sorted(b)) for b in blocks if b))
def noncrossing(P):
    for A, B in itertools.permutations(P, 2):
        for a1, a2 in itertools.combinations(A, 2):
            for b1, b2 in itertools.combinations(B, 2):
                if a1 < b1 < a2 < b2: return False
    return True
def parts(k): return [canon([set(b) for b in p]) for p in set_partitions(k)]
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

def crossing_parts(k): return sorted(q for q in parts(k) if not noncrossing(q))

def labels(crk, k):
    """int8 array (n, k): block label of each leg, blocks numbered 0.. in order of first leg."""
    L = np.zeros((len(crk), k), dtype=np.int8); nb = np.zeros(len(crk), dtype=np.int8)
    for a, P in enumerate(crk):
        for bi, B in enumerate(sorted(P)):
            for x in B: L[a, x] = bi
        nb[a] = len(P)
    return L, nb

def config_outputs(k, conf):
    s, t, j, o1, o2 = conf
    xs = [(s + i) % k for i in range(j)]; ys = [(t + j - 1 - i) % k for i in range(j)]
    outs = [('x', (s + j + i) % k) for i in range(k - j)]
    if o1: outs.append(('x', xs[0]))
    outs += [('y', (t + j + i) % k) for i in range(k - j)]
    if o2 and j > 1: outs.append(('x', xs[-1]))
    return xs, ys, outs

def pair_keys(L, nb, k, conf, prow):
    """For partitions prow (indices) against all partitions: output RGS code and closed-loop count.
    Returns codes (len(prow) x n, int64) and closed (int8)."""
    xs, ys, outs = config_outputs(k, conf)
    n = L.shape[0]; m = len(prow)
    # node labels: x-blocks 0..7, y-blocks 8..15 (offset 8)
    lab = np.empty((m, n, 16), dtype=np.int8)
    lab[:] = np.arange(16, dtype=np.int8)
    lx = L[prow]                          # m x k
    ly = L + 8                            # n x k
    for x, y in zip(xs, ys):
        u = np.broadcast_to(lx[:, x][:, None], (m, n)); v = np.broadcast_to(ly[:, y][None, :], (m, n))
        ru = np.take_along_axis(lab, u[:, :, None].astype(np.int64), 2)[:, :, 0]
        rv = np.take_along_axis(lab, v[:, :, None].astype(np.int64), 2)[:, :, 0]
        lab = np.where(lab == rv[:, :, None], ru[:, :, None], lab)
    # labels of output legs
    oc = []
    for side, v in outs:
        node = np.broadcast_to(lx[:, v][:, None], (m, n)) if side == 'x' else np.broadcast_to(ly[:, v][None, :], (m, n))
        oc.append(np.take_along_axis(lab, node[:, :, None].astype(np.int64), 2)[:, :, 0])
    oc = np.stack(oc, axis=2)             # m x n x nout
    nout = oc.shape[2]
    # restricted growth string
    rgs = np.zeros_like(oc); seen = np.full((m, n, 16), -1, dtype=np.int8); cnt = np.zeros((m, n), dtype=np.int8)
    ii, jj = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
    for o in range(nout):
        lbl = oc[:, :, o].astype(np.int64)
        cur = seen[ii, jj, lbl]
        new = cur < 0
        seen[ii[new], jj[new], lbl[new]] = cnt[new]
        rgs[:, :, o] = np.where(new, cnt, cur)
        cnt = cnt + new
    code = np.zeros((m, n), dtype=np.int64)
    for o in range(nout): code = code * 16 + rgs[:, :, o]
    # closed loops: components among existing nodes (x-blocks < nb_p, y-blocks < nb_q) minus output components
    valid = np.zeros((m, n, 16), dtype=bool)
    valid[:, :, :8] = (np.arange(8)[None, :] < nb[prow][:, None])[:, None, :]
    valid[:, :, 8:] = (np.arange(8)[None, :] < nb[:, None])[None, :, :]
    roots = np.where(valid, lab, -1)
    # number of distinct roots: roots are node ids (0..15); a node is a root of its component iff lab==id
    isroot = valid & (lab == np.arange(16, dtype=np.int8)[None, None, :])
    ncomp = isroot.sum(axis=2)
    closed = (ncomp - cnt).astype(np.int8)
    return code, closed, nout

def decode(code, nout):
    rgs = []
    for _ in range(nout): rgs.append(int(code % 16)); code //= 16
    return tuple(reversed(rgs))

def rgs_partition(rgs):
    bl = {}
    for i, l in enumerate(rgs): bl.setdefault(l, set()).add(i)
    return canon(list(bl.values()))

def forms(L, nb, k, conf, W, N, p=PR, chunk=200, crossing_cache=None):
    """Quadratic forms (symmetric d x d, mod p) of one ribbon config, for crossing outputs.
    W: n x d float64 basis (entries in [0,p)). Returns dict code -> matrix."""
    n, d = W.shape
    if crossing_cache is None: crossing_cache = {}
    if len(config_outputs(k, conf)[2]) < 4: return {}, 0      # fewer than 4 legs: no crossing outputs
    Npow = np.array([pow(N, e, p) for e in range(17)], dtype=np.float64)
    acc = {}
    for c0 in range(0, n, chunk):
        prow = np.arange(c0, min(n, c0 + chunk))
        code, closed, nout = pair_keys(L, nb, k, conf, prow)
        flat = code.ravel(); ucodes, inv = np.unique(flat, return_inverse=True)
        keep = []
        for ui, uc in enumerate(ucodes):
            if uc not in crossing_cache:
                crossing_cache[uc] = not noncrossing(rgs_partition(decode(int(uc), nout)))
            if crossing_cache[uc]: keep.append(ui)
        if not keep: continue
        keep = np.array(keep); kmap = -np.ones(len(ucodes), dtype=np.int64); kmap[keep] = np.arange(len(keep))
        kidx = kmap[inv.ravel()]
        sel = kidx >= 0
        P_ = np.repeat(prow, n)[sel]; Q_ = np.tile(np.arange(n), len(prow))[sel]
        wts = Npow[closed.ravel()[sel]]
        rowsS = kidx[sel] * len(prow) + (P_ - c0)
        S = sp.csr_matrix((wts, (rowsS, Q_)), shape=(len(keep) * len(prow), n))
        SW = np.mod(S @ W, p)                                    # (keys*m) x d
        SW = SW.reshape(len(keep), len(prow), d)
        Wp = W[prow]                                              # m x d
        for ki, ui in enumerate(keep):
            Q = np.mod(Wp.T @ SW[ki], p)
            uc = int(ucodes[ui])
            acc[uc] = np.mod(acc[uc] + Q, p) if uc in acc else Q
    for uc in acc: acc[uc] = np.mod(acc[uc] + acc[uc].T, p)
    return acc, nout

def kernel_basis(k, N, p=PR):
    """Basis of K_k(N) mod p (columns), over the sorted crossing partitions."""
    crk = crossing_parts(k); crl = crossing_parts(k - 1); il = {q: a for a, q in enumerate(crl)}
    M = np.zeros((2 * k * len(crl), len(crk)), dtype=np.int64); r0 = 0
    for typ, i in [('d', i) for i in range(k)] + [('m', i) for i in range(k)]:
        for a, P in enumerate(crk):
            if typ == 'd':
                sing, q = delete_leg(P, i, k)
                if q in il: M[r0 + il[q], a] += N if sing else 1
            else:
                q = merge(P, i, (i + 1) % k, k)
                if q in il: M[r0 + il[q], a] += 1
        r0 += len(crl)
    M %= p; rows, cols = M.shape; r = 0; piv = []
    for c in range(cols):
        nz = np.flatnonzero(M[r:, c])
        if len(nz) == 0: continue
        pr = r + nz[0]
        if pr != r: M[[r, pr]] = M[[pr, r]]
        M[r] = (M[r] * pow(int(M[r, c]), p - 2, p)) % p
        others = np.flatnonzero(M[:, c]); others = others[others != r]
        if len(others):
            f = M[others, c].copy(); M[others] = (M[others] - (f[:, None] * M[r][None, :]) % p) % p
        piv.append(c); r += 1
        if r == rows: break
    pivset = set(piv); free = [c for c in range(cols) if c not in pivset]
    W = np.zeros((cols, len(free)), dtype=np.int64)
    for b, fc in enumerate(free):
        W[fc, b] = 1
        for i, pc in enumerate(piv): W[pc, b] = (-M[i, fc]) % p
    return crk, W.astype(np.float64), free

class RankTracker:
    def __init__(self, dim, p=PR):
        self.p = p; self.dim = dim; self.piv = {}
    def add(self, v):
        p = self.p; v = np.mod(np.asarray(v, dtype=np.int64), p)
        for c in sorted(self.piv):
            if v[c]: v = (v - v[c] * self.piv[c]) % p
        nz = np.flatnonzero(v)
        if len(nz) == 0: return False
        c = nz[0]; v = (v * pow(int(v[c]), p - 2, p)) % p
        for c2 in list(self.piv):
            r2 = self.piv[c2]
            if r2[c]: self.piv[c2] = (r2 - r2[c] * v) % p
        self.piv[c] = v; return True
    @property
    def rank(self): return len(self.piv)

class BatchRank:
    """Incremental row-rank mod p (p < 2^20) with float64 matmuls; basis kept in RREF."""
    def __init__(self, dim, p=PR):
        self.p = p; self.dim = dim; self.B = np.zeros((0, dim)); self.piv = []
    def add(self, X):
        p = self.p; X = np.mod(np.asarray(X, dtype=np.float64), p)
        if self.B.shape[0]:
            X = np.mod(X - np.mod(X[:, self.piv] @ self.B, p), p)
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
        if not newrows: return 0
        Nw = np.array(newrows, dtype=np.float64)
        if self.B.shape[0]:
            self.B = np.mod(self.B - np.mod(self.B[:, newpiv] @ Nw, p), p)
        self.B = np.vstack([self.B, Nw]); self.piv += newpiv
        return len(newrows)
    @property
    def rank(self): return len(self.piv)

def upper(Q):
    d = Q.shape[0]; iu = np.triu_indices(d); return Q[iu]
