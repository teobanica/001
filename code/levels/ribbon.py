# Fast ribbon-contraction engine at level k, modulo a prime p, for exploration.
# Pure Python + numpy. A "config" is (s, t, j, o1, o2): strings x_{s+i} <-> y_{t+j-1-i}, i<j;
# o1: output leg on the first string, o2: output leg on the last string (needs j > 1).
import numpy as np
from levlin import parts, noncrossing, delete_leg, merge, canon

P_MOD = 1000000007

def blocks_of(P, k):
    lab = [0] * k
    for bi, B in enumerate(P):
        for x in B: lab[x] = bi
    return lab, len(P)

def out_legs(k, s, t, j, o1, o2):
    xs = [(s + i) % k for i in range(j)]; ys = [(t + j - 1 - i) % k for i in range(j)]
    outs = [('x', (s + j + i) % k) for i in range(k - j)]
    if o1: outs.append(('x', xs[0]))
    outs += [('y', (t + j + i) % k) for i in range(k - j)]
    if o2 and j > 1: outs.append(('x', xs[-1]))
    return xs, ys, outs

def pair_tensor(crk, k, conf):
    """For all pairs (p, q) of crossing partitions: output partition (canonical) and number of closed loops."""
    s, t, j, o1, o2 = conf
    xs, ys, outs = out_legs(k, s, t, j, o1, o2)
    labs = [blocks_of(P, k) for P in crk]
    n = len(crk)
    keyid = {}; keys = []
    K = np.zeros((n, n), dtype=np.int32); C = np.zeros((n, n), dtype=np.int8)
    for a in range(n):
        la, ba = labs[a]
        for b in range(n):
            lb, bb = labs[b]
            par = list(range(ba + bb))
            def f(x):
                while par[x] != x:
                    par[x] = par[par[x]]; x = par[x]
                return x
            for x, y in zip(xs, ys):
                rx, ry = f(la[x]), f(ba + lb[y])
                if rx != ry: par[rx] = ry
            orr = [f(la[v]) if side == 'x' else f(ba + lb[v]) for side, v in outs]
            seen = {}; rgs = []
            for r in orr:
                if r not in seen: seen[r] = len(seen)
                rgs.append(seen[r])
            nroots = len({f(i) for i in range(ba + bb)})
            closed = nroots - len(seen)
            key = tuple(rgs)
            if key not in keyid: keyid[key] = len(keys); keys.append(key)
            K[a, b] = keyid[key]; C[a, b] = closed
    return K, C, keys

def rgs_to_partition(rgs):
    bl = {}
    for i, l in enumerate(rgs): bl.setdefault(l, set()).add(i)
    return canon(list(bl.values()))

def kernel_basis_modp(k, N, p=P_MOD):
    """Basis of K_k(N) mod p on crossing columns (columns = crossing partitions of P(k))."""
    Pk = parts(k); Pl = parts(k - 1)
    crk = [q for q in Pk if not noncrossing(q)]; crl = [q for q in Pl if not noncrossing(q)]
    il = {q: a for a, q in enumerate(crl)}
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
        nz = np.nonzero(M[r:, c])[0]
        if len(nz) == 0: continue
        pr = r + nz[0]; M[[r, pr]] = M[[pr, r]]
        M[r] = (M[r] * pow(int(M[r, c]), p - 2, p)) % p
        fcol = M[:, c].copy(); fcol[r] = 0; nzr = np.nonzero(fcol)[0]
        if len(nzr): M[nzr] = (M[nzr] - (fcol[nzr, None] * M[r][None, :]) % p) % p
        piv.append(c); r += 1
        if r == rows: break
    free = [c for c in range(cols) if c not in set(piv)]
    W = np.zeros((cols, len(free)), dtype=np.int64)
    for bidx, fc in enumerate(free):
        W[fc, bidx] = 1
        for i, pc in enumerate(piv): W[pc, bidx] = (-M[i, fc]) % p
    return crk, W

def mulmod(A, B, p=P_MOD):
    """(A @ B) mod p for int64 arrays with entries in [0, p), via 15-bit splitting."""
    Bl = B & 0x7FFF; Bh = B >> 15
    lo = (A @ Bl) % p; hi = (A @ Bh) % p
    return (lo + (hi * 32768) % p) % p

def quadrics_modp(K, C, keys, W, N, crossing_only=True, p=P_MOD):
    """For one config: dict output-partition -> symmetric d x d matrix (mod p) of the quadratic form."""
    n, d = W.shape
    Npow = np.array([pow(N, e, p) for e in range(int(C.max()) + 1)], dtype=np.int64)
    Mw = Npow[C]                                   # n x n weights N^closed
    res = {}
    for kid, key in enumerate(keys):
        Pp = rgs_to_partition(key)
        if crossing_only and noncrossing(Pp): continue
        mask = (K == kid)
        if not mask.any(): continue
        E = np.where(mask, Mw, 0)                  # n x n
        T = mulmod(np.ascontiguousarray(W.T), E, p)     # d x n
        Q = mulmod(T, W, p)                        # d x d : Q[a,b] = sum w_a[p] E[p,q] w_b[q]
        res[Pp] = (Q + Q.T) % p                    # symmetrise (form c^T Q c)
    return res
