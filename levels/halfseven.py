# Level-8 half-turn theorem at a given N (Claude and Teo).
# Proves: if C first differs from NC at level 8, then the crossing part of C(8) is invariant under rotation by 4 legs.
# Certificate: (1) exact rational kernel basis of K_8(N) (multimodular + rational reconstruction + exact check),
# (2) in the rotation eigenbasis (mod p0 = 1048433 = 1 mod 8), quadrics x linear forms + two triple contractions
#     span all cubic monomials of odd rotation character; hence every odd eigen-coordinate y_a has y_a^3 in the ideal.
import sys, time, itertools, numpy as np
from math import gcd, isqrt
from engine import (crossing_parts, parts, labels, forms, config_outputs, kernel_basis, delete_leg, merge, canon,
                    BatchRank)
from triple8 import first_forms, second_cubics
P0 = 1048433; k = 8
import os
MAXEXTRA = int(os.environ.get('MAXEXTRA', '60'))

def log(*a):
    print(*a, flush=True)

def ratrec(a, m):
    """Rational reconstruction of a mod m: returns (num, den) with |num|, den <= sqrt(m/2), or None."""
    a %= m; B = isqrt(m // 2)
    r0, r1, s0, s1 = m, a, 0, 1
    while r1 > B:
        q = r0 // r1; r0, r1 = r1, r0 - q * r1; s0, s1 = s1, s0 - q * s1
    if s1 == 0 or abs(s1) > B: return None
    if s1 < 0: r1, s1 = -r1, -s1
    if gcd(r1, s1) != 1: return None
    return r1, s1

def exact_kernel(N, crk):
    """Rational kernel basis of M(N), certified by an exact integer check. Returns (free, num (int matrix), den (ints))."""
    crl = crossing_parts(k - 1); il = {q: a for a, q in enumerate(crl)}
    rows = []                                            # sparse integer rows of M(N)
    for typ, i in [('d', i) for i in range(k)] + [('m', i) for i in range(k)]:
        R = [dict() for _ in crl]
        for a, P in enumerate(crk):
            if typ == 'd':
                sing, q = delete_leg(P, i, k)
                if q in il: R[il[q]][a] = R[il[q]].get(a, 0) + (N if sing else 1)
            else:
                q = merge(P, i, (i + 1) % k, k)
                if q in il: R[il[q]][a] = R[il[q]].get(a, 0) + 1
        rows += [r for r in R if r]
    primes = [2147483647, 2147483629, 2147483587, 2147483579, 2147483563, 2147483549, 2147483543, 2147483497,
              2147483489, 2147483477, 2147483423, 2147483399, 2147483353, 2147483323, 2147483269, 2147483249]
    res = []; free0 = None; M = 1; acc = None
    for p in primes:
        _, Wp, free = kernel_basis(k, N, p); Wp = Wp.astype(np.int64)
        if free0 is None: free0 = free
        if free != free0: log('  (prime', p, 'gives different free columns, skipped)'); continue
        # CRT accumulate
        if acc is None: acc = [[int(x) for x in col] for col in Wp.T]; M = p
        else:
            inv = pow(M, -1, p)
            for b in range(Wp.shape[1]):
                col = acc[b]; wp = Wp[:, b]
                for i in range(len(col)):
                    t = ((int(wp[i]) - col[i]) * inv) % p; col[i] += M * t
            M *= p
        # try reconstruction
        ok = True; nums = []; dens = []
        for b, col in enumerate(acc):
            rr = [ratrec(x, M) if x else (0, 1) for x in col]
            if any(r is None for r in rr): ok = False; break
            D = 1
            for _, dd in rr: D = D * dd // gcd(D, dd)
            nums.append([n * (D // dd) for n, dd in rr]); dens.append(D)
        if not ok: continue
        # exact verification: M(N) num_b = 0 for all b
        good = all(sum(c * nums[b][a] for a, c in r.items()) == 0 for r in rows for b in range(len(nums)))
        log(f'  exact kernel: {len(primes[:primes.index(p)+1])} primes, reconstruction {"VERIFIED" if good else "failed check"}')
        if good: return free0, nums, dens
    raise RuntimeError('exact kernel not certified')

def main(N):
    t0 = time.time(); log(f'=== N = {N} ===')
    crk = crossing_parts(k)
    free, nums, dens = exact_kernel(N, crk)
    d = len(free); log(f"  dim K_8 = {d}")
    # reduce exact basis mod P0 (identity on free columns after normalisation) and compare with mod-P0 kernel
    W = np.zeros((len(crk), d), dtype=np.int64)
    for b in range(d):
        Dinv = pow(dens[b] % P0, -1, P0)
        for a in range(len(crk)): W[a, b] = nums[b][a] % P0 * Dinv % P0
    _, Wm, freem = kernel_basis(k, N, P0)
    assert freem == free and np.array_equal(W, Wm.astype(np.int64)), 'mod-P0 kernel is not the reduction of the exact one'
    log(f'  exact basis reduces to the mod-p0 kernel: OK   ({time.time()-t0:.0f}s)')
    # rotation eigenbasis mod P0
    idx = {P: a for a, P in enumerate(crk)}
    perm = np.array([idx[canon([set((x + 1) % k for x in B) for B in P])] for P in crk])
    PW = np.zeros_like(W); PW[perm] = W; Rot = PW[free] % P0
    g = next(x for x in range(2, P0) if pow(x, (P0 - 1) // 2, P0) != 1); om = pow(g, (P0 - 1) // 8, P0)
    def nullspace(Mx):
        Mx = Mx.copy() % P0; r = 0; piv = []; rows_, cols_ = Mx.shape
        for c in range(cols_):
            nz = np.flatnonzero(Mx[r:, c])
            if len(nz) == 0: continue
            pr = r + nz[0]; Mx[[r, pr]] = Mx[[pr, r]]; Mx[r] = (Mx[r] * pow(int(Mx[r, c]), P0 - 2, P0)) % P0
            oth = np.flatnonzero(Mx[:, c]); oth = oth[oth != r]
            if len(oth): Mx[oth] = (Mx[oth] - (Mx[oth, c][:, None] * Mx[r][None, :]) % P0) % P0
            piv.append(c); r += 1
            if r == rows_: break
        fr = [c for c in range(cols_) if c not in piv]; Nm = np.zeros((cols_, len(fr)), dtype=np.int64)
        for b, f in enumerate(fr):
            Nm[f, b] = 1
            for i, pc in enumerate(piv): Nm[pc, b] = (-Mx[i, f]) % P0
        return Nm
    Es = [nullspace((Rot - pow(om, j, P0) * np.eye(d, dtype=np.int64)) % P0) for j in range(8)]
    E = np.hstack(Es) % P0; chars = np.concatenate([[j] * e.shape[1] for j, e in enumerate(Es)]).astype(int)
    assert E.shape[1] == d
    log(f'  eigenspace dimensions {[e.shape[1] for e in Es]}')
    i0 = None
    if os.environ.get('RELATION') == '1':
        import math
        mu = lambda P_: math.prod((-1) ** (len(B) - 1) * math.factorial(len(B) - 1) for B in P_)
        r = np.array([mu(P_) % P0 for P_ in crk], dtype=np.int64)
        rf = r[free] % P0
        inK = np.array_equal(np.mod(np.mod(W @ (rf & 0x3FF), P0) + np.mod(np.mod(W @ (rf >> 10), P0) * 1024, P0), P0), r)
        log(f'  relation: crossing part lies in K_8 (mod p): {inK}')
        # coordinates of r in the eigenbasis: solve E y = rf
        Aug = np.hstack([E, rf[:, None]]) % P0; rr = 0; pivs = []
        for c in range(d):
            nz = np.flatnonzero(Aug[rr:, c]); pr = rr + nz[0]; Aug[[rr, pr]] = Aug[[pr, rr]]
            Aug[rr] = (Aug[rr] * pow(int(Aug[rr, c]), P0 - 2, P0)) % P0
            oth = np.flatnonzero(Aug[:, c]); oth = oth[oth != rr]
            if len(oth): Aug[oth] = (Aug[oth] - (Aug[oth, c][:, None] * Aug[rr][None, :]) % P0) % P0
            rr += 1
        yr = Aug[:, d]
        supp = np.flatnonzero(yr); log(f'  relation lies in characters {sorted(set(chars[supp].tolist()))}')
        i0 = int(supp[0]); E[:, i0] = rf; log(f'  eigen-coordinate {i0} replaced by the relation direction; imposing y_{i0} = 0')
    Wy = np.mod(np.mod(W @ (E & 0x3FF), P0) + np.mod(np.mod(W @ (E >> 10), P0) * 1024, P0), P0).astype(np.float64)
    # quadrics from configurations anchored at s = 0, split into rotation characters
    L, nb = labels(crk, k)
    ylist = [(a, b) for a in range(d) for b in range(a, d)]
    mchar = np.array([(chars[a] + chars[b]) % 8 for a, b in ylist])
    spans = {chi: BatchRank(len(ylist), P0) for chi in range(8)}
    confs = [(0, t, j, o1, o2) for t in range(k) for j in range(4, k + 1) for o1 in (0, 1) for o2 in (0, 1)
             if 2 * k - 2 * j + o1 + o2 <= k - 1 and len(config_outputs(k, (0, t, j, o1, o2))[2]) >= 4]
    cache = {}
    for conf in confs:
        acc, _ = forms(L, nb, k, conf, Wy, N, p=P0, crossing_cache=cache)
        if not acc: continue
        X = np.array([[Q[a, b] if a == b else (2 * Q[a, b]) % P0 for a, b in ylist] for Q in acc.values()], dtype=np.int64)
        for chi in range(8):
            Xc = np.where(mchar[None, :] == chi, X, 0)
            Xc = Xc[np.any(Xc != 0, axis=1)]
            if len(Xc): spans[chi].add(Xc)
    log(f'  quadric span by character {[spans[c].rank for c in range(8)]} (total {sum(spans[c].rank for c in range(8))})   ({time.time()-t0:.0f}s)')
    # triple contractions
    allP = sorted(parts(k)); Lall, nball = labels(allP, k); allP_index = {P: i for i, P in enumerate(allP)}
    cub = list(itertools.combinations_with_replacement(range(d), 3)); cubidx = {m: i for i, m in enumerate(cub)}
    iu = np.triu_indices(d)
    mp = np.array([[cubidx[tuple(sorted((a, b, e)))] for e in range(d)] for a, b in zip(*iu)])
    Y = first_forms(L, nb, k, (0, 0, 4, 0, 0), Wy, N, allP_index)
    trip = []
    for c2 in [(0, 0, 5, 0, 0), (0, 0, 5, 0, 1)]:
        trip += second_cubics(Lall, nball, L, nb, k, c2, Y, Wy, N, {'list': cub, 'map': mp})
    TR = np.mod(np.array(trip), P0).astype(np.int64)
    log(f'  triple-contraction cubics: {len(trip)}   ({time.time()-t0:.0f}s)')
    # odd-sector Macaulay test
    cchar = {}
    for m in cub: cchar.setdefault(int(sum(chars[list(m)]) % 8), []).append(m)
    def rank_mod(Mx):
        Mx = Mx % P0; r = 0; rows_, cols_ = Mx.shape
        for c in range(cols_):
            nz = np.flatnonzero(Mx[r:, c])
            if len(nz) == 0: continue
            pr = r + nz[0]
            if pr != r: Mx[[r, pr]] = Mx[[pr, r]]
            Mx[r, c:] = (Mx[r, c:] * pow(int(Mx[r, c]), P0 - 2, P0)) % P0
            below = r + 1 + np.flatnonzero(Mx[r + 1:, c])
            if len(below): Mx[below, c:] = (Mx[below, c:] - (Mx[below, c][:, None] * Mx[r, c:][None, :]) % P0) % P0
            r += 1
            if r == rows_: break
        return r
    import random
    def mm(A, B):
        Bl = np.mod(B, 1024); Bh = np.floor_divide(B, 1024)
        return np.mod(np.mod(A @ Bl, P0) + np.mod(np.mod(A @ Bh, P0) * 1024, P0), P0)
    class Blk:
        def __init__(self, psi):
            self.cols = cchar[psi]; ci = {m: i for i, m in enumerate(self.cols)}; self.colsel = np.array([cubidx[m] for m in self.cols])
            rows_ = []
            for i in range(d):
                chi = (psi - chars[i]) % 8
                for q in spans[chi].B.astype(np.int64):
                    row = np.zeros(len(self.cols), dtype=np.int64)
                    for mi in np.flatnonzero(q):
                        a, b = ylist[mi]; m = tuple(sorted((a, b, i))); row[ci[m]] = (row[ci[m]] + q[mi]) % P0
                    rows_.append(row)
            if i0 is not None:
                for m in self.cols:
                    if i0 in m:
                        u = np.zeros(len(self.cols), dtype=np.int64); u[ci[m]] = 1; rows_.append(u)
            self.B = np.zeros((0, len(self.cols))); self.piv = []; self.add(np.array(rows_))
        def add(self, X):
            X = np.mod(np.asarray(X, dtype=np.float64), P0)
            if self.B.shape[0]: X = np.mod(X - mm(X[:, self.piv], self.B), P0)
            X = X.astype(np.int64); r = 0; piv = []
            for c in range(X.shape[1]):
                nz = np.flatnonzero(X[r:, c])
                if len(nz) == 0: continue
                pr = r + nz[0]; X[[r, pr]] = X[[pr, r]]; X[r] = (X[r] * pow(int(X[r, c]), P0 - 2, P0)) % P0
                oth = np.flatnonzero(X[:, c]); oth = oth[oth != r]
                if len(oth): X[oth] = (X[oth] - (X[oth, c][:, None] * X[r][None, :]) % P0) % P0
                piv.append(c); r += 1
                if r == X.shape[0]: break
            if r:
                Nw = X[:r].astype(np.float64)
                if self.B.shape[0]: self.B = np.mod(self.B - mm(self.B[:, piv], Nw), P0)
                self.B = np.vstack([self.B, Nw]); self.piv += piv
        @property
        def gap(self): return len(self.cols) - len(self.piv)
    BL = [Blk(psi) for psi in (1, 3, 5, 7)]
    for b in BL: b.add(TR[:, b.colsel])
    log(f'  odd sectors after the standard triples: gaps {[b.gap for b in BL]}   ({time.time()-t0:.0f}s)')
    rng = random.Random(N); cachec = {}
    firsts = [(s_, t_, j, o1, o2) for s_ in range(k) for t_ in range(k) for j in (4, 5) for o1 in (0, 1) for o2 in (0, 1)
              if 2 * k - 2 * j + o1 + o2 == k]
    seconds = [(0, t_, 5, o1, o2) for t_ in range(k) for o1 in (0, 1) for o2 in (0, 1) if 2 * k - 10 + o1 + o2 <= k - 1]
    step = 0
    while any(b.gap for b in BL) and step < MAXEXTRA:
        f = rng.choice(firsts); Yf = first_forms(L, nb, k, f, Wy, N, allP_index)
        for c2 in rng.sample(seconds, 6):
            rows_ = second_cubics(Lall, nball, L, nb, k, c2, Yf, Wy, N, {'list': cub, 'map': mp}, cache=cachec)
            if rows_:
                R = np.mod(np.array(rows_), P0).astype(np.int64)
                for b in BL: b.add(R[:, b.colsel])
            step += 1
            log(f'    extra {step}: first {f} second {c2}: gaps {[b.gap for b in BL]}')
            if not any(b.gap for b in BL): break
    gaps = {psi: b.gap for psi, b in zip((1, 3, 5, 7), BL)}
    proved = all(g == 0 for g in gaps.values())
    log(f'  odd sectors: gaps {gaps}')
    log(f'RESULT N={N}: {"PROVED (half-turn invariance)" if proved else "not proved, gaps " + str(gaps)}   total {(time.time()-t0)/60:.1f} min')
    return proved

if __name__ == '__main__':
    for N in [int(x) for x in sys.argv[1:]]:
        main(N)
