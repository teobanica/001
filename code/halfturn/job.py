# Level-8 half-turn theorem at a given N (Claude and Teo).
# Proves: if C first differs from NC at level 8, then the crossing part of C(8) is invariant under rotation by 4 legs.
# Certificate: (1) exact rational kernel basis of K_8(N) (multimodular + rational reconstruction + exact check),
# (2) in the rotation eigenbasis (mod p0 = 1048433 = 1 mod 8), quadrics x linear forms + two triple contractions
#     span all cubic monomials of odd rotation character; hence every odd eigen-coordinate y_a has y_a^3 in the ideal.
import sys, time, itertools, numpy as np
from math import gcd, isqrt
from engine import (crossing_parts, parts, labels, forms, config_outputs, kernel_basis, delete_leg, merge, canon,
                    BatchRank)
from triple import first_forms, second_cubics
P0 = 1048433; k = 8

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
    gaps = {}
    for psi in (1, 3, 5, 7):
        cols = cchar[psi]; ci = {m: i for i, m in enumerate(cols)}; colsel = np.array([cubidx[m] for m in cols])
        rows_ = []
        for i in range(d):
            chi = (psi - chars[i]) % 8
            for q in spans[chi].B.astype(np.int64):
                row = np.zeros(len(cols), dtype=np.int64)
                for mi in np.flatnonzero(q):
                    a, b = ylist[mi]; m = tuple(sorted((a, b, i))); row[ci[m]] = (row[ci[m]] + q[mi]) % P0
                rows_.append(row)
        Mx = np.vstack([np.array(rows_), TR[:, colsel]])
        gaps[psi] = len(cols) - rank_mod(Mx)
    proved = all(g == 0 for g in gaps.values())
    log(f'  odd sectors: gaps {gaps}')
    log(f'RESULT N={N}: {"PROVED (half-turn invariance)" if proved else "not proved, gaps " + str(gaps)}   total {(time.time()-t0)/60:.1f} min')
    return proved

if __name__ == '__main__':
    for N in [int(x) for x in sys.argv[1:]]:
        main(N)
