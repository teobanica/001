#!/usr/bin/env python3
# evander.py -- Section 11 ("Pairings and uncappable vectors") of "On the maximality conjecture for quantum
# permutation groups" (Claude and Teo).  Exploration script ("boxer"), successor in spirit of tyson.py.
#
#   python3 evander.py test        self-test (about 10 minutes): reproduces the known results of Section 11,
#                                  then computes the big gluing table of level 12 (saved, used by a and b)
#   python3 evander.py a [hours]   laptop A: the Brauer problem at level 12, N = 6, 7, 9
#   python3 evander.py b [hours]   laptop B: the Brauer problem at level 12, N = 12, 10, 16
#   python3 evander.py c [hours]   Brauer levels 8, 10 at N = 6, 7, 9, 10, 16, Brauer 12 at N = 8, half-liberated 14 at
#                                  N = 9, 10, 16, 20, 6 (results in evanderout/progress.txt, as for a and b)
#   (hours: optional, default 7; the job stops by itself)
#
# Put evander.py in any folder.  Results go to  evanderout/progress.txt  (keep that file, and
# evanderout/results.json if you can).  Closing the terminal is harmless: run the same command again, and the
# finished items are skipped (an item in progress is redone; the big gluing table is saved once and for all).
#
# What is computed.  Let O_N <= G <= O_N^+ first differ from O_N^+ at level 2m, with category C, and
# W = C(2m) minus TL(2m).  Then W consists of uncappable vectors (Lemma 11.3), it is invariant under the rotation,
# and every x in W satisfies: x o_{m+1} Jx lies in TL(2m-2) (Lemma 11.5, deepest gluing).  So the coefficients
# of x o Jx on the crossing pairings of level 2m-2 vanish.  On each real rotation block V_k of the uncappable space
# we look for a combination of these quadratic forms (random combinations of the crossing coordinates, and all
# rotations, as in Lemma 11.9) which is positive definite.  If one exists on every block, the level is EXCLUDED.
# "fam" is HL (half-liberated pairings P_2^*, the problem O_N^* <= G), BR (all pairings P_2, the Brauer problem), or
# PR (projective version PO_N <= H <= PO_N^+, Section 12: Brauer pairings, rho^2 only, blocks k and m-k merged), or
# PH (the same with half-liberated pairings: PU_N <= H <= PO_N^+).
# Floating point, at given N >= m (so that all the pairing vectors involved are independent).
import sys, os, time, json, itertools
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
import scipy.linalg as sla

def robust_svd(X, full_matrices):
    """SVD with fallbacks: numpy (gesdd) can fail to converge on badly scaled matrices (seen at BR level 12, N = 12).
    Then try LAPACK gesvd, and finally the SVD of the column-normalized matrix, re-scaled afterwards only where needed."""
    try:
        return np.linalg.svd(X, full_matrices=full_matrices)
    except np.linalg.LinAlgError:
        pass
    try:
        return sla.svd(X, full_matrices=full_matrices, lapack_driver='gesvd')
    except Exception:
        pass
    if full_matrices: raise np.linalg.LinAlgError('SVD did not converge (kernel computation)')
    nrm = np.linalg.norm(X, axis=0); nrm[nrm == 0] = 1.0        # column scaling keeps the column space, which is
    return sla.svd(X / nrm, full_matrices=False, lapack_driver='gesvd')   # all that is used of this SVD

HOURS = 7.0
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'evanderout')
QUEUES = {'a': [('BR', 6, 6), ('BR', 6, 7), ('BR', 6, 9)], 'b': [('BR', 6, 12), ('BR', 6, 10), ('BR', 6, 16)],
          # queue c (31 Oct, daytime): Brauer levels 8 and 10 at the N of the level-12 runs, Brauer 12 at N = 8,
          # then half-liberated level 14 at new values of N (the paper has 7, 8, 12)
          'c': [('BR', 4, 6), ('BR', 5, 6), ('BR', 4, 7), ('BR', 5, 7), ('BR', 4, 9), ('BR', 5, 9), ('BR', 4, 10), ('BR', 5, 10),
                ('BR', 4, 16), ('BR', 5, 16), ('BR', 6, 8), ('HL', 7, 9), ('HL', 7, 10), ('HL', 7, 16), ('HL', 7, 20), ('HL', 7, 6)]}
# (fam, m, N, dim U, expected status): level 6 is a negative control, since O_N^* exists there
TESTS = [('HL', 3, 8, 1, 'OPEN'), ('BR', 3, 8, 4, 'OPEN'), ('HL', 4, 8, 2, 'EXCLUDED'), ('HL', 5, 8, 13, 'EXCLUDED'),
         ('HL', 5, 5, 13, 'EXCLUDED'), ('HL', 6, 8, 80, 'EXCLUDED'), ('BR', 4, 8, 31, 'EXCLUDED'), ('BR', 5, 8, 293, 'EXCLUDED'),
         ('BR', 3, 3, 4, 'OPEN'), ('BR', 3, 5, 4, 'OPEN'), ('PR', 4, 8, 31, 'EXCLUDED'), ('PH', 4, 8, 2, 'EXCLUDED')]

def progress(line):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'progress.txt'), 'a') as fh: fh.write(time.strftime('%d %b %H:%M') + '  ' + line + '\n')
    print(time.strftime('%H:%M:%S') + '  ' + line, flush=True)

# ======================================================================= pairings
_PAIR = {}
def pairings(n, fam):
    """Pairings of n points (partner arrays, int8), of the family HL (each pair joins an odd and an even point) or
    BR (all), sorted by code; also codes and a noncrossing flag."""
    key = (n, fam)
    if key in _PAIR: return _PAIR[key]
    res = []
    def rec(part, free):
        if not free: res.append(tuple(part)); return
        a = free[0]
        for b in free[1:]:
            if fam == 'HL' and (b - a) % 2 == 0: continue
            part[a] = b; part[b] = a
            rec(part, [c for c in free if c != a and c != b])
    rec([0] * n, list(range(n)))
    P = np.array(res, dtype=np.int8).reshape(-1, n)
    codes = encode(P)
    o = np.argsort(codes); P = P[o]; codes = codes[o]
    nc = np.array([all(not (a < c < b < d) for a in range(n) for b in [p[a]] if a < b
                       for c in range(n) for d in [p[c]] if c < d) for p in P])
    _PAIR[key] = (P, codes, nc)
    return _PAIR[key]

def encode(P):
    n = P.shape[-1]; w = np.array([n ** i for i in range(n)], dtype=np.int64)
    return (P.astype(np.int64) * w).sum(axis=-1)

def lookup(codes_sorted, c):
    i = np.searchsorted(codes_sorted, c)
    assert np.all(codes_sorted[np.minimum(i, len(codes_sorted) - 1)] == c), 'pairing not found'
    return i

def rotate(P):
    """One-click rotation: point i -> i+1 mod n."""
    n = P.shape[1]; R = np.empty_like(P); R[:, (np.arange(n) + 1) % n] = (P + 1) % n; return R

def reverse(P):
    n = P.shape[1]; R = np.empty_like(P); R[:, n - 1 - np.arange(n)] = n - 1 - P; return R

def cap0(P, fam, N):
    """Sparse matrix of the cap on the points 0,1: coefficients at level n -> coefficients at level n-2."""
    n = P.shape[1]; Q, qcodes, _ = pairings(n - 2, fam)
    loop = P[:, 0] == 1
    R = P.astype(np.int64).copy(); a = R[:, 0].copy(); b = R[:, 1].copy(); idx = np.arange(len(P))
    nl = ~loop
    R[idx[nl], a[nl]] = b[nl]; R[idx[nl], b[nl]] = a[nl]
    R = R[:, 2:] - 2
    rows = lookup(qcodes, encode(R)); vals = np.where(loop, float(N), 1.0)
    return sp.csr_matrix((vals, (rows, idx)), shape=(len(Q), len(P)))

# ======================================================================= uncappable space, by rotation blocks
def uncappable_blocks(fam, m, N):
    """Real orthonormal bases V_k of the real rotation blocks (eigenvalues w^k, w^-k, 0 <= k <= m, w = e^(2 pi i/2m))
    of the uncappable space U(2m), with the rotation matrix on each block."""
    n = 2 * m; P, codes, _ = pairings(n, fam); rot = lookup(codes, encode(rotate(P)))
    C0 = cap0(P, fam, N)
    seen = np.zeros(len(P), bool); orbits = []
    for p in range(len(P)):
        if seen[p]: continue
        orb = [p]; q = rot[p]
        while q != p: orb.append(q); q = rot[q]
        seen[orb] = True; orbits.append(orb)
    blocks = []
    for k in range(m + 1):
        w = np.exp(-2j * np.pi * k / n); cols = []
        for orb in orbits:
            s = len(orb)
            if (k * s) % n: continue
            v = np.zeros(len(P), complex); v[orb] = w ** np.arange(s); cols.append(v)
        if not cols: blocks.append((k, np.zeros((len(P), 0)), None)); continue
        B = np.array(cols).T                                    # rho B = w^-k ... eigenvalue e^{2 pi i k/n}
        M = C0 @ B
        if M.shape[0] == 0: K = np.eye(B.shape[1])
        else:
            u, s_, vh = robust_svd(M, full_matrices=True)
            tol = 1e-9 * max(1.0, s_.max() if len(s_) else 1.0); r = int((s_ > tol).sum())
            K = vh[r:].conj().T
        Uk = B @ K
        if Uk.shape[1] == 0: blocks.append((k, np.zeros((len(P), 0)), None)); continue
        X = np.hstack([Uk.real, Uk.imag]) if 0 < k < m else np.hstack([Uk.real, Uk.imag])
        u, s_, _ = robust_svd(X, full_matrices=False)
        r = int((s_ > 1e-8 * s_.max()).sum()); V = u[:, :r]
        RV = np.empty_like(V); RV[rot] = V                        # rho applied to the columns
        Mrot = V.T @ RV
        blocks.append((k, V, Mrot))
    return blocks

# ======================================================================= gluing tables (independent of N)
def glue_tables(fam, m):
    """T[p,q] = index (at level 2m-2) of the pairing p o_{m+1} rev(q), L[p,q] = number of loops."""
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, f'glue_{fam}_K{2*m}.npz')
    if os.path.exists(f):
        z = np.load(f); return z['T'], z['L']
    n = 2 * m; j = m - 1; a = m + 1; P, codes, _ = pairings(n, fam); Pr = reverse(P).astype(np.int64)
    Q, qcodes, _ = pairings(2 * j, fam)
    nP = len(P); T = np.empty((nP, nP), dtype=np.int16 if len(Q) < 32000 else np.int32); L = np.empty((nP, nP), np.int8)
    t0 = time.time(); chunk = max(1, 1000000 // nP); last = t0; Pr16 = Pr.astype(np.int16)
    for p0 in range(0, nP, chunk):
        pp = P[p0:p0 + chunk].astype(np.int16); cp = len(pp)
        A = np.repeat(pp, nP, axis=0); B = np.tile(Pr16, (cp, 1)); M_ = len(A); ar = np.arange(M_)
        res = np.empty((M_, 2 * j), np.int64); vis = np.zeros((M_, a), bool)
        for s in range(2 * j):
            # start at result point s: on p (point s) if s < j, else on q (point a + s - j)
            side = np.full(M_, 0 if s < j else 1); pt = np.full(M_, s if s < j else a + s - j)
            out = np.full(M_, -1, np.int64)
            for _ in range(a + 2):
                act = out < 0
                if not act.any(): break
                ia = ar[act]; sd = side[act]; x = pt[act]
                nxt = np.where(sd == 0, A[ia, x], B[ia, x]).astype(np.int64)     # follow the edge of the current diagram
                # exit points
                ex0 = (sd == 0) & (nxt < j); ex1 = (sd == 1) & (nxt >= a)
                out[ia[ex0]] = nxt[ex0]; out[ia[ex1]] = j + nxt[ex1] - a
                # glued: p point y >= j  <->  q point n-1-y < a
                g0 = (sd == 0) & ~ex0; g1 = (sd == 1) & ~ex1
                gidx0 = n - 1 - nxt[g0]; gidx1 = nxt[g1]
                vis[ia[g0], gidx0] = True; vis[ia[g1], gidx1] = True
                side[ia[g0]] = 1; pt[ia[g0]] = gidx0
                side[ia[g1]] = 0; pt[ia[g1]] = n - 1 - gidx1
            res[:, s] = out
        T[p0:p0 + cp] = lookup(qcodes, encode(res)).reshape(cp, nP)
        # loops: cycles among unvisited glued points g (p point n-1-g = q point g)
        lab = np.where(vis, 10 ** 6, np.arange(a)[None, :])
        nbp = n - 1 - A[:, n - 1 - np.arange(a)].astype(np.int64)
        nbq = B[:, :a].astype(np.int64)
        for _ in range(a):
            lab = np.minimum(lab, np.minimum(np.take_along_axis(lab, np.clip(nbp, 0, a - 1), 1),
                                             np.take_along_axis(lab, np.clip(nbq, 0, a - 1), 1)))
            lab = np.where(vis, 10 ** 6, lab)
        L[p0:p0 + cp] = ((lab == np.arange(a)[None, :]) & ~vis).sum(1).reshape(cp, nP)
        if time.time() - last > 600:
            last = time.time(); print(f'    gluing table {fam} level {n}: {p0 + cp}/{nP} rows ({(last - t0) / 60:.0f} min)', flush=True)
    np.savez(f + '.part.npz', T=T, L=L); os.replace(f + '.part.npz', f)
    progress(f'gluing table {fam} level {n}: {nP} x {nP}, done in {(time.time() - t0) / 60:.1f} min (saved)')
    return T, L

# ======================================================================= forms and certificates
def block_forms(fam, m, N, V, k, T, NPL, i0, i1, seed):
    """Forms V^T K_c V (symmetric, normalized, float32) for the random crossing combinations c number i0..i1-1
    of block k (reproducible: the seed of form i is (seed, k, i))."""
    Q, _, ncQ = pairings(2 * m - 2, fam); cross = ~ncQ; nP = T.shape[0]; ch = max(1, 4000000 // nP); out = []
    for i in range(i0, i1):
        rng = np.random.default_rng([seed, k, i])
        c = np.zeros(len(Q), np.float32); c[cross] = rng.standard_normal(int(cross.sum()))
        KV = np.empty(V.shape, np.float64)
        for r0 in range(0, nP, ch):
            KV[r0:r0 + ch] = (c[T[r0:r0 + ch]] * NPL[r0:r0 + ch]) @ V
        A = V.T @ KV; A = (A + A.T) / 2; nrm = np.linalg.norm(A)
        out.append((A / nrm if nrm > 0 else A).astype(np.float32))
    return out

def rot_order(R):
    d = R.shape[0]; X = np.eye(d)
    for s in range(1, 200):
        X = X @ R
        if np.allclose(X, np.eye(d), atol=1e-8) or np.allclose(X, -np.eye(d), atol=1e-8): return s
    return 1

def certify(F, R, t_end):
    """Maximize lambda_min of A(lam) = sum_r R^-r (sum_i lam_ir F_i) R^r, lam in [-1,1].  Returns (lmin, margin)."""
    F = np.asarray(F, dtype=np.float32); nf, d, _ = F.shape; s = rot_order(R)
    Rs = [np.linalg.matrix_power(R, r) for r in range(s)]
    def build(lam):
        lam = lam.reshape(nf, s)
        return sum(Rs[r].T @ np.tensordot(lam[:, r].astype(np.float32), F, 1).astype(np.float64) @ Rs[r] for r in range(s))
    best = [-np.inf, None]
    def fg(lam, beta):
        A = build(lam); mu, vecs = np.linalg.eigh(A)
        # margin: lambda_min relative to sum |lam| (each form has norm 1, rotations are orthogonal), so that
        # near-cancellations (A ~ 0 up to rounding) can never pass as certificates
        if mu[0] > best[0]: best[0] = mu[0]; best[1] = (lam.copy(), mu[0] / max(np.abs(lam).sum(), 1e-300))
        z = -beta * (mu - mu[0]); w = np.exp(z); w /= w.sum()
        phi = mu[0] - np.log(np.exp(z).sum()) / beta
        E = (vecs * w) @ vecs.T
        g = np.stack([np.tensordot(F, (Rs[r] @ E @ Rs[r].T).astype(np.float32), axes=([1, 2], [0, 1])) for r in range(s)], axis=1).ravel().astype(np.float64)
        return -phi, -g
    rng = np.random.default_rng(1); lam = rng.uniform(-0.1, 0.1, nf * s)
    for beta0 in (2.0, 10.0, 50.0, 250.0, 1250.0):
        if best[0] > 0 or time.time() > t_end: break
        sc = max(np.abs(np.linalg.eigvalsh(build(lam))).max(), 1e-12)
        r = minimize(fg, lam, args=(beta0 / sc,), jac=True, method='L-BFGS-B',
                     bounds=[(-1, 1)] * (nf * s), options={'maxiter': 300})
        lam = r.x
    return best[0], (best[1][1] if best[1] is not None else -1.0), s

MARGIN_MIN = 1e-5     # well above the float32 rounding of the forms (about 1e-7 per unit of sum |lam|)
SCHEDULE = (60, 150, 300, 500, 700)   # numbers of forms tried on a block, until it is certified

def run_item(fam, m, N, t_end, seed=7, verbose=True):
    t0 = time.time(); assert N >= m, 'need N >= m'
    if fam in ('PR', 'PH'):
        # projective versions, Section 12: PR is PO_N <= H <= PO_N^+ (Brauer pairings), PH is PU_N <= H <= PO_N^+
        # (half-liberated pairings).  Same uncappable space as the orthogonal problem (rho^2 and the linear caps give
        # back the cyclic cap, up to a rotation), but W is only rho^2-invariant: the rho-blocks k and m-k merge, and
        # only even rotations may be used in the gluing constraints
        bb = {k: (V, R) for (k, V, R) in uncappable_blocks('BR' if fam == 'PR' else 'HL', m, N)}; blocks = []
        for k in range(m // 2 + 1):
            ks = sorted({k, m - k}); Vs = [bb[j][0] for j in ks if bb[j][0].shape[1] > 0]
            if not Vs: blocks.append((k, bb[k][0], None)); continue
            V = np.hstack(Vs); d = V.shape[1]; R2 = np.zeros((d, d)); o = 0
            for j in ks:
                Vj, Rj = bb[j]
                if Vj.shape[1] == 0: continue
                e = Vj.shape[1]; R2[o:o + e, o:o + e] = Rj @ Rj; o += e
            blocks.append((k, V, R2))
        base = 'BR' if fam == 'PR' else 'HL'
    else:
        blocks = uncappable_blocks(fam, m, N); base = fam
    dims = {k: V.shape[1] for (k, V, _) in blocks}; dimU = sum(dims.values())
    progress(f'{fam} level {2*m}  N={N}: dim U = {dimU}, real rotation blocks {dims}  ({(time.time()-t0)/60:.1f} min)')
    if dimU == 0: return {'dimU': 0, 'status': 'EXCLUDED', 'blocks': {}}
    T, L = glue_tables(base, m); NPL = (np.float32(N) ** np.arange(m + 2, dtype=np.float32))[L]; res = {}
    for (k, V, R) in blocks:
        if V.shape[1] == 0: continue
        F = []; tb = time.time()
        for nf in SCHEDULE:
            if time.time() > t_end: return None
            F += block_forms(base, m, N, V, k, T, NPL, len(F), nf, seed)
            lmin, margin, s = certify(F, R, t_end)
            ok = lmin > 0 and margin > MARGIN_MIN
            res[str(k)] = {'dim': V.shape[1], 'ok': bool(ok), 'margin': float(margin), 'rotations': s, 'forms': nf}
            if verbose: progress(f'    block k={k} (dim {V.shape[1]}, {nf} forms x {s} rotations): '
                                 f'{"positive definite, margin %.2e" % margin if ok else "no certificate yet (best %.2e)" % margin}'
                                 f'  ({(time.time()-tb)/60:.1f} min)')
            if ok: break
        del F
    ok = all(v['ok'] for v in res.values())
    progress(f'{fam} level {2*m}  N={N}: {"EXCLUDED" if ok else "NOT excluded with these forms"}, '
             f'smallest margin {min(v["margin"] for v in res.values()):.2e}  ({(time.time()-t0)/60:.1f} min)')
    return {'dimU': dimU, 'status': 'EXCLUDED' if ok else 'OPEN', 'blocks': res}

def run(queue, hours, which):
    t_end = time.time() + hours * 3600; rf = os.path.join(OUT, 'results.json')
    progress(f'=== evander {which}: items {queue}, job started (stops after {hours:g} h) ===')
    for (fam, m, N) in queue:
        allres = json.load(open(rf)) if os.path.exists(rf) else {}
        key = f'{fam}_K{2*m}_N{N}'
        if key in allres: progress(f'{key}: already done ({allres[key]["status"]}), skipped'); continue
        r = run_item(fam, m, N, t_end)
        if r is None: break
        allres = json.load(open(rf)) if os.path.exists(rf) else {}
        allres[key] = r; json.dump(allres, open(rf + '.part', 'w'), indent=1); os.replace(rf + '.part', rf)
        if time.time() > t_end: break
    progress(f'=== evander {which}: time is up ===' if time.time() >= t_end else f'=== evander {which}: queue done ===')

def run_test():
    t0 = time.time(); ok = True
    print(f'python {sys.version.split()[0]}, numpy {np.__version__}, cores {os.cpu_count()}', flush=True)
    for (fam, m, N, expect, st) in TESTS:
        r = run_item(fam, m, N, time.time() + 3600, verbose=False)
        good = r is not None and r['dimU'] == expect and r['status'] == st
        print(f'[{time.time()-t0:5.0f}s] {fam} level {2*m}, N={N}: dim U = {r["dimU"] if r else "?"} (expected {expect}), '
              f'{r["status"] if r else "?"} (expected {st})  {"OK" if good else "WRONG"}', flush=True)
        ok &= good
    glue_tables('BR', 6)      # the big gluing table of the night (a few minutes), computed once and saved
    print('SELF-TEST PASSED' if ok else 'SELF-TEST FAILED -- send me the output', flush=True)

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else ''
    if what == 'test': run_test()
    elif what in QUEUES: run(QUEUES[what], float(sys.argv[2]) if len(sys.argv) > 2 else HOURS, what)
    elif what == 'item':      # python3 evander.py item FAM m N
        run([(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))], HOURS, 'item')
    else: print(open(__file__).read().split('import sys')[0])
