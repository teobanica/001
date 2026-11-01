#!/usr/bin/env python3
# freeness.py -- the level method for the freeness conjecture, orthogonal case (Section 12 of "On the maximality
# conjecture for quantum permutation groups", Claude and Teo).
#
#   python3 freeness.py                 table for the six envelopes D and levels k = 3..8  (log: freenessout.txt)
#   python3 freeness.py kmax            same, levels up to kmax
#   python3 freeness.py check           validates the generating sets of lowering maps against brute force (k <= 6)
#   python3 freeness.py one D k [N...]  one entry in detail (D in O B Bp Bs H Sp), kernel, eigenspaces, certificates
#
# Setting.  S_N^+ <= G <= O_N^+ means NC_2 <= C_G <= span(NC).  For N >= 4 the vectors T_pi, pi in NC(k), are
# linearly independent, so C_G(k) is a subspace of the coordinate space C^{NC(k)}.  Let D = {pi : T_pi in C_G}, a
# category of noncrossing partitions containing NC_2, hence one of the seven free easy categories (Weber); G_D is
# the smallest easy quantum group containing G.  If G is not easy, let k be the first level with C_G(k) != span D(k),
# and W = C_G(k) cap span(NC(k) \ D(k)) != 0.  Then
#   (L) for every partition p in D viewed as a map from k points to k' < k points, T_p(W) is in span D(k');
#   (R) W is invariant under the rotation rho and the antilinear involution J (reverse + conjugate);
#   (Q) for x in W and a > k/2, the gluing x o_a Jx (= the vector of T_x T_x^*, T_x : a -> k-a) lies in span D(2(k-a)).
# U = U_D(k) is the space of x in span(NC(k) \ D(k)) satisfying (L): the D-uncappable space.  Everything is
# polynomial in N (no Gram matrices).  Exclusion of (D, k): U = 0, or on every rotation eigenspace U_omega some real
# combination of the quadratic forms  Re coeff_tau(x o_a Jx), tau in NC(2(k-a)) \ D, is positive definite on the
# J-real vectors (Theorem "exclusion criterion" of Section 11, verbatim).
#
# The seven categories NC_2 <= D <= NC:
#   O  NC_2            pairings                              (O_N^+)
#   B  NC_{1,2}        blocks of size <= 2                   (B_N^+)
#   Bp NC_{1,2}^even   blocks <= 2, even number of points    (B_N'^+)
#   Bs NC_{1,2}^#      blocks <= 2, each pair has an even number of singletons on each side  (B_N^{#+})
#   H  NC_even         all blocks even                       (H_N^+)
#   Sp NC^even         even number of points                 (S_N'^+)
#   S  NC              everything                            (S_N^+, nothing to do)
import sys, os, time, itertools, random
from fractions import Fraction
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = open(os.path.join(HERE, 'freenessout.txt'), 'a')
def out(s=''):
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

# ----------------------------------------------------------------------------------------------- partitions
# A partition of {0..k-1} is a tuple of blocks, each block a tuple of increasing ints, blocks sorted by their minimum.
def canon(blocks):
    return tuple(sorted((tuple(sorted(b)) for b in blocks if b), key=lambda b: b[0]))

_NC = {}
def NC(k):
    """All noncrossing partitions of {0..k-1}."""
    if k in _NC: return _NC[k]
    if k == 0: res = [()]
    else:
        res = []
        # block of point 0: 0 = i0 < i1 < ... < ir ; gaps are independent NC partitions
        def rec(start, k):
            # noncrossing partitions of {start..k-1}
            if start >= k: yield (); return
            for sub in itertools.chain.from_iterable(itertools.combinations(range(start + 1, k), r) for r in range(k - start)):
                blk = (start,) + sub
                pts = [start] + list(sub) + [k]
                gaps = [list(gapparts(pts[i] + 1, pts[i + 1])) for i in range(len(pts) - 1)]
                for combo in itertools.product(*gaps):
                    yield (blk,) + tuple(b for part in combo for b in part)
        def gapparts(a, b):
            # noncrossing partitions of {a..b-1}, as tuples of blocks
            if a >= b: yield (); return
            yield from rec(a, b)
        res = [canon(p) for p in rec(0, k)]
        res = sorted(set(res))
    _NC[k] = res
    return res

def is_noncrossing(blocks):
    for b in blocks:
        for c in blocks:
            if b is c: continue
            for a1, a2 in itertools.combinations(b, 2):
                if any(a1 < x < a2 for x in c) and any(x < a1 or x > a2 for x in c): return False
    return True

def rotate(p, k, s=1):
    """rho^s: point i -> i+s mod k."""
    return canon([tuple((i + s) % k for i in b) for b in p])

def reverse(p, k):
    return canon([tuple(k - 1 - i for i in b) for b in p])

def tensor(p, q, k):
    """p on k points followed by q."""
    return canon(list(p) + [tuple(i + k for i in b) for b in q])

# ----------------------------------------------------------------------------------------------- categories
def inD(name, p, k):
    sizes = [len(b) for b in p]
    if name == 'S': return True
    if name == 'O': return all(s == 2 for s in sizes)
    if name == 'B': return all(s <= 2 for s in sizes)
    if name == 'Bp': return all(s <= 2 for s in sizes) and k % 2 == 0
    if name == 'H': return all(s % 2 == 0 for s in sizes)
    if name == 'Sp': return k % 2 == 0
    if name == 'Bs':
        if not all(s <= 2 for s in sizes) or k % 2: return False
        singles = sorted(b[0] for b in p if len(b) == 1)
        for b in p:
            if len(b) == 2:
                inside = sum(1 for s in singles if b[0] < s < b[1])
                if inside % 2: return False
        return True
    raise ValueError(name)

DNAMES = ['O', 'B', 'Bp', 'Bs', 'H', 'Sp']
DTEX = {'O': 'NC_2', 'B': 'NC_{1,2}', 'Bp': 'NC_{1,2}^{even}', 'Bs': 'NC_{1,2}^{#}', 'H': 'NC_{even}', 'Sp': 'NC^{even}', 'S': 'NC'}
DGROUP = {'O': 'O_N^+', 'B': 'B_N^+', 'Bp': "B_N'^+", 'Bs': 'B_N^{#+}', 'H': 'H_N^+', 'Sp': "S_N'^+", 'S': 'S_N^+'}

def check_category(name, kmax):
    """Sanity: D(<=kmax) closed under rotation, reversal, tensor (small), and all caps (adjacent and cyclic), and
    composition p o q for small sizes.  Returns the first offending example or None."""
    for k in range(kmax + 1):
        D = [p for p in NC(k) if inD(name, p, k)]
        for p in D:
            if not inD(name, rotate(p, k), k): return ('rot', p)
            if not inD(name, reverse(p, k), k): return ('rev', p)
            for i in range(k):
                q, loops = cap(p, k, i)
                if q is not None and not inD(name, q, k - 2): return ('cap', p, i)
        for k2 in range(kmax + 1 - k):
            D2 = [q for q in NC(k2) if inD(name, q, k2)]
            for p in D:
                for q in D2:
                    if not inD(name, tensor(p, q, k), k + k2): return ('tensor', p, q)
    # compositions: maps t <- k given by partitions on t+k points, composed with vectors
    for kt in range(1, kmax + 1):
        for t in range(0, kmax + 1 - kt):
            for m in NC(kt + t):
                if not inD(name, m, kt + t): continue
                for x in NC(kt):
                    if not inD(name, x, kt): continue
                    y, loops = compose(m, t, kt, x)
                    if not inD(name, y, t): return ('compose', m, t, kt, x)
    return None

# ----------------------------------------------------------------------------------------------- operations on vectors
def cap(p, k, i):
    """Contract points i and i+1 (mod k).  Returns (partition on k-2 points, number of loops) or (None, 0) if ...
    never None: capping two points of the same block of size 2 gives a loop, of a bigger block no loop."""
    j = (i + 1) % k
    bi = next(b for b in p if i in b); bj = next(b for b in p if j in b)
    loops = 0
    if bi is bj:
        if len(bi) == 2: loops = 1
        new = [b for b in p if b is not bi] + [tuple(x for x in bi if x not in (i, j))]
    else:
        new = [b for b in p if b is not bi and b is not bj] + [tuple(x for x in bi + bj if x not in (i, j))]
    # relabel: remove i and j
    rem = sorted(x for x in range(k) if x not in (i, j))
    idx = {x: n for n, x in enumerate(rem)}
    return canon([tuple(idx[x] for x in b) for b in new if b]), loops

def compose(m, t, k, x):
    """m: a map from k points to t points, given as a noncrossing partition of the t+k cyclic points
    [top_0..top_{t-1}, bot_{k-1}..bot_0] (so that planarity = noncrossing).  x: partition on k points (a vector).
    Returns (partition on t points, loops)."""
    # union-find on t + k (m's points) ; x's point i is identified with m's point t + (k-1-i)
    parent = list(range(t + k))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for b in m:
        for a in b[1:]: union(b[0], a)
    for b in x:
        for a in b[1:]: union(t + k - 1 - b[0], t + k - 1 - a)
    groups = {}
    for a in range(t + k): groups.setdefault(find(a), []).append(a)
    outblocks, loops = [], 0
    for g in groups.values():
        top = [a for a in g if a < t]
        if top: outblocks.append(tuple(top))
        else: loops += 1
    return canon(outblocks), loops

def glue(x_p, y_p, k, a):
    """x o_a y : glue the last a points of x to the first a points of y, rainbow; result on 2(k-a) points, ordered
    x_0..x_{k-a-1}, y_{k-1}..y_{a} ... wait: rainbow glues x_{k-1} with y_0, x_{k-2} with y_1, ..., x_{k-a} with y_{a-1}.
    Remaining points: x_0..x_{k-a-1}, then y_a..y_{k-1} (in this order, planar).  Returns (partition, loops)."""
    n = 2 * k
    parent = list(range(n))
    def find(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]; q = parent[q]
        return q
    def union(p1, p2):
        r1, r2 = find(p1), find(p2)
        if r1 != r2: parent[r1] = r2
    for b in x_p:
        for c in b[1:]: union(b[0], c)
    for b in y_p:
        for c in b[1:]: union(k + b[0], k + c)
    for i in range(a): union(k - 1 - i, k + i)
    groups = {}
    for q in range(n): groups.setdefault(find(q), []).append(q)
    keep = list(range(k - a)) + list(range(k + a, 2 * k))
    idx = {q: n2 for n2, q in enumerate(keep)}
    outblocks, loops = [], 0
    for g in groups.values():
        pts = [idx[q] for q in g if q in idx]
        if pts: outblocks.append(tuple(pts))
        else: loops += 1
    return canon(outblocks), loops

# ----------------------------------------------------------------------------------------------- lowering maps
def all_maps(name, k, t):
    """All p in D viewed as maps k -> t: noncrossing partitions of t+k cyclic points in D."""
    return [m for m in NC(t + k) if inD(name, m, t + k)]

def outer_blocks(p, k):
    """Blocks of p not nested inside another block."""
    res = []
    for b in p:
        nested = any(c is not b and c[0] < b[0] and b[-1] < c[-1] for c in p)
        if not nested: res.append(b)
    return res

def generating_maps(name, k):
    """A generating set of lowering maps k -> t < k, as (t, m).  Contractions: a bottom partition pi, a subset of its
    outer blocks goes through to the top with 1 or 2 top points (at most one block with 2), plus optional top-only
    singletons (needed for Bs); every block must be in D and the whole map in D.  For the block-closed categories
    O, B, H the single-top merges / killers suffice (proof: peel bottom-only blocks, split through-blocks);
    for Bp, Sp parity is restored by one block with two top points; for Bs the top singletons are needed.
    All maps are closed under rotation of the bottom, so cyclic positions are covered."""
    res = []
    seen = set()
    for pi in NC(k):
        outer = outer_blocks(pi, k)
        nout = len(outer)
        for r in range(0, nout + 1):
            for through in itertools.combinations(range(nout), r):
                # tops: 1 each, optionally one of them 2
                top_opts = [[1] * r]
                for j in range(r):
                    v = [1] * r; v[j] = 2; top_opts.append(v)
                if r == 0: top_opts = [[]]
                for tops in top_opts:
                    # top-only singletons: s of them, distributed in r+1 gaps (ordered), only if '1' blocks allowed
                    maxs = 0
                    if name in ('Bs', 'B', 'Bp', 'Sp', 'S'):
                        maxs = k - 1 - sum(tops)
                        if maxs < 0: maxs = 0
                        if name in ('B', 'S'): maxs = 0        # block-closed: not needed
                    for s in range(0, maxs + 1):
                        if sum(tops) + s >= k: continue
                        for gaps in compositions(s, r + 1):
                            # build the map: top points in order: gaps[0] singles, block0 tops, gaps[1] singles, ...
                            t = sum(tops) + s
                            # cyclic points: top_0..top_{t-1}, then bot_{k-1}..bot_0 at indices t + (k-1-i)
                            blocks = []
                            pos = 0
                            topassign = {}
                            for j in range(r):
                                for _ in range(gaps[j]): blocks.append((pos,)); pos += 1
                                topassign[j] = list(range(pos, pos + tops[j])); pos += tops[j]
                            for _ in range(gaps[r]): blocks.append((pos,)); pos += 1
                            assert pos == t
                            thr = set(through)
                            for bi, b in enumerate(pi):
                                bot = [t + (k - 1 - i) for i in b]
                                j = None
                                for jj, o in enumerate(outer):
                                    if o is b and jj in thr: j = list(through).index(jj)
                                if j is not None: blocks.append(tuple(topassign[j] + bot))
                                else: blocks.append(tuple(bot))
                            m = canon(blocks)
                            if not inD(name, m, t + k): continue
                            if (t, m) in seen: continue
                            seen.add((t, m)); res.append((t, m))
    return res

def compositions(s, parts):
    if parts == 1: yield (s,); return
    for a in range(s + 1):
        for rest in compositions(s - a, parts - 1): yield (a,) + rest

# ----------------------------------------------------------------------------------------------- linear algebra mod p
P1, P2 = 2147483647, 1000000007

def rank_mod(rows, ncols, p):
    """rows: list of dicts col -> int. Returns rank mod p."""
    M = [dict((c, v % p) for c, v in r.items() if v % p) for r in rows]
    M = [r for r in M if r]
    rank = 0
    pivots = {}
    for r in M:
        r = dict(r)
        while r:
            c = min(r)
            if c in pivots:
                pr, pv = pivots[c]
                f = r[c] * pow(pr[c], p - 2, p) % p
                for cc, vv in pr.items():
                    r[cc] = (r.get(cc, 0) - f * vv) % p
                    if r[cc] == 0: del r[cc]
            else:
                pivots[c] = (r, r[c]); rank += 1; break
    return rank

def kernel_rational(rows, ncols):
    """Exact kernel over Q of the system rows (dicts col->Fraction/int). Returns list of basis vectors (lists of Fraction)."""
    M = [[Fraction(0)] * ncols for _ in rows]
    for i, r in enumerate(rows):
        for c, v in r.items(): M[i][c] = Fraction(v)
    piv = []
    ri = 0
    for c in range(ncols):
        pr = None
        for i in range(ri, len(M)):
            if M[i][c] != 0: pr = i; break
        if pr is None: continue
        M[ri], M[pr] = M[pr], M[ri]
        inv = 1 / M[ri][c]
        M[ri] = [v * inv for v in M[ri]]
        for i in range(len(M)):
            if i != ri and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[ri])]
        piv.append(c); ri += 1
        if ri == len(M): break
    free = [c for c in range(ncols) if c not in piv]
    basis = []
    for fc in free:
        v = [Fraction(0)] * ncols; v[fc] = Fraction(1)
        for i, pc in enumerate(piv): v[pc] = -M[i][fc]
        basis.append(v)
    return basis

# ----------------------------------------------------------------------------------------------- the (D, k) problem
class Level:
    def __init__(self, name, k, maps=None):
        self.name, self.k = name, k
        self.ncs = NC(k)
        self.D = [p for p in self.ncs if inD(name, p, k)]
        self.off = [p for p in self.ncs if not inD(name, p, k)]
        self.col = {p: i for i, p in enumerate(self.off)}
        self.maps = maps if maps is not None else generating_maps(name, k)
        # constraint rows as polynomials in N: list of (t, dict target -> dict col -> (exponent -> coeff)) ; we store
        # per constraint a dict col -> exponent (each T_p T_pi is a single monomial N^loops T_sigma)
        self.rows = []   # list of dict col -> loops
        # rows: for every map m and every rotation rho^s (W is rotation invariant), the coefficient of each sigma
        # not in D in T_m(rho^s x) must vanish; each T_m T_pi is a single monomial N^loops T_sigma
        seen = set()
        for t, m in self.maps:
            for s in range(k):
                targets = {}
                for p in self.off:
                    sigma, loops = compose(m, t, k, rotate(p, k, s))
                    if inD(name, sigma, t): continue
                    targets.setdefault(sigma, {})[self.col[p]] = loops
                for r in targets.values():
                    key = tuple(sorted(r.items()))
                    if key not in seen:
                        seen.add(key); self.rows.append(r)

    def rank_at(self, N, p=P1):
        rows = [{c: pow(N, e, p) for c, e in r.items()} for r in self.rows]
        return rank_mod(rows, len(self.off), p)

    def dimU(self, N, p=P1):
        return len(self.off) - self.rank_at(N, p)

    def kernel_at(self, N):
        rows = [{c: Fraction(N) ** e for c, e in r.items()} for r in self.rows]
        return kernel_rational(rows, len(self.off))

    # rotation and J on the off-coordinates
    def rot_matrix(self):
        n = len(self.off)
        R = np.zeros((n, n))
        for p in self.off:
            R[self.col[rotate(p, self.k)], self.col[p]] = 1
        return R

    def rev_perm(self):
        return [self.col[reverse(p, self.k)] for p in self.off]

    def quadratic_forms(self, N):
        """List of (a, tau, bilinear coefficient dict): coeff_tau(x o_a y') where y' = Jy = reverse(conj y).
        As a function of (x, y): sum over pairs (p, q) of off-partitions x_p conj(y_{rev q}) N^loops [glue(p, q) = tau].
        Returns list of complex-free matrices B (n x n) with B[p, q'] where q' = col(rev q): coeff = x^T B conj(y)."""
        k, n = self.k, len(self.off)
        forms = []
        for a in range((k + 1) // 2, k):
            tdeg = 2 * (k - a)
            if tdeg < 2: continue
            mats = {}
            for p in self.off:
                for q in self.off:
                    tau, loops = glue(p, q, k, a)
                    if inD(self.name, tau, tdeg): continue
                    # same level (tdeg == k): the off-D part must lie in W, a subspace of U: handled in certificates
                    B = mats.setdefault(tau, np.zeros((n, n)))
                    # y' = Jy has coordinates y'_q = conj(y_{rev q}); so coefficient x_p * y'_q = x_p conj(y_{rev q})
                    B[self.col[p], self.col[reverse(q, k)]] += float(N) ** loops
            for tau, B in mats.items(): forms.append((a, tau, B))
        return forms

def certificates(L, N, verbose=False, tries=200):
    """Rotation eigenspaces of U at N, J-real forms, positivity search.  Returns list of (omega_index, dimU_omega,
    dim real, best min eigenvalue, status, number of forms)."""
    k = L.k
    basis = L.kernel_at(N)
    d = len(basis)
    if d == 0: return [], 'EXCLUDED (U = 0)'
    if d == 1 and sum(1 for v in basis[0] if v != 0) == 1:
        return [(0, 1, 1, 0.0, 'single partition vector', 0)], 'EXCLUDED (U = C T_pi)'
    n = len(L.off)
    Umat = np.array([[float(v) for v in b] for b in basis]).T   # n x d
    Umat, _ = np.linalg.qr(Umat)
    R = L.rot_matrix()
    C = Umat.T @ R @ Umat
    assert np.allclose(Umat @ C, R @ Umat, atol=1e-7), 'U not rotation invariant?'
    Pperp = np.eye(n) - Umat @ Umat.T          # projection onto the annihilator of U (off-D coordinates mod U)
    evals, evecs = np.linalg.eig(C)
    revp = L.rev_perm()
    forms = L.quadratic_forms(N)
    results = []
    allok = True
    omegas = [np.exp(2j * np.pi * j / k) for j in range(k)]
    for j, om in enumerate(omegas):
        sel = [i for i in range(d) if abs(evals[i] - om) < 1e-6]
        if not sel: continue
        V = Umat @ evecs[:, sel]            # n x dj complex, basis of U_omega
        dj = len(sel)
        JV = np.conj(V)[revp, :]
        A = np.concatenate([np.concatenate([(V - JV).real, -((V + JV).imag)], axis=1),
                            np.concatenate([(V - JV).imag, (V + JV).real], axis=1)], axis=0)
        u, sv, vt = np.linalg.svd(A)
        null = vt[np.sum(sv > 1e-8):].T
        r = null.shape[1]
        cs = null[:dj] + 1j * null[dj:]
        X = V @ cs                          # n x r, J-real vectors (real form basis)
        Q = []
        for a, tau, B in forms:
            if 2 * (k - a) == k:
                # same level: coefficient tau of x o Jx is one coordinate; take the projection mod U of the whole
                # vector instead: collect per a below
                continue
            M = X.T @ B @ np.conj(X)
            M = (M + M.T) / 2
            if np.linalg.norm(M.real) > 1e-9: Q.append(M.real)
            if np.linalg.norm(M.imag) > 1e-9: Q.append(M.imag)
        if k % 2 == 0:
            # same-level gluing a = k/2: vector z(x) = x o_{k/2} Jx in off-coordinates (full, all tau), then Pperp z = 0
            Bs = {}
            for a, tau, B in forms:
                if 2 * (k - a) == k: Bs[tau] = B
            if Bs:
                Z = np.zeros((n, r, r), dtype=complex)
                for tau, B in Bs.items():
                    M = X.T @ B @ np.conj(X); Z[L.col[tau]] = (M + M.T) / 2
                PZ = np.tensordot(Pperp, Z, axes=([1], [0]))    # n x r x r
                for i in range(n):
                    if np.linalg.norm(PZ[i].real) > 1e-9: Q.append(PZ[i].real)
                    if np.linalg.norm(PZ[i].imag) > 1e-9: Q.append(PZ[i].imag)
        G = (X.T @ np.conj(X)).real
        Gi = np.linalg.inv(np.linalg.cholesky(G))
        Q = [Gi @ q @ Gi.T for q in Q]
        best, status = -np.inf, 'open'
        if r == 0: status = 'no J-real vectors?'
        elif not Q: status = 'no forms'
        else:
            best = find_pd(Q, tries)
            status = 'EXCLUDED' if best > 1e-7 else 'open'
        if status != 'EXCLUDED': allok = False
        results.append((j, dj, r, best, status, len(Q)))
        if verbose: out(f'    omega = exp(2 pi i {j}/{k}): dim U_omega = {dj}, real form {r}, {len(Q)} forms, best min eig {best:.3e} -> {status}')
    return results, ('EXCLUDED' if allok else 'open')

def find_pd(Q, tries):
    """max over ||c|| <= 1 of lambda_min(sum c_j Q_j): concave, projected subgradient ascent with a few restarts.
    Returns the best lambda_min relative to the largest |eigenvalue| (so > 0 means positive definite)."""
    m = len(Q); r = Q[0].shape[0]
    if r == 1:
        return max(abs(q[0, 0]) for q in Q)
    Qs = np.array(Q)                       # m x r x r
    rng = np.random.default_rng(0)
    best = -np.inf
    for trial in range(min(tries, 8)):
        c = rng.standard_normal(m); c /= np.linalg.norm(c)
        for it in range(1, 400):
            S = np.tensordot(c, Qs, axes=1)
            ev, V = np.linalg.eigh(S)
            v = V[:, 0]
            g = np.einsum('i,kij,j->k', v, Qs, v)     # subgradient of lambda_min
            c = c + g / (np.linalg.norm(g) + 1e-15) * (0.3 / np.sqrt(it))
            n = np.linalg.norm(c)
            if n > 1: c /= n
        S = np.tensordot(c, Qs, axes=1)
        ev = np.linalg.eigvalsh(S)
        val = ev[0] / max(1e-12, np.max(np.abs(ev)))
        best = max(best, val)
        if best > 1e-3: break
    return best

# ----------------------------------------------------------------------------------------------- drivers
def check(kmax=6):
    out('=== check: categories closed, generating sets vs brute force ===')
    for name in DNAMES:
        bad = check_category(name, min(kmax, 6))
        out(f'{name:3s} closure check (k <= {min(kmax,6)}): {"ok" if bad is None else bad}')
    for name in DNAMES:
        for k in range(3, kmax + 1):
            t0 = time.time()
            gen = generating_maps(name, k)
            full = [(t, m) for t in range(k) for m in all_maps(name, k, t)]
            Lg = Level(name, k, gen); Lf = Level(name, k, full)
            dims = []
            for N in (4, 5, 6, 7, 11):
                dg, df = Lg.dimU(N), Lf.dimU(N)
                dims.append((N, dg, df))
            ok = all(dg == df for _, dg, df in dims)
            out(f'{name:3s} k={k}: |NC|={len(Lg.ncs)} |D|={len(Lg.D)} gen maps {len(gen)} rows {len(Lg.rows)} | full maps {len(full)} rows {len(Lf.rows)} | dim U: '
                + ' '.join(f'N={N}:{dg}/{df}' for N, dg, df in dims) + (' OK' if ok else ' MISMATCH') + f'  ({time.time()-t0:.1f}s)')

def table(kmax=8, Ns=(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 20)):
    out(f'=== freeness levels, orthogonal case, k = 3..{kmax} ===')
    out('D   k  |NC(k)| |D(k)|  rows  dim U at N = ' + ' '.join(str(N) for N in Ns) + ' | generic | certificates at N (per rotation eigenspace)')
    for name in DNAMES:
        for k in range(3, kmax + 1):
            t0 = time.time()
            L = Level(name, k)
            dims = [L.dimU(N) for N in Ns]
            gen = L.dimU(random.randrange(10 ** 6, 10 ** 8), P2)
            line = f'{name:3s} {k:2d} {len(L.ncs):7d} {len(L.D):6d} {len(L.rows):6d}  ' + ' '.join(f'{d:3d}' for d in dims) + f' | {gen:3d} |'
            if gen == 0 and all(d == 0 for d in dims):
                line += ' EXCLUDED (U = 0)'
                out(line + f'  ({time.time()-t0:.0f}s)'); continue
            if gen == 1 and all(d == 1 for d in dims):
                b = L.kernel_at(5)[0]
                if sum(1 for v in b if v != 0) == 1:
                    line += ' EXCLUDED (U = C T_pi, pi = %s)' % (L.off[[i for i, v in enumerate(b) if v][0]],)
                    out(line + f'  ({time.time()-t0:.0f}s)'); continue
            # certificates at a few N
            certs = []
            for N in (4, 5, 7, 10):
                if L.dimU(N) == 0: certs.append(f'N={N}:U=0'); continue
                res, st = certificates(L, N)
                certs.append(f'N={N}:' + st + '[' + ','.join(f'{dj}' for (_, dj, _, _, _, _) in res) + ']' + ('' if st == 'EXCLUDED' else '{' + ','.join(f'{b:.1e}' for (_, _, _, b, s, _) in res if s != 'EXCLUDED') + '}'))
            out(line + ' ' + ' '.join(certs) + f'  ({time.time()-t0:.0f}s)')

def one(name, k, Ns):
    L = Level(name, k)
    out(f'=== {name} ({DTEX[name]}, envelope {DGROUP[name]}), level {k} ===')
    out(f'|NC({k})| = {len(L.ncs)}, |D({k})| = {len(L.D)}, unknowns {len(L.off)}, generating maps {len(L.maps)}, rows {len(L.rows)}')
    for N in Ns:
        d = L.dimU(N)
        out(f'N = {N}: dim U = {d}')
        if d:
            res, st = certificates(L, N, verbose=True)
            out(f'  -> {st}')

if __name__ == '__main__':
    args = sys.argv[1:]
    out(f'--- freeness.py {" ".join(args)}  {time.strftime("%d %b %H:%M")} ---')
    if args and args[0] == 'check': check(int(args[1]) if len(args) > 1 else 6)
    elif args and args[0] == 'one': one(args[1], int(args[2]), [int(a) for a in args[3:]] or [4, 5, 7])
    else: table(int(args[0]) if args else 8)
