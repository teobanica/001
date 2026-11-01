#!/usr/bin/env python3
# sol.py -- solution groups of binary linear systems as seeds, for "On the maximality conjecture for quantum
# permutation groups" (Claude and Teo), Section 3 (Propositions on the magic square and on the A_7 system).
#
#   python3 sol.py magic [kmax]   magic square, real 2-qubit Paulis, N = 18: joint dimensions for k = 4..kmax (default 8)
#   python3 sol.py a7             the A_7 system, N = 210: no nontrivial laws of length <= 8
#
# All blocks are of type Z_2, so we work with patterns (kappa, M) (Corollary "laws lemma, Z_2-blocks"): kappa is the
# partition of the k legs by orbit, M the set of legs carrying the nontrivial label.  The pattern is balanced when every
# block of kappa meets M in an even number of legs, and active when some assignment of DISTINCT generators to the blocks
# of kappa gives a nontrivial ordered word along M.  For a balanced pattern, the coefficient of xi_pi is
# 2^(|pi| - k) if every block of pi lies in a block of kappa and meets M evenly, and 0 otherwise.  The dimension is
# dim span{xi_pi} - rank(rows of the active patterns), with the rank computed modulo p = 1048573 after a random
# compression; so the output is an upper bound, and a Catalan output proves equality.
import sys, itertools, time
import numpy as np

P = 1048573

def set_partitions(k):
    def rec(l):
        if l == 0: yield []; return
        for q in rec(l - 1):
            for i in range(len(q)): yield q[:i] + [q[i] + [l - 1]] + q[i + 1:]
            yield q + [[l - 1]]
    return [sorted(sorted(b) for b in q) for q in rec(k)]

def noncrossing(q):
    for A, B in itertools.permutations(q, 2):
        for a1, a2 in itertools.combinations(A, 2):
            for b1, b2 in itertools.combinations(B, 2):
                if a1 < b1 < a2 < b2: return False
    return True

def rank_mod_p(A, bs=64):
    """Rank modulo P by blocked elimination; all products exact in float64 (entries < 2^20, inner dims <= bs)."""
    A = np.mod(A, P).astype(np.float64); n, m = A.shape; r = 0; c = 0
    while c < m and r < n:
        c1 = min(c + bs, m); w = c1 - c
        panel = A[r:, c:c1].copy(); F = np.zeros((n - r, w)); inv = []; q = 0
        for col in range(w):
            if q >= n - r: break
            nz = np.flatnonzero(panel[q:, col])
            if len(nz) == 0: continue
            pr = q + nz[0]
            if pr != q:
                panel[[q, pr]] = panel[[pr, q]]; F[[q, pr]] = F[[pr, q]]; A[[r + q, r + pr]] = A[[r + pr, r + q]]
            iv = pow(int(panel[q, col]), P - 2, P); inv.append(iv)
            panel[q] = np.mod(panel[q] * iv, P)
            f = panel[q + 1:, col].copy()
            panel[q + 1:] = np.mod(panel[q + 1:] - np.outer(f, panel[q]), P); F[q + 1:, q] = f
            q += 1
        T = A[r:, c1:]
        if T.shape[1]:
            for j in range(q): T[j] = np.mod(np.mod(T[j] - F[j, :j] @ T[:j], P) * inv[j], P)
            if q < n - r: T[q:] = np.mod(T[q:] - F[q:, :q] @ T[:q], P)
        r += q; c = c1
    return r

# ----------------------------------------------------------------------------------------- magic square
def magic_generators():
    X = np.array([[0, 1], [1, 0]]); Z = np.array([[1, 0], [0, -1]]); I = np.eye(2, dtype=np.int64); kr = np.kron
    return [kr(X, I), kr(I, X), kr(X, X), kr(I, Z), kr(Z, I), kr(Z, Z), kr(X, Z), kr(Z, X), kr(X @ Z, X @ Z)]

def active(kappa, M, gens):
    """Some assignment of distinct generators to the blocks of kappa meeting M gives a nontrivial word along M."""
    k = sum(len(b) for b in kappa); blk = [0] * k
    for j, b in enumerate(kappa):
        for l in b: blk[l] = j
    word_blocks = sorted({blk[l] for l in M}); e = np.eye(gens[0].shape[0], dtype=np.int64)
    for assign in itertools.permutations(range(len(gens)), len(word_blocks)):
        g = dict(zip(word_blocks, assign)); w = e
        for l in sorted(M): w = w @ gens[g[blk[l]]]
        if not np.array_equal(w, e): return True
    return False

def joint_dim_z2(k, gens, seed=1, maxrows_factor=4):
    t0 = time.time(); parts = set_partitions(k)
    cross = [q for q in parts if not noncrossing(q)]; ncr = len(cross); N = 2 * len(gens)
    assert N >= k
    dimspan = len(parts)
    maxb = max(len(q) for q in cross)
    Bm = np.zeros((ncr, maxb), dtype=np.int64)
    for i, q in enumerate(cross):
        for j, b in enumerate(q): Bm[i, j] = sum(1 << l for l in b)
    pc = np.array([bin(x).count('1') for x in range(1 << k)])
    rows = []; nkappa = 0
    for kappa in parts:
        if len(kappa) > len(gens): continue
        blockmask = [sum(1 << l for l in b) for b in kappa]
        # M: a subset meeting every block of kappa evenly; M = empty gives the trivial word
        choices = [[S for r in range(0, len(b) + 1, 2) for S in itertools.combinations(b, r)] for b in kappa]
        for Ms in itertools.product(*choices):
            M = [l for S in Ms for l in S]
            if not M or not active(kappa, M, gens): continue
            Mmask = sum(1 << l for l in M)
            rows.append((blockmask, Mmask))
    nb = len(rows); rng = np.random.default_rng(seed)
    if nb > maxrows_factor * (ncr + 20):
        rows = [rows[i] for i in rng.choice(nb, maxrows_factor * (ncr + 20), replace=False)]
    inv2 = [pow(pow(2, P - 2, P), j, P) for j in range(k + 1)]
    R_ = ncr + 20; Mc = np.zeros((R_, ncr))
    allm = np.arange(1 << k)
    for c0 in range(0, len(rows), 512):
        batch = rows[c0:c0 + 512]; b = len(batch)
        V = np.zeros((b, 1 << k), dtype=np.int64); V[:, 0] = 1
        for a, (bmask, Mmask) in enumerate(batch):
            for bm in bmask:     # good masks: nonempty subsets m of a block of kappa with |m & M| even
                sub = allm[(allm & bm) == allm]; sub = sub[sub > 0]
                ev = sub[(pc[sub & Mmask] % 2) == 0]
                V[a, ev] = [inv2[pc[m] - 1] for m in ev]
        vals = np.ones((b, ncr), dtype=np.int64)
        for j in range(maxb): vals = (vals * V[:, Bm[:, j]]) % P
        Rb = rng.integers(0, 1 << 12, size=(R_, b)).astype(np.float64)
        Mc = np.mod(Mc + Rb @ vals.astype(np.float64), P)
    rank = rank_mod_p(Mc)
    return dimspan - rank, nb, time.time() - t0

# ----------------------------------------------------------------------------------------- the A_7 system
def even_partitions(l):
    return [q for q in set_partitions(l) if all(len(b) % 2 == 0 for b in q)]

def freely_trivial(word):
    st = []
    for x in word:
        if st and st[-1] == x: st.pop()
        else: st.append(x)
    return not st

def a7_check(lmax=8, tries=20000, seed=3):
    pts = range(7)
    dts = []
    for a, b, c, d in itertools.permutations(pts, 4):
        if a < b and c < d and a < c:
            p = list(pts); p[a], p[b] = b, a; p[c], p[d] = d, c; dts.append(tuple(p))
    assert len(dts) == 105
    comp = lambda p, q: tuple(p[q[i]] for i in range(7)); idp = tuple(pts)
    rng = np.random.default_rng(seed); total = 0; inactive = []
    for l in range(2, lmax + 1, 2):
        for q in even_partitions(l):
            word = [0] * l
            for j, b in enumerate(q):
                for x in b: word[x] = j
            if freely_trivial(word): continue
            total += 1; found = False
            for _ in range(tries):
                g = rng.choice(105, len(q), replace=False); w = idp
                for x in word: w = comp(w, dts[g[x]])
                if w != idp: found = True; break
            if not found: inactive.append(word)
    return total, inactive

CAT = {4: 14, 5: 42, 6: 132, 7: 429, 8: 1430}
BELL = {4: 15, 5: 52, 6: 203, 7: 877, 8: 4140}

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'magic'
    if what == 'magic':
        kmax = int(sys.argv[2]) if len(sys.argv) > 2 else 8; g = magic_generators()
        for k in range(4, kmax + 1):
            d, nb, secs = joint_dim_z2(k, g)
            print(f'magic square, N=18, k={k}: dim = {d}  (Catalan {CAT[k]}, Bell {BELL[k]})  [{nb} active patterns, {secs:.0f} s]', flush=True)
    elif what == 'a7':
        total, inactive = a7_check()
        print(f'A_7 system: {total} balanced patterns of length <= 8 that are not freely trivial; '
              f'{total - len(inactive)} of them active in A_7' + ('' if not inactive else f'; NOT shown active: {inactive}'))
