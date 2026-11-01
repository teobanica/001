#!/usr/bin/env python3
# twistpu.py -- the twists PU_n^sigma of the projective unitary groups, for "On the maximality conjecture for quantum
# permutation groups" (Claude and Teo), Sections 4.6-4.7 (Propositions 4.12 and 4.15).
#
#   python3 twistpu.py            all the lines of Proposition 4.15 (a few minutes on a laptop)
#   python3 twistpu.py Z3 7       one line: H = Z2, Z3, Z4, Z2xZ2 or Z5, up to the given level
#
# Setting.  H finite abelian of order n, Lambda = H x H^ of order N = n^2, Weyl matrices W_(i,a), cocycle
# c((i,a),(j,b)) = b(i), twist sigma = conj(c).  In the basis W_alpha of C^N_sigma = C[Lambda]:
#   * xi_pi has coefficient N^{|pi|-k} on the tuples alpha in Lambda^k balanced on the blocks of pi;
#   * Fix_{PU_n^sigma}(k) = Theta(span{c_tau : tau in S_k}) (Proposition 4.12), where c_tau(alpha) is the coefficient
#     of the leg permutation P_tau on W_alpha1 (x) ... (x) W_alphak, a product over the cycles of tau of normalized
#     traces of products of Weyl matrices, and Theta(alpha) = prod_{s<t} sigma(alpha_s, alpha_t).
# So dim Fix_{<S_N, PU_n^sigma>}(k) = dim V + dim W - rank [V | Theta W].  All vectors involved are invariant under the
# symplectic group Sp(Lambda) acting diagonally (checked on random samples below), so the rank is computed on the
# rows indexed by orbit representatives.  Everything is exact modulo p = 1048261 = 1 mod 60, after a random
# compression (the rank can only drop: the output is an upper bound, and a Catalan output is a proof).
import sys, itertools, time
import numpy as np

P = 1048261
SIGN = +1      # Theta = prod_{s<t} c(alpha_s, alpha_t)^SIGN; +1 is the convention matching the coefficients c_tau below

def factor(n):
    f, d = set(), 2
    while d * d <= n:
        while n % d == 0: f.add(d); n //= d
        d += 1
    if n > 1: f.add(n)
    return f
GEN = next(g for g in range(2, P) if all(pow(g, (P - 1) // q, P) != 1 for q in factor(P - 1)))
def omega(m): return pow(GEN, (P - 1) // m, P)

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

def stirling2(k, j):
    S = [[0] * (j + 1) for _ in range(k + 1)]; S[0][0] = 1
    for n in range(1, k + 1):
        for m in range(1, j + 1): S[n][m] = m * S[n - 1][m] + S[n - 1][m - 1]
    return S[k][j]

def cycles(t):
    seen = [False] * len(t); out = []
    for i in range(len(t)):
        if seen[i]: continue
        c = []; j = i
        while not seen[j]: seen[j] = True; c.append(j); j = t[j]
        out.append(c)
    return out

def lis_free(t, n):
    """No increasing subsequence of length n + 1."""
    best = []
    import bisect
    for x in t:
        i = bisect.bisect_left(best, x)
        if i == len(best): best.append(x)
        else: best[i] = x
    return len(best) <= n

def rank_mod_p(A, bs=64):
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

class Weyl:
    """H = Z_{d1} x ... ; Lambda = H x H^, elements as integer vectors (i_1..i_r, a_1..a_r)."""
    def __init__(self, orders):
        self.o = list(orders); r = len(orders); self.r = r; self.n = int(np.prod(orders)); self.N = self.n ** 2
        self.M = int(np.lcm.reduce(orders)); self.w = omega(self.M)
        self.els = [tuple(v) for v in itertools.product(*[range(d) for d in self.o + self.o])]
        self.idx = {v: i for i, v in enumerate(self.els)}
        self.E = np.array(self.els, dtype=np.int64)
        mod = np.array(self.o + self.o)
        self.add = np.array([[self.idx[tuple((self.E[x] + self.E[y]) % mod)] for y in range(self.N)] for x in range(self.N)])
        self.neg = np.array([self.idx[tuple((-self.E[x]) % mod)] for x in range(self.N)])
        # c((i,a),(j,b)) = b(i) = exp(2 pi i sum_x b_x i_x / d_x), as an exponent of w = exp(2 pi i / M)
        sc = np.array([self.M // d for d in self.o])
        I = self.E[:, :r]; A = self.E[:, r:]
        self.cexp = (np.einsum('xk,yk,k->xy', I, A, sc)) % self.M          # cexp[x, y] = exponent of c(x, y)
        self.zero = self.idx[tuple([0] * (2 * r))]
    def symplectic(self):
        """The automorphisms of Lambda preserving eps(x,y) = c(x,y) conj(c(y,x)): all automorphisms are tried, as
        matrices over the product of the cyclic groups (fine for the small cases here)."""
        eps = (self.cexp - self.cexp.T) % self.M
        gens = []; mod = np.array(self.o + self.o); dim = 2 * self.r
        basis = [self.idx[tuple(int(i == j) for j in range(dim))] for i in range(dim)]
        out = []
        for imgs in itertools.product(range(self.N), repeat=dim):
            # the map sending the standard generators to imgs, if it is well defined and bijective
            ok = all(((mod[i] * self.E[imgs[i]]) % mod == 0).all() for i in range(dim))
            if not ok: continue
            img = np.array([self.idx[tuple((sum(self.E[x][i] * self.E[imgs[i]] for i in range(dim))) % mod)] for x in range(self.N)])
            if len(set(img.tolist())) != self.N: continue
            if (eps[np.ix_(img, img)] == eps).all(): out.append(img)
        return out

def chain_phase(W, labels):
    """Exponent (of w) of the phase in W_l1 W_l2 ... W_lj = phase * W_{sum}; returns (sum, exponent)."""
    s = labels[:, 0].copy(); e = np.zeros(len(labels), dtype=np.int64)
    for j in range(1, labels.shape[1]):
        e = (e + W.cexp[s, labels[:, j]]) % W.M; s = W.add[s, labels[:, j]]
    return s, e

def joint(orders, k, seed=1, check_sp=True, use_sp=True):
    t0 = time.time(); W = Weyl(orders); n, N = W.n, W.N
    # coordinates: tuples with total sum 0 (all vectors involved are supported there)
    tups = np.array([t for t in itertools.product(range(N), repeat=k - 1)], dtype=np.int64).reshape(-1, k - 1)
    last = np.zeros(len(tups), dtype=np.int64); s = np.full(len(tups), W.zero)
    for j in range(k - 1): s = W.add[s, tups[:, j]]
    tups = np.hstack([tups, W.neg[s][:, None]])
    sp = W.symplectic() if use_sp else None
    if sp is not None and len(sp) > 1:
        # any subset S of Sp(Lambda) gives a valid reduction: equal codes min_{g in S} g(t) imply equal Sp-orbits
        S = sp if len(sp) <= 48 else [sp[i] for i in np.random.default_rng(2).choice(len(sp), 48, replace=False)]
        best = None
        for g in S:
            c = np.zeros(len(tups), dtype=np.int64)
            for j in range(k): c = c * N + g[tups[:, j]]
            best = c if best is None else np.minimum(best, c)
        _, rep = np.unique(best, return_index=True); rows = tups[np.sort(rep)]
    else:
        rows = tups
    parts = set_partitions(k); perms = list(itertools.permutations(range(k)))
    dimV = sum(stirling2(k, j) for j in range(1, min(k, N) + 1))
    dimW = sum(1 for t in perms if lis_free(t, n)); ncat = sum(1 for q in parts if noncrossing(q))
    R_ = dimV + dimW - ncat + 20
    invN = pow(N, P - 2, P); invn = pow(n, P - 2, P); wp = np.array([pow(W.w, e, P) for e in range(W.M)], dtype=np.int64)
    # W^*_x = conj(c(-x, x)) W_{-x}: exponent of the phase
    star = (-W.cexp[W.neg, np.arange(N)]) % W.M
    def block(R):                                 # the columns [V | Theta W] on the rows R (mod P)
        T = len(R); X = np.zeros((T, len(parts) + len(perms)))
        for i, q in enumerate(parts):
            ok = np.ones(T, dtype=bool)
            for b in q:
                s = np.full(T, W.zero)
                for l in b: s = W.add[s, R[:, l]]
                ok &= s == W.zero
            X[ok, i] = pow(invN, k - len(q), P)
        th = np.zeros(T, dtype=np.int64)          # Theta = prod_{s<t} sigma(alpha_s, alpha_t), sigma = conj(c)
        for a in range(k):
            for b in range(a + 1, k): th = (th + SIGN * W.cexp[R[:, a], R[:, b]]) % W.M
        for i, t in enumerate(perms):
            e = th.copy(); ok = np.ones(T, dtype=bool); cyc = cycles(t)
            for cy in cyc:
                lab = W.neg[R[:, cy]]             # W^*_{alpha} = phase * W_{-alpha}
                e = (e + star[R[:, cy]].sum(axis=1)) % W.M
                ssum, ph = chain_phase(W, lab); ok &= ssum == W.zero; e = (e + ph) % W.M
            X[ok, len(parts) + i] = (wp[e[ok]] * pow(invn, k - len(cyc), P)) % P
        return X
    if check_sp and sp is not None and len(sp) > 1:     # the Sp(Lambda)-invariance used for the reduction
        rng0 = np.random.default_rng(5); samp = tups[rng0.choice(len(tups), min(200, len(tups)), replace=False)]
        X0 = block(samp)
        for g in [sp[i] for i in rng0.choice(len(sp), min(6, len(sp)), replace=False)]:
            assert np.array_equal(block(g[samp]), X0), 'Sp(Lambda)-invariance fails'
    rng = np.random.default_rng(seed); Mc = np.zeros((R_, len(parts) + len(perms)))
    for c0 in range(0, len(rows), 2048):
        X = block(rows[c0:c0 + 2048]); Rb = rng.integers(0, 1 << 12, size=(R_, len(X))).astype(np.float64)
        Mc = np.mod(Mc + Rb @ X, P)
    rank = rank_mod_p(Mc)
    return dimV + dimW - rank, dimW, len(rows), len(sp) if sp is not None else 1, time.time() - t0

CASES = {'Z2': ((2,), 6), 'Z3': ((3,), 7), 'Z4': ((4,), 6), 'Z2xZ2': ((2, 2), 6), 'Z5': ((5,), 5)}
CAT = {4: 14, 5: 42, 6: 132, 7: 429}

if __name__ == '__main__':
    todo = list(CASES) if len(sys.argv) < 2 else [sys.argv[1]]
    for name in todo:
        orders, kmax = CASES[name]
        if len(sys.argv) > 2: kmax = int(sys.argv[2])
        for k in range(4, kmax + 1):
            d, dW, nrows, nsp, s = joint(orders, k)
            print(f'H={name}, N={int(np.prod(orders))**2}, k={k}: joint dim {d} (Catalan {CAT[k]}), '
                  f'dim Fix_PU^sigma = {dW}  [{nrows} rows, |Sp| = {nsp}, {s:.0f} s]', flush=True)
