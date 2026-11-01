#!/usr/bin/env python3
# orthotwists.py -- the sign twists O_N^eps and the theta-twists of O_N, Section 10.6 of "On the maximality conjecture
# for quantum permutation groups" (Claude and Teo).
#
#   python3 orthotwists.py signs      all graphs eps on N = 3, 4 vertices (levels 4, 6, 8) and N = 5 (levels 4, 6):
#                                     <O_N, O_N^eps> is O_N (empty graph), O_N^* (complete bipartite) or O_N^+ (otherwise)
#   python3 orthotwists.py theta      theta-twists at N = 4, 6 for theta in {1/2, 1/3, 1/4, 1/5, 1/6, 2/5, sqrt2 - 1, random}
#   python3 orthotwists.py            both; log in orthotwistsout.txt
#
# Sign twists.  Twisting O_N by its diagonal subgroup Z_2^N with the bicharacter beta(e_i, e_j) = -1 for i < j an edge of
# eps (and +1 otherwise) gives C_{O_N^eps}(k) = Theta_eps Br(k), with Theta_eps e_i = (-1)^{#{s<t : i_s < i_t, {i_s,i_t} in eps}} e_i.
# So dim C_{<O_N, O_N^eps>}(k) = dim(Br cap Theta Br) = 2 dim Br - rank [Br | Theta Br], exactly, modulo a large prime.
# Theta-twists.  In the weight basis of the maximal torus T^{N/2} of O_N (letters +-e_j, j <= N/2, with e_j conjugate
# to -e_j), a pairing vector is the sum of the weight words with w_t = -w_s on each pair, and the twist by the
# bicharacter B(e_j, e_l) = theta for j < l (and 0 for j >= l) acts by Theta(w) = prod_{s<t} exp(2 pi i B(w_s, w_t)); with
# this convention theta = 1/2 is the complete bipartite sign twist between the planes.  Same formula for the joint
# dimension, in floating point.  Expected values: Br, HL, TL dimensions at the given N.
import sys, time, itertools, numpy as np

log = open('orthotwistsout.txt', 'a')
def out(s):
    print(s, flush=True); log.write(s + '\n'); log.flush()

def pairings(k):
    if k == 0: return [()]
    res = []
    for j in range(1, k):
        rest = [i for i in range(1, k) if i != j]
        for p in pairings(k - 2): res.append(((0, j),) + tuple((rest[a], rest[b]) for a, b in p))
    return res
def noncrossing(p): return not any(a < c < b < d or c < a < d < b for (a, b) in p for (c, d) in p)
def halflib(p): return all((a - b) % 2 for a, b in p)

P_MOD = 2147483647
def rank_mod(M, p=P_MOD):
    M = np.array(M, dtype=np.int64) % p; n, m = M.shape; r = 0
    for c in range(m):
        nz = np.flatnonzero(M[r:, c])
        if len(nz) == 0: continue
        pr = r + nz[0]; M[[r, pr]] = M[[pr, r]]
        M[r] = (M[r] * pow(int(M[r, c]), p - 2, p)) % p
        others = np.flatnonzero(M[:, c]); others = others[others != r]
        if len(others): M[others] = (M[others] - (M[others, c][:, None] * M[r][None, :]) % p) % p
        r += 1
        if r == n: break
    return r
def rank_float(M, tol=1e-8):
    s = np.linalg.svd(M, compute_uv=False); return int((s > tol * s[0]).sum()) if len(s) else 0

def pairing_vectors(N, k, conj):
    """columns xi_pi in the basis of words over N letters, with the involution conj on letters"""
    P = pairings(k); words = np.array(list(itertools.product(range(N), repeat=k)), dtype=np.int64)
    cw = conj[words]; X = np.ones((len(words), len(P)), dtype=np.int64)
    for a, p in enumerate(P):
        for s, t in p: X[:, a] &= (words[:, t] == cw[:, s]).astype(np.int64)
    return P, words, X

def joint(X, theta_diag, exact):
    """dim(V cap Theta V), V = column span of X, Theta diagonal (vector), via 2 dim V - rank [X | Theta X]"""
    if exact:
        dimV = rank_mod(X.T @ X)                                       # Gram, exact
        Y = np.hstack([X, theta_diag[:, None] * X]); return 2 * dimV - rank_mod(Y.T @ Y)
    dimV = rank_float(X.astype(complex)); Y = np.hstack([X, theta_diag[:, None] * X]); return 2 * dimV - rank_float(Y)

def expected(X, P):
    hl = [i for i, p in enumerate(P) if halflib(p)]; nc = [i for i, p in enumerate(P) if noncrossing(p)]
    G = X.T @ X
    return {'Br': rank_mod(G), 'HL': rank_mod(G[np.ix_(hl, hl)]), 'TL': rank_mod(G[np.ix_(nc, nc)])}

def graphs(N):
    """all graphs on N vertices up to isomorphism, as edge sets"""
    pairs = list(itertools.combinations(range(N), 2)); seen = set(); res = []
    for bits in range(1 << len(pairs)):
        E = frozenset(pairs[i] for i in range(len(pairs)) if bits >> i & 1)
        can = min(tuple(sorted(tuple(sorted((q[a], q[b]))) for a, b in E)) for q in itertools.permutations(range(N)))
        if can not in seen: seen.add(can); res.append(sorted(E))
    return res
def complete_bipartite(E, N):
    if not E: return True
    # bipartite with all cross edges present
    col = {}; adj = {i: set() for i in range(N)}
    for a, b in E: adj[a].add(b); adj[b].add(a)
    for s in range(N):
        if s in col or not adj[s]: continue
        col[s] = 0; stack = [s]
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y in col:
                    if col[y] == col[x]: return False
                else: col[y] = 1 - col[x]; stack.append(y)
    if len(col) < N: return False                      # an isolated vertex would have to see the whole other side
    A = [v for v in col if col[v] == 0]; B = [v for v in col if col[v] == 1]
    return len(E) == len(A) * len(B)

def signs():
    for N, levels in ((3, (4, 6, 8)), (4, (4, 6, 8)), (5, (4, 6))):
        G = graphs(N); out(f'  N={N}: {len(G)} graphs, levels {levels}')
        data = {k: pairing_vectors(N, k, np.arange(N)) for k in levels}
        exp = {k: expected(data[k][2], data[k][0]) for k in levels}
        out(f'    expected dimensions {exp}')
        bad = 0
        for E in G:
            kind = 'O_N' if not E else ('O_N^*' if complete_bipartite(E, N) else 'O_N^+')
            target = {'O_N': 'Br', 'O_N^*': 'HL', 'O_N^+': 'TL'}[kind]
            dims = {}
            for k in levels:
                P, words, X = data[k]; Es = set(map(tuple, E))
                cnt = np.zeros(len(words), dtype=np.int64)
                for s in range(k):
                    for t in range(s + 1, k):
                        a, b = words[:, s], words[:, t]
                        edge = np.array([(min(x, y), max(x, y)) in Es for x, y in zip(a, b)])
                        cnt += (edge & (a < b)).astype(np.int64)
                theta = np.where(cnt % 2 == 0, 1, -1).astype(np.int64)
                dims[k] = joint(X, theta, exact=True)
            ok = all(dims[k] == exp[k][target] for k in levels)
            if not ok: bad += 1
            out(f'    eps = {E if E else "empty"}: {kind}, joint dims {dims} {"ok" if ok else "MISMATCH"}')
        out(f'  N={N}: {bad} mismatches')

def theta_twists():
    rng = np.random.default_rng(3)
    for N, levels in ((4, (4, 6, 8)), (6, (4, 6))):
        n = N // 2
        # letters: 2j = +e_j, 2j+1 = -e_j ; conj swaps them
        conj = np.array([2 * j + 1 - (i % 2) for j in range(n) for i in range(2)])
        wvec = np.zeros((N, n));
        for j in range(n): wvec[2 * j, j] = 1; wvec[2 * j + 1, j] = -1
        data = {k: pairing_vectors(N, k, conj) for k in levels}
        exp = {k: expected(data[k][2], data[k][0]) for k in levels}
        out(f'  N={N}: levels {levels}, expected dimensions {exp}')
        for name, th in [('1/2', 0.5), ('1/3', 1 / 3), ('1/4', 0.25), ('1/5', 0.2), ('1/6', 1 / 6), ('2/5', 0.4),
                         ('sqrt2-1', np.sqrt(2) - 1), ('random', rng.random())]:
            B = np.zeros((n, n))                                     # triangular convention: theta = 1/2 is the
            for j in range(n):                                       # complete bipartite sign twist between the planes
                for l in range(j + 1, n): B[j, l] = th
            dims = {}
            for k in levels:
                P, words, X = data[k]; W = wvec[words]                        # (words, k, n)
                ph = np.zeros(len(words))
                for s in range(k):
                    for t in range(s + 1, k): ph += np.einsum('wi,ij,wj->w', W[:, s], B, W[:, t])
                theta = np.exp(2j * np.pi * ph)
                dims[k] = joint(X, theta, exact=False)
            which = [nm for nm in ('TL', 'HL', 'Br') if all(dims[k] == exp[k][nm] for k in levels)]
            out(f'    theta = {name}: joint dims {dims} -> {which[0] if which else "none of Br, HL, TL"}')

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'both'
    out(time.strftime('%d %b %H:%M') + f'  orthotwists.py {what}')
    if what in ('signs', 'both'): signs()
    if what in ('theta', 'both'): theta_twists()
