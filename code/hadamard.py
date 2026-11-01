#!/usr/bin/env python3
# hadamard.py -- Hadamard models as seeds, for "On the maximality conjecture for quantum permutation groups"
# (Claude and Teo), Section 5 (Propositions 5.9 and 5.13, with the closed-walk formula of Proposition 5.12).
#
#   python3 hadamard.py joint          Proposition 5.9: dim Fix(G_H)(2),(3) and the joint dimensions for k <= 5
#   python3 hadamard.py dita           the Dita scan at k = 4: all pairs of 12th roots of unity, both families
#   python3 hadamard.py wild [kmax]    Proposition 5.13: the numbers lambda_k(H) for N = 6, 7, up to the levels of the table
#                                      (about two hours on a laptop; kmax caps the level)
#   python3 hadamard.py paley [kmax]   Proposition 5.13: the Paley matrices (heavy: several hours, mostly N = 24, 28 at k = 4)
#
# Model: u_ij = Proj(xi_ij), xi_ij = H_i / H_j, and T_k = (id (x) tr)(u^(x)k).  Closed-walk formula (Proposition 5.12):
#   T_k = N^{-(k+1)} sum_x A_x A_x^*,  A_x = a_{x_k x_1} (x) a_{x_1 x_2} (x) ... ,  a_{yx}(i) = H_iy conj(H_ix).
# * joint dimension at level k: the multiplicity of the eigenvalue 1 of the compression of T_k to span{xi_pi}, i.e. the
#   number of eigenvalues 1 of G^{-1/2} M G^{-1/2}, with M_{pi,sigma} = <T_k xi_sigma, xi_pi> and G the Gram matrix;
#   here <A_x, xi_pi> is a product over the blocks b of pi of sum_i prod_{r in b} a_{x_{r-1} x_r}(i).
# * lambda_k(H): the largest eigenvalue of T_k on the orthogonal complement W_k of the noncrossing vectors, by the
#   Lanczos method (scipy eigsh), with T_k applied as the cyclic chain T_{ij} = N^{-(k+1)} prod_s Q(i_s, j_s, i_{s+1}, j_{s+1}),
#   Q(a,b,c,d) = sum_x conj(H_ax) H_bx H_cx conj(H_dx), at cost about k N^{k+3}, and memory N^{k+1}.
# All computations are in floating point ("computational" / "numerical" in the paper).  Everything printed is also
# appended to hadamardout.txt, next to the script.
import sys, itertools, time
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

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

def join_count(p, q, k):
    par = list(range(k))
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for part in (p, q):
        for b in part:
            for l in b[1:]: par[f(l)] = f(b[0])
    return len({f(l) for l in range(k)})

# ----------------------------------------------------------------------------------------- the matrices
def fourier(n): w = np.exp(2j * np.pi / n); return np.array([[w ** (i * j) for j in range(n)] for i in range(n)])

def tao():
    w = np.exp(2j * np.pi / 3)
    return np.array([[1, 1, 1, 1, 1, 1], [1, 1, w, w, w**2, w**2], [1, w, 1, w**2, w**2, w], [1, w, w**2, 1, w, w**2],
                     [1, w**2, w**2, w, 1, w], [1, w**2, w, w**2, w, 1]])

def haagerup(q):
    I = 1j; qb = np.conj(q)
    return np.array([[1, 1, 1, 1, 1, 1], [1, -1, I, I, -I, -I], [1, I, -1, -I, q, -q], [1, I, -I, -1, -q, q],
                     [1, -I, qb, -qb, I, -1], [1, -I, -qb, qb, -1, I]])

def dita(h, k, l):
    """(h (x)_l k)_{ia,jb} = h_ij l_aj k_ab."""
    m, n = h.shape[0], k.shape[0]; H = np.zeros((m * n, m * n), complex)
    for i, a, j, b in itertools.product(range(m), range(n), range(m), range(n)): H[i * n + a, j * n + b] = h[i, j] * l[a, j] * k[a, b]
    return H

def F23(r, s): return dita(fourier(2), fourier(3), np.array([[1, 1], [1, r], [1, s]]))
def F32(r, s): return dita(fourier(3), fourier(2), np.array([[1, 1, 1], [1, r, s]]))

def petrescu(q):
    w2 = np.exp(2j * np.pi / 3); qb = np.conj(q)
    return np.array([[-q, q, w2, 1, w2, 1, w2], [q, -q, w2, 1, 1, w2, w2], [w2, w2, -w2, 1, w2, w2, 1],
                     [1, 1, 1, -1, w2, w2, w2], [w2, 1, w2, w2, -qb * w2, qb * w2, 1], [1, w2, w2, w2, qb * w2, -qb * w2, 1],
                     [w2, w2, 1, w2, 1, 1, -1]])

def lampio(q):
    w = np.exp(2j * np.pi / 6)
    return np.array([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, w**2, w**3, w**3, w**4], [1, w, w**4, w**4, q, q * w**3, w**2],
                     [1, w**2, w**3, 1, q * w**4, q * w, w**4], [1, w**3, w**5, w**3, q * w**3, q, w],
                     [1, w**4, w**2, q * w**2, q, q * w**5, q * w**3], [1, w**4, w**2, q * w**5, q * w**2, q * w**3, q]])

def paley(q, kind):
    """Paley I (q = 3 mod 4) or II (q = 1 mod 4), for q prime (enough here) or q = 9, 27 via GF(p^m) arithmetic."""
    F = gf(q); els = F['els']
    chi = lambda x: 0 if x == F['zero'] else (1 if x in F['squares'] else -1)
    Q = np.array([[chi(F['sub'](b, a)) for b in els] for a in els])
    one = np.ones((q, 1))
    if kind == 1:
        S = np.block([[np.zeros((1, 1)), one.T], [-one, Q]]); return np.eye(q + 1) + S
    C = np.block([[np.zeros((1, 1)), one.T], [one, Q]])
    return np.kron(C, np.array([[1, 1], [1, -1]])) + np.kron(np.eye(q + 1), np.array([[1, -1], [-1, -1]]))

def gf(q):
    p = next(d for d in range(2, q + 1) if q % d == 0); m = round(np.log(q) / np.log(p))
    if m == 1:
        els = list(range(q)); mul = lambda a, b: a * b % q; sub = lambda a, b: (a - b) % q; zero = 0
    else:   # GF(p^m) as polynomials modulo a fixed irreducible: x^2 + 1 for 9, x^3 - x - 1 ... found by search
        polys = list(itertools.product(range(p), repeat=m))
        def pmul(a, b, mod):
            r = [0] * (2 * m - 1)
            for i in range(m):
                for j in range(m): r[i + j] = (r[i + j] + a[i] * b[j]) % p
            for d in range(2 * m - 2, m - 1, -1):          # reduce by x^m = -(mod[0] + mod[1] x + ...)
                c = r[d]; r[d] = 0
                for t in range(m): r[d - m + t] = (r[d - m + t] - c * mod[t]) % p
            return tuple(r[:m])
        for mod in itertools.product(range(p), repeat=m):
            if all(pmul(a, b, mod) != tuple([0] * m) for a in polys[1:] for b in polys[1:]): break
        els = polys; mul = lambda a, b: pmul(a, b, mod); sub = lambda a, b: tuple((x - y) % p for x, y in zip(a, b))
        zero = tuple([0] * m)
    squares = {mul(a, a) for a in els if a != zero}
    return {'els': els, 'zero': zero, 'sub': sub, 'squares': squares}

def check_hadamard(H):
    N = H.shape[0]; assert np.allclose(np.abs(H), 1) and np.allclose(H @ H.conj().T, N * np.eye(N)), 'not Hadamard'

# ----------------------------------------------------------------------------------------- joint dimensions
def avec(H):
    """a[y, x, i] = H_iy conj(H_ix)."""
    return np.einsum('iy,ix->yxi', H, H.conj())

def fix_full(H, k, tol=1e-8):
    """dim Fix(G_H)(k), the multiplicity of 1 in the spectrum of T_k on (C^N)^(x)k (small k)."""
    N = H.shape[0]; a = avec(H); cols = []
    for x in itertools.product(range(N), repeat=k):
        v = np.ones(1)
        for r in range(k): v = np.kron(v, a[x[r - 1], x[r]])
        cols.append(v)
    A = np.array(cols).T; T = A @ A.conj().T / N ** (k + 1)
    ev = np.linalg.eigvalsh(T); return int((np.abs(ev - 1) < tol).sum())

def joint(H, k, tol=1e-7):
    N = H.shape[0]; a = avec(H); parts = set_partitions(k); B = len(parts)
    G = np.array([[float(N) ** join_count(p, q, k) for q in parts] for p in parts])
    walks = np.array(list(itertools.product(range(N), repeat=k)))
    # s[S][x] = sum_i prod_{r in S} a_{x_{r-1} x_r}(i), for the subsets S of legs
    s = {}
    for r in range(1, k + 1):
        for S in itertools.combinations(range(k), r):
            prod = np.ones((len(walks), N), complex)
            for l in S: prod = prod * a[walks[:, l - 1], walks[:, l]]
            s[S] = prod.sum(axis=1)
    Phi = np.ones((len(walks), B), complex)
    for j, p in enumerate(parts):
        for b in p: Phi[:, j] *= s[tuple(b)]
    M = (Phi.T @ Phi.conj()) / N ** (k + 1)
    w, V = np.linalg.eigh(G); Gm = V @ np.diag(1 / np.sqrt(w)) @ V.conj().T
    ev = np.linalg.eigvalsh(Gm @ M @ Gm)
    return int((np.abs(ev - 1) < tol).sum())

# ----------------------------------------------------------------------------------------- lambda_k
class TOp:
    def __init__(self, H, k):
        self.H = H; self.N = N = H.shape[0]; self.k = k
        Hb = H.conj(); self.Q = np.einsum('ax,bx,cx,dx->abcd', Hb, H, H, Hb) / N ** ((k + 1) / k)
        self.real = np.allclose(H.imag, 0)
        if self.real: self.Q = self.Q.real
    def apply(self, v):
        """(T v)(i) = sum_j prod_s Q(i_s, j_s, i_{s+1}, j_{s+1}) v(j), cyclically; loop over i_1 to save memory."""
        N, k, Q = self.N, self.k, self.Q
        v = v.reshape((N,) * k); out = np.empty((N,) * k, dtype=np.result_type(v, Q))
        for i1 in range(N):
            # Y[j1, i2, j2, j3, ..., jk] = Q(i1, j1, i2, j2) v(j1, j2, ..., jk)
            Y = np.einsum('abc,ac...->abc...', Q[i1], v)
            for s in range(2, k):
                # Y indices (j1, i2..i_s, j_s, j_{s+1}, rest): multiply by Q(i_s, j_s, i_{s+1}, j_{s+1}), sum j_s
                Y = np.moveaxis(Y, [s - 1, s, s + 1], [-3, -2, -1])
                Y = np.einsum('...abd,abcd->...acd', Y, Q)        # (rest..., i_s, i_{s+1}, j_{s+1})
                Y = np.moveaxis(Y, [-3, -2, -1], [s - 1, s, s + 1])
            # Y indices (j1, i2, ..., ik, jk): multiply by Q(ik, jk, i1, j1), sum jk and j1
            out[i1] = np.einsum('z...ab,abz->...a', Y, Q[:, :, i1, :]) if k > 1 else None
        return out.reshape(-1)

def nc_projector(N, k):
    parts = [p for p in set_partitions(k) if noncrossing(p)]
    idx = []
    for p in parts:
        lab = [0] * k
        for j, b in enumerate(p):
            for l in b: lab[l] = j
        grids = np.array(list(itertools.product(range(N), repeat=len(p))))
        flat = np.zeros(len(grids), dtype=np.int64)
        for l in range(k): flat = flat * N + grids[:, lab[l]]
        idx.append(flat)
    G = np.array([[float(N) ** join_count(p, q, k) for q in parts] for p in parts])
    Ginv = np.linalg.inv(G)
    def proj(v):
        c = np.array([v[f].sum() for f in idx]); c = Ginv @ c; w = v.copy()
        for f, cc in zip(idx, c): w[f] -= cc
        return w
    return proj, idx

def lam(H, k, tol=1e-6):
    N = H.shape[0]; T = TOp(H, k); proj, idx = nc_projector(N, k)
    dt = float if T.real else complex
    xi = np.zeros(N ** k); xi[idx[0]] = 1; err = np.abs(T.apply(xi) - xi).max()   # a noncrossing vector is fixed
    op = LinearOperator((N ** k, N ** k), matvec=lambda v: proj(T.apply(proj(v.astype(dt)))), dtype=dt)
    val = eigsh(op, k=1, which='LA', tol=tol, return_eigenvectors=False)[0]
    return float(np.real(val)), err

# ----------------------------------------------------------------------------------------- runs
GENERIC6 = np.exp(2j * np.pi * 0.1234); GENERIC7R = np.exp(2j * np.pi * 0.3711)
w3, w6 = np.exp(2j * np.pi / 3), np.exp(1j * np.pi / 3)

def models_joint():
    out = [('T6', tao())]
    for nm, q in [('1', 1), ('i', 1j), ('e^{2pi i/3}', w3), ('e^{i pi/3}', w6), ('generic', GENERIC6)]: out.append((f'H6^q, q={nm}', haagerup(q)))
    r0, s0 = np.exp(2j * np.pi * 0.1234), np.exp(2j * np.pi * 0.4321)
    out += [('F23^rs generic', F23(r0, s0)), ('F32^rs generic', F32(r0, s0))]
    for nm, q in [('1', 1), ('i', 1j), ('e^{2pi i/3}', w3), ('generic', GENERIC6)]: out.append((f'P7^q, q={nm}', petrescu(q)))
    for nm, q in [('1', 1), ('-1', -1), ('i', 1j), ('e^{2pi i/3}', w3), ('generic', GENERIC7R)]: out.append((f'R7^q, q={nm}', lampio(q)))
    return out

def out(s, **kw):
    print(s, flush=True); open('hadamardout.txt', 'a').write(s + '\n')

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'joint'
    out(time.strftime('%d %b %H:%M') + f'  hadamard.py {what}')
    if what == 'joint':
        for name, H in models_joint():
            check_hadamard(H); f2, f3 = fix_full(H, 2), fix_full(H, 3)
            js = [joint(H, k) for k in (4, 5)]
            out(f'  {name:22s} dim Fix(G_H)(2),(3) = {f2},{f3}   joint k=4,5: {js}', flush=True)
    elif what == 'dita':
        roots = [np.exp(2j * np.pi * t / 12) for t in range(12)]
        for fam, F in (('F23', F23), ('F32', F32)):
            res = {}
            for r, s in itertools.product(roots, roots):
                H = F(r, s); d = joint(H, 4); res.setdefault(d, []).append((round(np.angle(r) / (2 * np.pi) * 12) % 12, round(np.angle(s) / (2 * np.pi) * 12) % 12))
            out(f'  {fam}: ' + ', '.join(f'{len(v)} pairs give {d}' for d, v in sorted(res.items())), flush=True)
    elif what in ('wild', 'paley'):
        kcap = int(sys.argv[2]) if len(sys.argv) > 2 else 99     # optional cap on the levels
        if what == 'wild':
            mats = [('T6', tao(), 7), ('H6^q, q=1', haagerup(1), 7), ('H6^q, q=i', haagerup(1j), 7),
                    ('H6^q, q=e^{2pi i/3}', haagerup(w3), 6), ('H6^q, q=e^{i pi/3}', haagerup(w6), 6), ('H6^q generic', haagerup(GENERIC6), 7),
                    ('P7^q, q=1', petrescu(1), 6), ('P7^q, q=e^{2pi i/3}', petrescu(w3), 6), ('P7^q, q=i', petrescu(1j), 6),
                    ('P7^q generic', petrescu(GENERIC6), 7), ('R7^q, q=1', lampio(1), 6), ('R7^q, q=-1', lampio(-1), 6),
                    ('R7^q, q=e^{2pi i/3}', lampio(w3), 6), ('R7^q, q=i', lampio(1j), 6), ('R7^q generic', lampio(GENERIC7R), 7)]
        else:
            mats = [('N=12, Paley I (q=11)', paley(11, 1), 6), ('N=12, Paley II (q=5)', paley(5, 2), 6), ('N=20, Paley I (q=19)', paley(19, 1), 4),
                    ('N=20, Paley II (q=9)', paley(9, 2), 4), ('N=24, Paley I (q=23)', paley(23, 1), 4), ('N=28, Paley I (q=27)', paley(27, 1), 4),
                    ('N=28, Paley II (q=13)', paley(13, 2), 4)]
        for name, H, kmax in mats:
            check_hadamard(H); vals = []
            for k in range(2, min(kmax, kcap) + 1):
                t = time.time(); v, err = lam(H, k); vals.append(v)
                out(f'  {name}: k={k} lambda = {v:.6f}   (noncrossing vector fixed up to {err:.1e}, {time.time()-t:.0f} s)', flush=True)
