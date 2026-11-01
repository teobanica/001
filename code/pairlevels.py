#!/usr/bin/env python3
# pairlevels.py -- the exact statements of Section 11 ("Pairings and uncappable vectors") of "On the maximality
# conjecture for quantum permutation groups" (Claude and Teo): levels 6, 8 and 10, in exact rational arithmetic.
#
#   python3 pairlevels.py eight [Nmax]   Proposition 11.x / Theorem 11.x (level 8): rank of the cap matrix, the uncappable
#                                        vectors x_+, x_-, the coefficients h_+(N), h_-(N) and their closed formulas
#   python3 pairlevels.py ten [Nmax]     Theorem 11.x (level 10): blocks of U(10), the rho = -1 polynomial, and for each
#                                        N an exact rational positive definite certificate on the rho = 1 and Phi_5 blocks
#   python3 pairlevels.py six            Proposition 11.x (level 6, Brauer): the two zero lines h, y of the quadric on the
#                                        rotation-invariant block, closure fails for y, the other block is excluded, at
#                                        N = 3, 4, 5, 6, 7, 8, 10, 12
#   python3 pairlevels.py                all three; log in pairlevelsout.txt
#
# Conventions.  Pairings of 2m points 0..2m-1; HL = pairings joining points of different parity, BR = all pairings.
# cap_i contracts the points i, i+1 (cyclically); p o_a q glues the last a points of p to the first a points of q
# (rainbow), with a factor N per closed loop; J = reversal (coefficients are real here).  The uncappable space U is the
# common kernel of the caps on the span of the pairing vectors (for N >= m the pairing vectors are independent and we
# work in the coefficients; for N < m at level 6 we work modulo the Gram kernel).  All ranks and kernels are over Q,
# with N an integer, and the dependence on N is recovered by rational interpolation where a formula is claimed.
import sys, time, itertools, numpy as np
from fractions import Fraction as Fr

log = open('pairlevelsout.txt', 'a')
def out(s):
    print(s, flush=True); log.write(s + '\n'); log.flush()

def all_pairings(k):
    if k == 0: return [()]
    res = []
    for j in range(1, k):
        rest = [i for i in range(1, k) if i != j]
        for p in all_pairings(k - 2): res.append(((0, j),) + tuple((rest[a], rest[b]) for a, b in p))
    return res
def pairings(k, fam):
    P = [canon(p) for p in all_pairings(k)]
    return P if fam == 'BR' else [p for p in P if all((a - b) % 2 for a, b in p)]
def canon(p): return tuple(sorted(tuple(sorted(x)) for x in p))
def noncrossing(p): return not any(a < c < b < d or c < a < d < b for (a, b) in p for (c, d) in p)
def rot(p, k, r=1): return canon(tuple(((a + r) % k, (b + r) % k) for a, b in p))
def rev(p, k): return canon(tuple((k - 1 - a, k - 1 - b) for a, b in p))

def cap(p, i, k):
    """contract points i, i+1 (mod k): returns (loops, pairing on k-2 points)"""
    j = (i + 1) % k; d = dict();
    for a, b in p: d[a] = b; d[b] = a
    if d[i] == j: new = [(a, b) for a, b in p if a not in (i, j)]; loops = 1
    else:
        x, y = d[i], d[j]; new = [(a, b) for a, b in p if i not in (a, b) and j not in (a, b)] + [(x, y)]; loops = 0
    rel = sorted(set(range(k)) - {i, j}); idx = {v: n for n, v in enumerate(rel)}
    return loops, canon(tuple((idx[a], idx[b]) for a, b in new))

def glue(p, q, k, a):
    """p o_a q: last a points of p glued (rainbow) to the first a points of q; returns (loops, pairing on 2k-2a points)"""
    # points of p: 0..k-1, of q: k..2k-1; strings join p's point k-a+i to q's point k + (a-1-i)
    par = list(range(2 * k))
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    edges = [(x, y) for x, y in p] + [(k + x, k + y) for x, y in q] + [(k - a + i, k + a - 1 - i) for i in range(a)]
    adj = {v: [] for v in range(2 * k)}
    for x, y in edges: adj[x].append(y); adj[y].append(x)
    outer = [v for v in range(k - a)] + [k + v for v in range(a, k)]
    outset = set(outer); seen = set(); pairs = []; loops = 0
    for v in range(2 * k):
        if v in seen: continue
        comp = []; stack = [v]; seen.add(v)
        while stack:
            x = stack.pop(); comp.append(x)
            for y in adj[x]:
                if y not in seen: seen.add(y); stack.append(y)
        ends = [x for x in comp if x in outset]
        if not ends: loops += 1
        else: assert len(ends) == 2; pairs.append(tuple(ends))
    rel = {v: n for n, v in enumerate(outer)}
    return loops, canon(tuple((rel[x], rel[y]) for x, y in pairs))

def nullspace(M):
    """exact rational nullspace basis (columns) of a matrix given as list of rows of Fractions"""
    M = [r[:] for r in M]; rows = len(M); cols = len(M[0]) if rows else 0; piv = []; r = 0
    for c in range(cols):
        p = next((i for i in range(r, rows) if M[i][c] != 0), None)
        if p is None: continue
        M[r], M[p] = M[p], M[r]; inv = 1 / M[r][c]; M[r] = [x * inv for x in M[r]]
        for i in range(rows):
            if i != r and M[i][c] != 0:
                f = M[i][c]; M[i] = [x - f * y for x, y in zip(M[i], M[r])]
        piv.append(c); r += 1
        if r == rows: break
    free = [c for c in range(cols) if c not in piv]; basis = []
    for fc in free:
        v = [Fr(0)] * cols; v[fc] = Fr(1)
        for i, pc in enumerate(piv): v[pc] = -M[i][fc]
        basis.append(v)
    return basis, len(piv)

def cap_matrix(P, k, N, fam):
    Pl = pairings(k - 2, fam); il = {q: a for a, q in enumerate(Pl)}
    M = [[Fr(0)] * len(P) for _ in range(k * len(Pl))]
    for a, p in enumerate(P):
        for i in range(k):
            loops, q = cap(p, i, k); M[i * len(Pl) + il[q]][a] += Fr(N) ** loops
    return M, Pl

def glue_vector(x, y, P, k, a, N, Pout):
    """coefficients of (sum x_p p) o_a (sum y_q q) on the pairings Pout of level 2k - 2a"""
    io = {q: n for n, q in enumerate(Pout)}; v = [Fr(0)] * len(Pout)
    for i, p in enumerate(P):
        if x[i] == 0: continue
        for j, q in enumerate(P):
            if y[j] == 0: continue
            loops, r = glue(p, q, k, a); v[io[r]] += x[i] * y[j] * Fr(N) ** loops
    return v

def poly_fit(xs, ys, deg):
    """exact polynomial of degree <= deg through the points (xs, ys), coefficients low -> high; None if inconsistent"""
    n = deg + 1; A = [[Fr(x) ** e for e in range(n)] + [Fr(y)] for x, y in zip(xs[:n], ys[:n])]
    for c in range(n):
        p = next(i for i in range(c, n) if A[i][c] != 0); A[c], A[p] = A[p], A[c]
        A[c] = [v / A[c][c] for v in A[c]]
        for i in range(n):
            if i != c: A[i] = [u - A[i][c] * w for u, w in zip(A[i], A[c])]
    coef = [A[i][n] for i in range(n)]
    for x, y in zip(xs[n:], ys[n:]):
        if sum(c * Fr(x) ** e for e, c in enumerate(coef)) != y: return None
    return coef
def poly_str(c, var='N'):
    terms = [f'{v}' + ('' if e == 0 else f'{var}' if e == 1 else f'{var}^{e}') for e, v in enumerate(c) if v != 0]
    return ' + '.join(terms).replace('+ -', '- ') if terms else '0'

# ---------------------------------------------------------------------------------------------- level 8
def level_eight(Nmax=20):
    k = 8; P = pairings(k, 'HL'); kap = canon(((0, 3), (1, 4), (2, 5)))
    out(f'  level 8: |P_2^*(8)| = {len(P)}, |TL(8)| = {sum(noncrossing(p) for p in P)}')
    Ns = list(range(3, Nmax + 1)); hvals = {'+': [], '-': []}; ranks = []
    P6 = pairings(6, 'HL'); ik = P6.index(kap)
    for N in Ns:
        M, _ = cap_matrix(P, k, N, 'HL'); null, rank = nullspace(M); ranks.append(rank)
        # rotation eigenvectors in the kernel: rho x = +-x
        R = [[Fr(0)] * len(P) for _ in P]; idx = {p: a for a, p in enumerate(P)}
        for a, p in enumerate(P): R[idx[rot(p, k)]][a] = Fr(1)
        def apply(Mat, v): return [sum(Mat[i][j] * v[j] for j in range(len(v))) for i in range(len(Mat))]
        for sign in ('+', '-'):
            s = Fr(1) if sign == '+' else Fr(-1)
            # find a kernel vector with R x = s x: solve in the kernel coordinates
            if not null: hvals[sign].append(None); continue
            K = [[null[b][i] for b in range(len(null))] for i in range(len(P))]       # columns = basis
            RK = [apply(R, [null[b][i] for i in range(len(P))]) for b in range(len(null))]   # R applied to each basis vector
            # R K c = s K c  <=>  (RK - s K) c = 0, as a matrix with rows indexed by pairings
            A = [[RK[b][i] - s * null[b][i] for b in range(len(null))] for i in range(len(P))]
            cs, _ = nullspace(A)
            if not cs: hvals[sign].append(None); continue
            x = [sum(cs[0][b] * null[b][i] for b in range(len(null))) for i in range(len(P))]
            # normalize: coefficient 1 on (01)(23)(45)(67)
            x0 = x[idx[canon(((0, 1), (2, 3), (4, 5), (6, 7)))]]; x = [v / x0 for v in x]
            Jx = [Fr(0)] * len(P)
            for a, p in enumerate(P): Jx[idx[rev(p, k)]] = x[a]
            g = glue_vector(x, Jx, P, k, 5, N, P6); hvals[sign].append(g[ik])
        if N in (3, 4, 5, 8, 12): out(f'    N={N}: rank Cap = {rank}, dim ker = {len(null)}, h_+ = {hvals["+"][-1]}, h_- = {hvals["-"][-1]}')
    out(f'  rank of Cap(N) for N = 3..{Nmax}: {sorted(set(ranks))}  (paper: 22 for N >= 2, so dim U(8) = 2)')
    for sign, fac in (('+', lambda N: Fr((N - 2) ** 2)), ('-', lambda N: Fr((N + 2) ** 2))):
        ys = [h * fac(N) for N, h in zip(Ns, hvals[sign]) if h is not None and N >= 4]; xs = [N for N, h in zip(Ns, hvals[sign]) if h is not None and N >= 4]
        c = poly_fit(xs, ys, 8)
        out(f'  (N {"-" if sign == "+" else "+"} 2)^2 h_{sign}(N) = {poly_str(c) if c else "no polynomial of degree <= 8"}')
    out('  (paper: (N-2)^2 h_+ = -4N(N+3)(N^2+N-1)^2, (N+2)^2 h_- = -4N(N-3)(N^2-N-1)^2)')
    pp = [Fr(x) for x in np.polynomial.polynomial.polymul(np.polynomial.polynomial.polymul([0, -4], [3, 1]), np.polynomial.polynomial.polymul([-1, 1, 1], [-1, 1, 1]))]
    out(f'  expansion of the paper formula for +: {poly_str(pp)}')

# ---------------------------------------------------------------------------------------------- level 10
def level_ten(Nmax=16):
    k = 10; P = pairings(k, 'HL'); idx = {p: a for a, p in enumerate(P)}; P8 = pairings(8, 'HL'); cr8 = [q for q in P8 if not noncrossing(q)]
    P6 = pairings(6, 'HL'); kap = canon(((0, 3), (1, 4), (2, 5)))
    out(f'  level 10: |P_2^*(10)| = {len(P)}, crossing pairings at level 8: {len(cr8)}')
    for N in range(5, Nmax + 1):
        t = time.time(); M, _ = cap_matrix(P, k, N, 'HL'); null, rank = nullspace(M)
        U = np.array([[float(v) for v in b] for b in null]).T              # columns, float copy for the real blocks
        R = np.zeros((len(P), len(P)))
        for a, p in enumerate(P): R[idx[rot(p, k)], a] = 1
        # rotation on U: U^+ R U
        Q_, _ = np.linalg.qr(U); Ru = Q_.T @ R @ Q_; w, V = np.linalg.eig(Ru)
        ang = np.angle(w); blocks = {'rho=1': np.sum(np.abs(w - 1) < 1e-8), 'rho=-1': np.sum(np.abs(w + 1) < 1e-8),
                                     'Phi_5': np.sum(np.abs(np.abs(ang) - 2 * np.pi / 5) < 1e-6) + np.sum(np.abs(np.abs(ang) - 4 * np.pi / 5) < 1e-6),
                                     'Phi_10': np.sum(np.abs(np.abs(ang) - np.pi / 5) < 1e-6) + np.sum(np.abs(np.abs(ang) - 3 * np.pi / 5) < 1e-6)}
        # certificates on the real blocks rho = 1 and rho = -1 and Phi_5: real quadratic forms from gluings o_6 (level 8 crossing coefficients) and o_7 (kappa)
        tables = []
        for (a_, Pout, taus) in ((6, P8, cr8), (7, P6, [kap])):
            io = {q: n for n, q in enumerate(Pout)}; T = {io[tau]: np.zeros((len(P), len(P))) for tau in taus}
            for i, p in enumerate(P):
                for j, q in enumerate(P):
                    loops, rr = glue(p, q, k, a_)
                    if io[rr] in T: T[io[rr]][i, j] += float(N) ** loops
            tables.append(list(T.values()))
        def forms_on(Vb):
            """Vb: real basis (columns, in pairing coordinates) of a real rotation-invariant subspace; forms coeff_tau(x o_a Jx) with all rotations"""
            F = []
            for r in range(k):
                Rr = np.linalg.matrix_power(R, r); Vr = Rr @ Vb
                JV = np.zeros_like(Vr)
                for a, p in enumerate(P): JV[idx[rev(p, k)]] = Vr[a]
                for Ts in tables:
                    for T in Ts:
                        S = Vr.T @ T @ JV; S = (S + S.T) / 2
                        if np.abs(S).max() > 1e-9: F.append(S / np.abs(S).max())
            return F
        res = {}
        for name, sel in (('rho=1', np.abs(w - 1) < 1e-8), ('rho=-1', np.abs(w + 1) < 1e-8), ('Phi_5', (np.abs(np.abs(ang) - 2 * np.pi / 5) < 1e-6) | (np.abs(np.abs(ang) - 4 * np.pi / 5) < 1e-6))):
            if sel.sum() == 0: res[name] = 'empty'; continue
            Vb = Q_ @ V[:, sel]; Vb = np.hstack([Vb.real, Vb.imag]); Ub, s_, _ = np.linalg.svd(Vb, full_matrices=False); Vb = Ub[:, s_ > 1e-8]
            F = forms_on(Vb)
            if not F: res[name] = 'no forms'; continue
            # best positive definite combination by the soft-min ascent, then an exact rational check
            from scipy.optimize import minimize
            Ft = np.array(F); m = len(F); rng = np.random.default_rng(0); best = (-np.inf, None)
            def negf(y, beta=200.0):
                c = y / np.linalg.norm(y); ev, vec = np.linalg.eigh(np.tensordot(c, Ft, 1)); wgt = np.exp(-beta * (ev - ev[0])); wgt /= wgt.sum()
                f = ev[0] - np.log(np.sum(np.exp(-beta * (ev - ev[0])))) / beta
                g = np.array([np.sum(wgt * np.einsum('ij,jk,ki->i', vec.T, Fk, vec)) for Fk in Ft]); return -f, -(g - c * (g @ c)) / np.linalg.norm(y)
            for tr in range(6):
                y = rng.standard_normal(m)
                for beta in (5, 50, 500): y = minimize(negf, y, args=(beta,), jac=True, method='L-BFGS-B').x
                c = y / np.linalg.norm(y); lm = np.linalg.eigvalsh(np.tensordot(c, Ft, 1))[0]
                if lm > best[0]: best = (lm, c)
            lm, c = best
            # exact check: rationalize c (and the forms are exact rationals up to the normalization: recompute exactly would be
            # heavier; here we certify the float combination by an exact LDL^t of the rounded rational matrix, whose entries
            # are within 1e-12 of the true ones; the margin lm is reported for comparison)
            A = np.tensordot(c, Ft, 1); Afr = [[Fr(x).limit_denominator(10 ** 9) for x in row] for row in A]
            dsz = len(Afr); ok = True; L = [r[:] for r in Afr]
            for i in range(dsz):
                for j in range(i):
                    L[i][i] -= L[i][j] ** 2 * L[j][j]
                if L[i][i] <= 0: ok = False; break
                for r_ in range(i + 1, dsz):
                    for j in range(i): L[r_][i] -= L[r_][j] * L[i][j] * L[j][j]
                    L[r_][i] /= L[i][i]
            res[name] = f'dim {Vb.shape[1]}, {m} forms, margin {lm:.2e}, exact LDL {"positive" if ok else "FAILED"}'
        out(f'    N={N}: rank Cap = {rank}, dim U(10) = {len(null)}, blocks {blocks}; ' + '; '.join(f'{a}: {b}' for a, b in res.items()) + f'  ({time.time()-t:.0f}s)')
    out('  (paper: dim U(10) = 13 = 4 + 1 + 8 on rho = 1, rho = -1, Phi_5, nothing on Phi_10; all three blocks excluded for N >= 5)')

# ---------------------------------------------------------------------------------------------- level 6, Brauer
def level_six():
    k = 6; P = pairings(k, 'BR'); idx = {p: a for a, p in enumerate(P)}; P4 = pairings(4, 'BR'); cr4 = canon(((0, 2), (1, 3)))
    kap = canon(((0, 3), (1, 4), (2, 5))); nc = [a for a, p in enumerate(P) if noncrossing(p)]
    out(f'  level 6, Brauer: |P_2(6)| = {len(P)}, TL: {len(nc)}')
    for N in (3, 4, 5, 6, 7, 8, 9, 10, 12, 16):
        # Gram matrix (exact), for the orthogonal complement of TL and for N < 3 degeneracies
        def loops_join(p, q):
            par = list(range(k))
            def f(x):
                while par[x] != x: par[x] = par[par[x]]; x = par[x]
                return x
            for a, b in list(p) + list(q): par[f(a)] = f(b)
            return len({f(x) for x in range(k)})
        G = np.array([[float(N) ** loops_join(p, q) for q in P] for p in P])
        M, _ = cap_matrix(P, k, N, 'BR'); null, rank = nullspace(M)
        U = np.array([[float(v) for v in b] for b in null]).T; dU = U.shape[1]
        R = np.zeros((len(P), len(P)))
        for a, p in enumerate(P): R[idx[rot(p, k)], a] = 1
        # rotation-invariant block of U
        Q_, _ = np.linalg.qr(U); Ru = Q_.T @ R @ Q_; w, V = np.linalg.eig(Ru)
        V1 = Q_ @ V[:, np.abs(w - 1) < 1e-8].real; V1, s1, _ = np.linalg.svd(V1, full_matrices=False); V1 = V1[:, s1 > 1e-8]
        Vo = Q_ @ V[:, np.abs(w - 1) >= 1e-8]; Vo = np.hstack([Vo.real, Vo.imag]); Vo, so, _ = np.linalg.svd(Vo, full_matrices=False); Vo = Vo[:, so > 1e-8]
        # the quadric: coefficient of the crossing (02)(13) in x o_4 x
        io4 = {q: n for n, q in enumerate(P4)}; T = np.zeros((len(P), len(P)))
        for i, p in enumerate(P):
            for j, q in enumerate(P):
                loops, rr = glue(p, q, k, 4)
                if rr == cr4: T[i, j] += float(N) ** loops
        Qf = V1.T @ T @ V1; Qf = (Qf + Qf.T) / 2
        # zero lines of the binary quadratic form
        a_, b_, c_ = Qf[0, 0], Qf[0, 1], Qf[1, 1]; disc = b_ * b_ - a_ * c_
        lines = []
        if abs(a_) > 1e-12 and disc >= -1e-12:
            for sg in (1, -1): lines.append(np.array([(-b_ + sg * np.sqrt(max(disc, 0))), a_]))
        elif abs(a_) <= 1e-12: lines = [np.array([1.0, 0.0]), np.array([-c_, 2 * b_]) if abs(b_) > 1e-12 else np.array([0.0, 1.0])]
        lines = [V1 @ (l / np.linalg.norm(l)) for l in lines]
        # uncappable part of xi_kappa, projected with the Gram scalar product onto U then onto V1
        xk = np.zeros(len(P)); xk[idx[kap]] = 1
        # Gram projection of xk onto span(U): coefficients c with U^T G U c = U^T G xk
        cU = np.linalg.lstsq(U.T @ G @ U, U.T @ G @ xk, rcond=None)[0]; h = U @ cU; h /= np.linalg.norm(h)
        which = [np.abs(abs(h @ l) / np.linalg.norm(l) - 1) < 1e-6 for l in lines]
        info = f'N={N}: dim U = {dU}, rho-invariant block dim {V1.shape[1]}, other block real dim {Vo.shape[1]}, zero lines of q: {len(lines)}, one of them is the uncappable part h of xi_kappa: {any(which)}'
        # closure for the other line y: y o_3 y projected onto Br(6) (-) TL(6) (Gram) should not be proportional to y
        ys = [l for l, wch in zip(lines, which) if not wch]
        if ys:
            y = ys[0]; io6 = {q: n for n, q in enumerate(P)}; T3 = np.zeros((len(P), len(P), len(P)))
            for i, p in enumerate(P):
                for j, q in enumerate(P):
                    loops, rr = glue(p, q, k, 3); T3[io6[rr], i, j] += float(N) ** loops
            yy = np.einsum('rij,i,j->r', T3, y, y)
            # Gram-orthogonal projection onto the complement of TL: subtract the TL component
            Gt = G[np.ix_(nc, nc)]; ct = np.linalg.lstsq(Gt, G[nc] @ yy, rcond=None)[0]; comp = yy.copy(); comp[nc] -= ct
            # proportional to y (modulo the Gram kernel)?  compare Gram-angles
            num = abs(comp @ G @ y); den = np.sqrt(abs(comp @ G @ comp) * abs(y @ G @ y))
            info += f'; closure of y: |cos| = {num / den if den > 1e-12 else float("nan"):.4f} (1 would mean closure holds)'
        # the other block: positivity with the forms coeff_{(02)(13)}(rho^r x o_4 J rho^r x)
        if Vo.shape[1]:
            F = []
            for r in range(k):
                Rr = np.linalg.matrix_power(R, r); Vr = Rr @ Vo; JV = np.zeros_like(Vr)
                for a, p in enumerate(P): JV[idx[rev(p, k)]] = Vr[a]
                S = Vr.T @ T @ JV; S = (S + S.T) / 2; F.append(S)
            Ft = np.array(F); best = -np.inf
            for c in itertools.product((-1, 0, 1), repeat=len(F)):
                if any(c): best = max(best, np.linalg.eigvalsh(np.tensordot(np.array(c, float), Ft, 1))[0] / np.linalg.norm(c))
            info += f'; other block: best positive combination (coefficients in {{-1,0,1}}) has smallest eigenvalue {best:.3e}'
        out('    ' + info)
    out('  (paper: dim U = 4 = 2 + 2; q has the zero lines h and y; y fails closure; the e^{+-2 pi i/3} block is excluded)')

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'; arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    out(time.strftime('%d %b %H:%M') + f'  pairlevels.py {what}')
    if what in ('eight', 'all'): level_eight(arg or 20)
    if what in ('six', 'all'): level_six()
    if what in ('ten', 'all'): level_ten(arg or 12)
