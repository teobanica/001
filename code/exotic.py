#!/usr/bin/env python3
# exotic.py -- Section 6 ("Exotic quantum permutation groups") of "On the maximality conjecture for quantum permutation
# groups" (Claude and Teo): the finite case, Theorem 6.x (index >= 31, >= 32 for N >= 7), the level-8 bound q >= 123,
# the per-level thresholds, Proposition 6.x (q >= 1 + d_4, e.g. 1394 at N = 8), and the multiplicities f_j of the
# trivial representation of S_N in the irreducible representations r_j of S_N^+ (1, 0, 0, 0, 1, 1, 6, 20, 86, 393, ...).
#
#   python3 exotic.py            everything (a second); exact rational arithmetic throughout
#
# Setting.  G finite exotic, D = dim C(G) = q N!, mu_G = (1/q) mu_{S_N} + (1 - 1/q) nu with nu a probability measure
# on [0, N).  If G agrees with S_N^+ through level k0, the moments of nu are nu_k = C_k + t (C_k - s_k) for k <= k0,
# with t = 1/(q-1), C_k the Catalan numbers and s_k = dim Fix_{S_N}(u^(x)k) (Bell numbers for k <= N).  Positivity of
# nu on [0, oo) forces the shifted Hankel matrices (nu_{i+j+1}) to be positive semidefinite, and positivity on the line
# forces (nu_{i+j}) >= 0; a negative determinant excludes the value of t.
import sys
from fractions import Fraction as Fr
from math import comb

def bell(n):
    B = [1]
    for k in range(n): B.append(sum(comb(k, i) * B[i] for i in range(k + 1)))
    return B
def catalan(n): return [comb(2 * k, k) // (k + 1) for k in range(n + 1)]
C = catalan(12); B = bell(12)

def det(M):
    M = [r[:] for r in M]; n = len(M); d = Fr(1)
    for c in range(n):
        p = next((i for i in range(c, n) if M[i][c] != 0), None)
        if p is None: return Fr(0)
        if p != c: M[c], M[p] = M[p], M[c]; d = -d
        d *= M[c][c]
        for i in range(c + 1, n):
            f = M[i][c] / M[c][c]
            for j in range(c, n): M[i][j] -= f * M[c][j]
    return d

def hankel_poly(s, shift, n):
    """det (nu_{i+j+shift})_{0<=i,j<n} as a polynomial in t (coefficients low -> high), by interpolation at n+1 points"""
    deg = n; xs = list(range(deg + 1)); ys = []
    for x in xs:
        nu = [C[k] + Fr(x) * (C[k] - s[k]) for k in range(len(C))]
        ys.append(det([[nu[i + j + shift] for j in range(n)] for i in range(n)]))
    A = [[Fr(x) ** e for e in range(deg + 1)] + [y] for x, y in zip(xs, ys)]; m = deg + 1
    for c in range(m):
        p = next(i for i in range(c, m) if A[i][c] != 0); A[c], A[p] = A[p], A[c]
        A[c] = [v / A[c][c] for v in A[c]]
        for i in range(m):
            if i != c: A[i] = [a - A[i][c] * b for a, b in zip(A[i], A[c])]
    return [A[i][m] for i in range(m)]

def ev(p, x): return sum(c * x ** i for i, c in enumerate(p))
def smallest_positive_root(p, lo=Fr(0), hi=Fr(1, 2), steps=200):
    """first sign change of p on (0, hi], by bisection (p(0) = 1 > 0 in all cases below)"""
    x0, x1 = lo, hi
    n = 1
    while ev(p, x1) > 0 and n < 60: x1 *= 2; n += 1
    assert ev(p, x1) <= 0
    # first root: scan
    grid = [x1 * Fr(i, 4000) for i in range(1, 4001)]
    xprev = lo
    for x in grid:
        if ev(p, x) <= 0:
            a, b = xprev, x
            for _ in range(steps):
                m = (a + b) / 2
                if ev(p, m) > 0: a = m
                else: b = m
            return b
        xprev = x
def poly_str(p):
    terms = []
    for i, c in enumerate(p):
        if c == 0: continue
        cs = f'{c}' if c.denominator == 1 else f'({c})'
        terms.append(cs if i == 0 else (f'{cs}t' if i == 1 else f'{cs}t^{i}'))
    return ' + '.join(terms).replace('+ -', '- ')

print('== Theorem 6.x: shifted Hankel (nu_{i+j+1})_{0<=i,j<4} from levels 1..7')
p7 = hankel_poly(B, 1, 4); print('  N >= 7 (s_k = B_k):          det =', poly_str(p7))
s6 = B[:]; s6[7] = B[7] - 1
p6 = hankel_poly(s6, 1, 4); print('  N = 6   (s_7 = B_7 - 1):     det =', poly_str(p6))
for name, p in (('N>=7', p7), ('N=6', p6)):
    x = 1
    while ev(p, Fr(1, x)) < 0: x += 1
    # x = q - 1 = 1/t: the determinant is negative for 1 <= q-1 < x
    neg = [y for y in range(1, x) if ev(p, Fr(1, y)) < 0]
    print(f'  {name}: det < 0 at t = 1/x for x = 1..{max(neg)}, det > 0 at x = {x}  =>  q - 1 >= {x},  q >= {x + 1}')

print('\n== Thresholds for t from the Hankel matrices available at each settled level k0 (N >= 8)')
for k0, (shift, n) in ((4, (0, 3)), (5, (1, 3)), (6, (0, 4)), (7, (1, 4)), (8, (0, 5))):
    p = hankel_poly(B, shift, n); r = smallest_positive_root(p)
    print(f'  level {k0}: ({"nu_{i+j}" if shift == 0 else "nu_{i+j+1}"}), {n}x{n}: first root t = {float(r):.4f}  =>  q >= {int(1 + 1 / r) + 1}')

print('\n== Proposition 6.x: dimensions d_k of the irreducible representations of S_N^+, and q >= 1 + d_4')
for N in (6, 7, 8, 9, 10, 12, 37):
    d = [1, N - 1]
    for k in range(1, 5): d.append((N - 2) * d[k] - d[k - 1])
    print(f'  N={N}: d_0..d_5 = {d}, bound q >= {1 + d[4]}')

print('\n== Multiplicities f_j of the trivial representation of S_N in r_j (N large, from s_k = B_k = sum_j m_kj f_j)')
# m_kj = multiplicity of r_j in u^(x)k, from u = r_0 + r_1 and r_1 r_j = r_{j-1} + r_j + r_{j+1} (SO_3 fusion rules)
K = 10; m = [[0] * (K + 2) for _ in range(K + 1)]; m[0][0] = 1
for k in range(K):
    for j in range(K + 1):
        if m[k][j]:
            m[k + 1][j] += m[k][j]                 # r_0 (x) r_j = r_j
            if j == 0: m[k + 1][1] += m[k][j]      # r_1 (x) r_0 = r_1
            else:                                  # r_1 (x) r_j = r_{j-1} + r_j + r_{j+1}
                m[k + 1][j - 1] += m[k][j]; m[k + 1][j] += m[k][j]; m[k + 1][j + 1] += m[k][j]
assert [m[k][0] for k in range(K + 1)] == C[:K + 1]
f = []
for j in range(K):
    f.append(B[j] - sum(m[j][i] * f[i] for i in range(j)))     # m_jj = 1
print('  f_0..f_9 =', f, '  (paper: 1, 0, 0, 0, 1, 1, 6, 20, 86, 393)')
print('  check: sum_j m_kj f_j = B_k for k <= 9:', all(sum(m[k][j] * f[j] for j in range(k + 1)) == B[k] for k in range(K)))
