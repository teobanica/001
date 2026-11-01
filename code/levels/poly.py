# Univariate polynomials over Q in the variable N: lists of Fractions, low -> high.
from fractions import Fraction as Fr
def trim(a):
    a = list(a)
    while a and a[-1] == 0: a.pop()
    return a
def add(a, b):
    n = max(len(a), len(b))
    return trim([(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)])
def neg(a): return [-x for x in a]
def sub(a, b): return add(a, neg(b))
def mul(a, b):
    if not a or not b: return []
    r = [Fr(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                if y: r[i + j] += x * y
    return trim(r)
def scal(s, a): return trim([s * x for x in a])
def ev(a, x):
    r = Fr(0)
    for c in reversed(a): r = r * x + c
    return r
def deg(a): return len(trim(a)) - 1
def divmod_(a, b):
    a = trim(a); b = trim(b); q = [Fr(0)] * max(len(a) - len(b) + 1, 0)
    while len(a) >= len(b) and a:
        f = a[-1] / b[-1]; k = len(a) - len(b); q[k] = f
        a = trim([a[i] - (f * b[i - k] if 0 <= i - k < len(b) else 0) for i in range(len(a))])
    return trim(q), a
def gcd(a, b):
    a, b = trim(a), trim(b)
    while b: a, b = b, divmod_(a, b)[1]
    return scal(1 / a[-1], a) if a else a
def interp(xs, ys):
    """Lagrange/Newton interpolation, exact."""
    n = len(xs); coef = [Fr(y) for y in ys]
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (xs[i] - xs[i - j])
    p = []
    for i in range(n - 1, -1, -1):
        p = add(mul(p, [Fr(-xs[i]), Fr(1)]), [coef[i]])
    return p
def X(): return [Fr(0), Fr(1)]
def const(c): return trim([Fr(c)])
def fmt(a, v='N'):
    a = trim(a)
    if not a: return '0'
    terms = []
    for i in range(len(a) - 1, -1, -1):
        c = a[i]
        if c == 0: continue
        m = '' if i == 0 else (v if i == 1 else f'{v}^{i}')
        cs = str(c) if (m == '' or abs(c) != 1) else ('-' if c < 0 else '')
        terms.append(f'{cs}{"*" if m and cs not in ("", "-") else ""}{m}')
    return ' + '.join(terms).replace('+ -', '- ')
def int_roots(a):
    """Integer roots of a polynomial with rational coefficients (exact)."""
    from math import lcm, gcd as igcd
    a = trim(a)
    L = lcm(*[x.denominator for x in a]); z = [int(x * L) for x in a]
    roots = []
    while z and z[0] == 0: roots.append(0); z = z[1:]
    if len(z) <= 1: return roots
    c0 = abs(z[0])
    # divisors of constant term can be huge; use bound from Cauchy and test candidates dividing c0
    B = 1 + max(abs(Fr(x, z[-1])) for x in z[:-1])
    cand = set()
    d = 1
    while d <= min(c0, int(B) + 1):
        if c0 % d == 0: cand.update({d, -d})
        d += 1
    for r in sorted(cand):
        v = 0
        for c in reversed(z): v = v * r + c
        if v == 0: roots.append(r)
    return sorted(set(roots))
