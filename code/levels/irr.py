# Exact irreducibility over Q of the char polys, via distinct-degree factorisation mod p.
import pickle, sys
from fractions import Fraction as Fr
from math import lcm
def polymod(a, m, p):          # a mod m over F_p, lists high->low
    a = [x % p for x in a]
    while len(a) >= len(m):
        if a[0]:
            f = a[0] * pow(m[0], -1, p) % p
            for i in range(len(m)): a[i] = (a[i] - f * m[i]) % p
        a.pop(0)
    while a and a[0] == 0: a.pop(0)
    return a
def mulmod(a, b, m, p):
    r = [0] * (len(a) + len(b) - 1) if a and b else []
    for i, x in enumerate(a):
        for j, y in enumerate(b): r[i + j] = (r[i + j] + x * y) % p
    return polymod(r, m, p)
def powmod(base, e, m, p):
    r = [1]; b = polymod(base, m, p)
    while e:
        if e & 1: r = mulmod(r, b, m, p)
        b = mulmod(b, b, m, p); e >>= 1
    return r
def sub(a, b, p):
    n = max(len(a), len(b)); a = [0]*(n-len(a)) + a; b = [0]*(n-len(b)) + b
    r = [(x - y) % p for x, y in zip(a, b)]
    while r and r[0] == 0: r.pop(0)
    return r
def gcd(a, b, p):
    while b: a, b = b, polymod(a, b, p)
    if a: inv = pow(a[0], -1, p); a = [x * inv % p for x in a]
    return a
def ddf(f, p):                 # degrees of irreducible factors of squarefree f mod p
    degs = []; g = f; h = [1, 0]; d = 0
    while len(g) - 1 >= 2 * (d + 1):
        d += 1; h = powmod(h, p, g, p) if len(g) > 1 else h
        q = gcd(g, sub(h, [1, 0], p), p)
        k = len(q) - 1
        if k > 0:
            degs += [d] * (k // d)
            # divide g by q
            num = g[:]; quo = []
            while len(num) >= len(q):
                c = num[0] * pow(q[0], -1, p) % p; quo.append(c)
                for i in range(len(q)): num[i] = (num[i] - c * q[i]) % p
                num.pop(0)
            g = quo; h = polymod(h, g, p) if len(g) > 1 else [0]
    if len(g) > 1: degs.append(len(g) - 1)
    return sorted(degs)
def primes():
    p = 101
    while True:
        p += 2
        if all(p % q for q in range(3, int(p**.5) + 1, 2)): yield p
for N in sys.argv[1:]:
    cp = [Fr(x) for x in pickle.load(open(f'elim_N{N}.pkl', 'rb'))['charpoly']]
    L = lcm(*[x.denominator for x in cp]); f = [int(x * L) for x in cp]
    pats = set(); pp = primes()
    for _ in range(200):
        p = next(pp)
        if f[0] % p == 0: continue
        fp = [x % p for x in f]
        der = [(x * (4 - i)) % p for i, x in enumerate(fp[:-1])]
        if len(gcd(fp, der, p)) > 1: continue        # not squarefree mod p
        pats.add(tuple(ddf(fp, p)))
    irred = (4,) in pats or ((1, 3) in pats and (2, 2) in pats)
    print(f'N={N}: factorisation patterns mod p seen: {sorted(pats)} -> irreducible over Q: {irred}')
