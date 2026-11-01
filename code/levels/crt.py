# CRT reconstruction of an integer polynomial from its reductions mod many primes,
# with a proven coefficient bound; then integer-root analysis on [Nmin, oo).
import json, sys, math
from fractions import Fraction as Fr
import poly as P

def crt_poly(res, bound_bits):
    """res: dict p -> list of coeffs mod p (low->high). Returns integer coefficient list (symmetric residues)."""
    ps = sorted(res)
    M = 1; used = []
    for p in ps:
        used.append(p); M *= p
        if M.bit_length() > bound_bits + 2: break
    assert M.bit_length() > bound_bits + 1, 'not enough primes'
    L = max(len(res[p]) for p in used)
    coeffs = []
    for e in range(L):
        x = 0; m = 1
        for p in used:
            a = res[p][e] if e < len(res[p]) else 0
            # combine x mod m with a mod p
            t = ((a - x) * pow(m, -1, p)) % p
            x += m * t; m *= p
        if x > m // 2: x -= m
        coeffs.append(x)
    while len(coeffs) > 1 and coeffs[-1] == 0: coeffs.pop()
    return coeffs, used

def check_unused(coeffs, res, used):
    """Consistency check against primes not used in the reconstruction."""
    ok = True
    for p in res:
        if p in used: continue
        r = [c % p for c in coeffs]
        q = list(res[p]) + [0] * (len(r) - len(res[p]))
        r = r + [0] * (len(q) - len(r))
        ok &= (r == q)
    return ok

def analyse(coeffs, Nmin, name):
    pol = [Fr(c) for c in coeffs]
    print(f'{name}: degree {len(pol)-1}', flush=True)
    rem = pol; facs = {}
    for r in range(-40, 41):
        while P.ev(rem, r) == 0:
            rem, z = P.divmod_(rem, [Fr(-r), Fr(1)]); assert not z; facs[r] = facs.get(r, 0) + 1
    print(f'  integer linear factors (N-r)^e: {facs}; remaining degree {P.deg(rem)}', flush=True)
    def shift(p_, a):
        sh = [Fr(0)]
        for c in reversed(p_): sh = P.add(P.mul(sh, [Fr(a), Fr(1)]), [c])
        return sh
    def changes(p_):
        s = [x > 0 for x in p_ if x != 0]; return sum(1 for a, b in zip(s, s[1:]) if a != b)
    ch = changes(shift(rem, Nmin))
    print(f'  Descartes sign changes of remaining factor on ({Nmin}, oo): {ch}', flush=True)
    M = Nmin
    while changes(shift(rem, M)) > 0: M *= 2
    bad = [N for N in range(Nmin, M + 1) if P.ev(rem, N) == 0]
    print(f'  no roots beyond {M}; integer roots in [{Nmin}, {M}]: {bad}', flush=True)
    signs = ''.join('+' if P.ev(rem, N) > 0 else '-' for N in range(Nmin, min(M, Nmin + 40) + 1))
    print(f'  signs at N = {Nmin}..: {signs}', flush=True)
    return facs, rem, bad
