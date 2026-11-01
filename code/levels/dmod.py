# D(N) = det of the 105 selected level-7 quadric rows, modulo a prime p, as a polynomial over F_p.
# Pure numpy; this is the unit of work for the laptops.
import pickle, sys, time, math
import numpy as np

def load(path='exactrows.pkl'):
    E = pickle.load(open(path, 'rb'))
    polys = E['polys']                         # 105 rows x 105 entries, each list of Fractions low->high
    den = 1
    for row in polys:
        for pol in row:
            for c in pol: den = den * c.denominator // math.gcd(den, c.denominator)
    rows = [[[int(c * den) for c in pol] for pol in row] for row in polys]
    return rows, den

def hadamard_bits(rows):
    return sum(math.log2(max(1, sum(sum(abs(c) for c in pol) for pol in row))) for row in rows)

def det_mod(M, p):
    M = M.copy(); n = M.shape[0]; D = 1
    for c in range(n):
        nz = np.nonzero(M[c:, c])[0]
        if len(nz) == 0: return 0
        r = c + nz[0]
        if r != c: M[[c, r]] = M[[r, c]]; D = -D
        piv = int(M[c, c]); D = D * piv % p
        inv = pow(piv, p - 2, p)
        f = (M[c + 1:, c] * inv) % p
        M[c + 1:, c:] = (M[c + 1:, c:] - (f[:, None] * M[c, c:][None, :]) % p) % p
    return D % p

def poly_mod(rows, p, deg_bound):
    n = len(rows); L = max(len(pol) for row in rows for pol in row)
    Cf = np.zeros((n, n, L), dtype=np.int64)
    for i, row in enumerate(rows):
        for j, pol in enumerate(row):
            for e, c in enumerate(pol): Cf[i, j, e] = c % p
    xs = list(range(deg_bound + 1)); ys = []
    for x in xs:
        V = np.zeros((n, n), dtype=np.int64)
        for e in range(L - 1, -1, -1): V = (V * x + Cf[:, :, e]) % p
        ys.append(det_mod(V, p))
    # Newton interpolation mod p
    coef = ys[:]
    for j in range(1, len(xs)):
        for i in range(len(xs) - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) * pow(xs[i] - xs[i - j], p - 2, p) % p
    poly = [0]
    for i in range(len(xs) - 1, -1, -1):
        new = [0] * (len(poly) + 1)
        for t, a in enumerate(poly):
            new[t + 1] = (new[t + 1] + a) % p
            new[t] = (new[t] - a * xs[i]) % p
        new[0] = (new[0] + coef[i]) % p
        poly = new
    while len(poly) > 1 and poly[-1] == 0: poly.pop()
    return poly

if __name__ == '__main__':
    rows, den = load()
    rdeg = [max(len(pol) - 1 for pol in row) for row in rows]
    DB = sum(rdeg)
    hb = hadamard_bits(rows)
    print(f'degree bound {DB}; Hadamard bound on |coefficients of D|: 2^{hb:.0f} -> {math.ceil((hb + 2) / 30)} primes of 30 bits', flush=True)
    p = 1073741789
    t = time.time(); poly = poly_mod(rows, p, DB)
    print(f'one prime: {time.time()-t:.1f}s; degree of D mod p = {len(poly) - 1}', flush=True)
    z = 0
    while poly[z] == 0: z += 1
    print('order of vanishing at N=0 (mod p):', z)
