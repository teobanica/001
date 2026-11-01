#!/usr/bin/env python3
# Level-7 Step 1 rank certificate, laptop job (Claude and Teo).
# Computes Delta(N) = det of a fixed 434x434 minor of M(N) = A + N B modulo a list of primes,
# as a polynomial in N over F_p (degree <= 434), and writes the results to a JSON file.
#
# Usage:   python3 lapjob.py a        (first laptop)
#          python3 lapjob.py b        (second laptop)
#          python3 lapjob.py test     (quick timing test, about 1-2 minutes)
# Needs:   numpy   (install once with:  python3 -m pip install numpy)
import sys, json, time, os, itertools
import multiprocessing as mp

WORKERS = 6           # MacBook Air M3: 6 of the 8 cores, to stay cool

# ---------------- partitions (self-contained) ----------------
def set_partitions(k):
    def rec(l):
        if l == 0:
            yield []
            return
        for p in rec(l - 1):
            for i in range(len(p)):
                yield p[:i] + [p[i] + [l - 1]] + p[i + 1:]
            yield p + [[l - 1]]
    return list(rec(k))
def canon(blocks):
    return tuple(sorted(tuple(sorted(b)) for b in blocks if b))
def noncrossing(P):
    for A, B in itertools.permutations(P, 2):
        for a1, a2 in itertools.combinations(A, 2):
            for b1, b2 in itertools.combinations(B, 2):
                if a1 < b1 < a2 < b2:
                    return False
    return True
def parts(k):
    return [canon([set(b) for b in p]) for p in set_partitions(k)]
def delete_leg(P, i, k):
    sing = any(b == (i,) for b in P)
    nb = [set(x if x < i else x - 1 for x in b if x != i) for b in P]
    return sing, canon(nb)
def merge(P, i, j, k):
    bi = [b for b in P if i in b][0]; bj = [b for b in P if j in b][0]
    blocks = [set(b) for b in P if b not in (bi, bj)] + [set(bi) | set(bj)]
    if j == i + 1:
        f = lambda x: x if x < j else x - 1
        nb = [set(f(x) for x in b if x != j) for b in blocks]
    else:
        nb = [set(x for x in b if x != i) for b in blocks]
    return canon(nb)

def build(k):
    import numpy as np
    Pk = parts(k); Pl = parts(k - 1)
    crk = sorted([q for q in Pk if not noncrossing(q)])
    crl = sorted([q for q in Pl if not noncrossing(q)])
    il = {q: a for a, q in enumerate(crl)}
    A = np.zeros((2 * k * len(crl), len(crk)), dtype=np.int64); B = np.zeros_like(A)
    r0 = 0
    for typ, i in [('d', i) for i in range(k)] + [('m', i) for i in range(k)]:
        for a, P in enumerate(crk):
            if typ == 'd':
                sing, q = delete_leg(P, i, k)
                if q in il:
                    (B if sing else A)[r0 + il[q], a] += 1
            else:
                q = merge(P, i, (i + 1) % k, k)
                if q in il:
                    A[r0 + il[q], a] += 1
        r0 += len(crl)
    return A, B

# ---------------- modular linear algebra ----------------
def det_mod(M, p):
    import numpy as np
    M = M.copy(); n = M.shape[0]; D = 1
    for c in range(n):
        nz = np.nonzero(M[c:, c])[0]
        if len(nz) == 0:
            return 0
        r = c + int(nz[0])
        if r != c:
            M[[c, r]] = M[[r, c]]; D = -D
        piv = int(M[c, c]); D = D * piv % p
        inv = pow(piv, p - 2, p)
        f = (M[c + 1:, c] * inv) % p
        M[c + 1:, c:] = (M[c + 1:, c:] - (f[:, None] * M[c, c:][None, :]) % p) % p
    return D % p

def interp_mod(xs, ys, p):
    coef = list(ys)
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
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
    return poly

G = {}
def init(job):
    import numpy as np
    A, B = build(job['k'])
    rs, cs = job['rows'], job['cols']
    G['A'] = A[np.ix_(rs, cs)]; G['B'] = B[np.ix_(rs, cs)]; G['deg'] = job['degree_bound']

def one_prime(p):
    t = time.time()
    A = G['A'] % p; B = G['B'] % p
    xs = list(range(G['deg'] + 1))
    ys = [det_mod((A + x * B) % p, p) for x in xs]
    return p, interp_mod(xs, ys, p), time.time() - t

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'test'
    here = os.path.dirname(os.path.abspath(__file__))
    job = json.load(open(os.path.join(here, 'job.json')))
    primes = job['primes_' + which] if which != 'test' else job['primes_a'][:1]
    print(f'job {which}: {len(primes)} primes, {WORKERS} workers', flush=True)
    t0 = time.time(); out = {}
    if which == 'test':
        init(job); p, poly, dt = one_prime(primes[0])
        print(f'test prime done in {dt:.1f}s, degree {len(poly) - 1}. Setup OK.', flush=True)
        return
    with mp.Pool(WORKERS, initializer=init, initargs=(job,)) as pool:
        for i, (p, poly, dt) in enumerate(pool.imap_unordered(one_prime, primes), 1):
            out[str(p)] = poly
            el = time.time() - t0
            print(f'  {i}/{len(primes)} primes done   elapsed {el/60:.1f} min   remaining about {el/i*(len(primes)-i)/60:.1f} min', flush=True)
    res = os.path.join(here, 'result' + which + '.json')
    json.dump({'k': job['k'], 'which': which, 'polys': out}, open(res, 'w'))
    print(f'finished: {res}  (send this file back)', flush=True)

if __name__ == '__main__':
    mp.freeze_support()
    main()
