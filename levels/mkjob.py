import json, math, numpy as np, sys
sys.path.insert(0, '.')
from lapjob import build, det_mod
k = 7; A, B = build(k); p = 1073741789
M = (A + 11 * B) % p
Mw = M.copy(); rows, cols = Mw.shape; r = 0; piv = []; perm = list(range(rows))
for c in range(cols):
    nz = np.nonzero(Mw[r:, c])[0]
    if len(nz) == 0: continue
    pr = r + int(nz[0]); Mw[[r, pr]] = Mw[[pr, r]]; perm[r], perm[pr] = perm[pr], perm[r]
    inv = pow(int(Mw[r, c]), p - 2, p); f = (Mw[r + 1:, c] * inv) % p
    Mw[r + 1:] = (Mw[r + 1:] - (f[:, None] * Mw[r][None, :]) % p) % p
    piv.append(c); r += 1
rs, cs = perm[:r], piv
print('minor size', len(rs), 'dim K =', cols - len(cs))
S = np.abs(A[np.ix_(rs, cs)]) + np.abs(B[np.ix_(rs, cs)])
bits = sum(math.log2(max(1, int(s))) for s in S.sum(axis=1))
need = math.ceil((bits + 2) / 29)
print(f'Hadamard bound 2^{bits:.0f}; primes needed {need}')
def primes_below(start, count):
    out = []; n = start
    while len(out) < count:
        n -= 1
        if n % 2 and all(n % q for q in range(3, int(n ** .5) + 1, 2)): out.append(n)
    return out
ps = primes_below(2 ** 30 - 1000, need + 8)
job = {'k': k, 'rows': [int(x) for x in rs], 'cols': [int(x) for x in cs], 'degree_bound': len(rs),
       'primes_a': ps[0::2], 'primes_b': ps[1::2]}
json.dump(job, open('job.json', 'w'))
print('primes per laptop:', len(job['primes_a']), len(job['primes_b']))
