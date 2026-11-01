import sys, time, numpy as np
from levlin import parts, noncrossing, delete_leg, merge
p = 1073741789
k = int(sys.argv[1]); Ns = [int(x) for x in sys.argv[2:]]
t = time.time()
Pk = parts(k); Pl = parts(k - 1)
crk = [q for q in Pk if not noncrossing(q)]; crl = [q for q in Pl if not noncrossing(q)]
il = {q: a for a, q in enumerate(crl)}
entries = []   # (row, col, a, b)
r0 = 0
for typ, i in [('d', i) for i in range(k)] + [('m', i) for i in range(k)]:
    for a, P in enumerate(crk):
        if typ == 'd':
            sing, q = delete_leg(P, i, k)
            if q in il: entries.append((r0 + il[q], a, 0 if sing else 1, 1 if sing else 0))
        else:
            q = merge(P, i, (i + 1) % k, k)
            if q in il: entries.append((r0 + il[q], a, 1, 0))
    r0 += len(crl)
print(f'level {k}: {len(crk)} crossing cols, {r0} rows, built in {time.time()-t:.0f}s', flush=True)
def rank_mod(M):
    # work on the transpose if beneficial: rank is the same
    M = M % p; r = 0; rows, cols = M.shape
    for c in range(cols):
        col = M[r:, c]; nz = np.flatnonzero(col)
        if len(nz) == 0: continue
        pr = r + nz[0]
        if pr != r: M[[r, pr]] = M[[pr, r]]
        inv = pow(int(M[r, c]), p - 2, p); M[r, c:] = (M[r, c:] * inv) % p
        below = r + 1 + np.flatnonzero(M[r + 1:, c])
        if len(below):
            f = M[below, c].copy()
            M[np.ix_(below, np.arange(c, cols))] = (M[np.ix_(below, np.arange(c, cols))] - (f[:, None] * M[r, c:][None, :]) % p) % p
        r += 1
        if r == rows: break
    return r
for N in Ns:
    M = np.zeros((r0, len(crk)), dtype=np.int64)
    for (ro, co, a, b) in entries: M[ro, co] += a + N * b
    t = time.time(); rk = rank_mod(M)
    print(f'  N={N}: rank {rk}, dim K = {len(crk) - rk}  ({time.time()-t:.0f}s)', flush=True)
