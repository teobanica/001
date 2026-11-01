# dim K_k(N) mod p with numpy: K_k = ker of leg-sums and merges on crossing columns, into crossing part of level k-1.
import sys, time, numpy as np
from levlin import parts, noncrossing, delete_leg, merge
p = 2147483629
def build(k):
    Pk = parts(k); Pl = parts(k - 1)
    crk = [q for q in Pk if not noncrossing(q)]; crl = [q for q in Pl if not noncrossing(q)]
    il = {q: a for a, q in enumerate(crl)}
    A = np.zeros((2 * k * len(crl), len(crk)), dtype=np.int64); B = np.zeros_like(A)
    r0 = 0
    for typ, i in [('d', i) for i in range(k)] + [('m', i) for i in range(k)]:
        for a, P in enumerate(crk):
            if typ == 'd':
                sing, q = delete_leg(P, i, k)
                if q in il: (B if sing else A)[r0 + il[q], a] += 1
            else:
                q = merge(P, i, (i + 1) % k, k)
                if q in il: A[r0 + il[q], a] += 1
        r0 += len(crl)
    return A, B, crk, crl
def rank_mod(M):
    M = M.copy() % p; r = 0; rows, cols = M.shape
    for c in range(cols):
        nz = np.nonzero(M[r:, c])[0]
        if len(nz) == 0: continue
        piv = r + nz[0]; M[[r, piv]] = M[[piv, r]]
        inv = pow(int(M[r, c]), p - 2, p); M[r] = (M[r] * inv) % p
        f = M[:, c].copy(); f[r] = 0
        nzr = np.nonzero(f)[0]
        if len(nzr): M[nzr] = (M[nzr] - (f[nzr, None] * M[r][None, :]) % p) % p
        r += 1
        if r == rows: break
    return r
k = int(sys.argv[1])
t = time.time(); A, B, crk, crl = build(k)
print(f'level {k}: {len(crk)} crossing columns, {len(crl)} crossing partitions at level {k-1}, matrix {A.shape}, built in {time.time()-t:.1f}s', flush=True)
for N in [int(x) for x in sys.argv[2:]]:
    t = time.time(); r = rank_mod(A + N * B)
    print(f'  N={N}: rank {r}, dim K = {len(crk) - r}   ({time.time()-t:.1f}s)', flush=True)
