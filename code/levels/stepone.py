# Step 1 at level k with N symbolic (sparse exact elimination), generalising step1sym.py.
# Usage: python3 stepone.py k
import sys, time, pickle
from fractions import Fraction as Fr
from levlin import parts, noncrossing, delete_leg, merge
import poly as P
k = int(sys.argv[1])
Pk = parts(k); Pl = parts(k - 1)
crk = [p for p in Pk if not noncrossing(p)]; crl = [q for q in Pl if not noncrossing(q)]
il = {q: a for a, q in enumerate(crl)}
rowsA = []; rowsB = []                      # sparse rows: dict col -> int
for typ, i in [('d', i) for i in range(k)] + [('m', i) for i in range(k)]:
    RA = [dict() for _ in crl]; RB = [dict() for _ in crl]
    for a, p in enumerate(crk):
        if typ == 'd':
            sing, q = delete_leg(p, i, k)
            if q in il:
                R = RB if sing else RA; R[il[q]][a] = R[il[q]].get(a, 0) + 1
        else:
            q = merge(p, i, (i + 1) % k, k)
            if q in il: RA[il[q]][a] = RA[il[q]].get(a, 0) + 1
    rowsA += RA; rowsB += RB
ncol = len(crk)
print(f'level {k}: M(N) = A + N B of size {len(rowsA)} x {ncol}', flush=True)

def kernel(N):
    rows = []
    for ra, rb in zip(rowsA, rowsB):
        r = {c: Fr(ra.get(c, 0) + N * rb.get(c, 0)) for c in set(ra) | set(rb)}
        r = {c: v for c, v in r.items() if v != 0}
        if r: rows.append(r)
    piv = {}                                 # col -> reduced row
    for r in rows:
        r = dict(r)
        while True:
            hit = [c for c in r if c in piv]
            if not hit: break
            c = min(hit); f = r[c]
            for cc, v in piv[c].items():
                nv = r.get(cc, 0) - f * v
                if nv == 0: r.pop(cc, None)
                else: r[cc] = nv
        if not r: continue
        c = min(r); f = r[c]; r = {cc: v / f for cc, v in r.items()}
        for pc in list(piv):
            pr = piv[pc]
            if c in pr:
                g = pr[c]
                for cc, v in r.items():
                    nv = pr.get(cc, 0) - g * v
                    if nv == 0: pr.pop(cc, None)
                    else: pr[cc] = nv
        piv[c] = r
    free = [c for c in range(ncol) if c not in piv]
    basis = []
    for fc in free:
        v = {fc: Fr(1)}
        for pc, pr in piv.items():
            if fc in pr: v[pc] = -pr[fc]
        basis.append(v)
    return free, basis
t = time.time(); free, basis = kernel(11)
print(f'N=11: dim K = {len(free)}, free columns {free}, time {time.time()-t:.1f}s', flush=True)
from math import lcm
print('  lcm of denominators:', lcm(*[x.denominator for v in basis for x in v.values()]), flush=True)
