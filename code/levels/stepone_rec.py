# Reconstruct the level-k kernel basis as (polynomial vector)/D(N), then verify symbolically.
import sys, pickle, time
sys.argv = [sys.argv[0], sys.argv[1]]
exec(open('stepone.py').read().split("t = time.time(); free, basis = kernel(11)")[0])
from math import lcm
Ns = list(range(7, 7 + 70))
data = {}
for N in Ns:
    fr, b = kernel(N); assert fr == kernel(11)[0] if N == 7 else True
    data[N] = (fr, b)
frees = {tuple(data[N][0]) for N in Ns}; assert len(frees) == 1, frees
free = data[Ns[0]][0]
L = {N: lcm(*[x.denominator for v in data[N][1] for x in v.values()]) for N in Ns}
print('lcm denominators N=7..14:', [L[N] for N in Ns[:8]], flush=True)
# Pade reconstruction of the largest-denominator entry: find p/q with deg q = dq, deg p = dp
def pade(vals, xs, dp, dq):
    # unknowns: p_0..p_dp, q_0..q_dq (q monic: q_dq = 1). Equations p(x) - v q(x) = 0.
    n = dp + 1 + dq
    rows = []
    for x, v in zip(xs, vals):
        r = [Fr(x) ** i for i in range(dp + 1)] + [-v * Fr(x) ** i for i in range(dq)]
        rows.append(r + [v * Fr(x) ** dq])
    M = [r[:] for r in rows]; piv = []; rk = 0
    for c in range(n):
        pr = next((i for i in range(rk, len(M)) if M[i][c] != 0), None)
        if pr is None: continue
        M[rk], M[pr] = M[pr], M[rk]; v = M[rk][c]; M[rk] = [x / v for x in M[rk]]
        for i in range(len(M)):
            if i != rk and M[i][c] != 0:
                f = M[i][c]; M[i] = [x - f * y for x, y in zip(M[i], M[rk])]
        piv.append(c); rk += 1
    if any(any(x != 0 for x in r) for r in M[rk:]): return None
    if rk < n: return None
    sol = [M[i][n] for i in range(n)]
    return sol[:dp + 1], sol[dp + 1:] + [Fr(1)]
# find the entry whose denominator is largest at N=11
best = max(((b, c) for b in range(len(free)) for c in data[11][1][b]), key=lambda bc: data[11][1][bc[0]][bc[1]].denominator)
b0, c0 = best
xs = Ns[:40]
vals = [data[N][1][b0].get(c0, Fr(0)) for N in xs]
found = None
for tot in range(1, 30):
    for dq in range(1, tot + 1):
        dp = tot - dq
        r = pade(vals, xs, dp, dq)
        if r:
            p_, q_ = r
            if all(P.ev(p_, N) == data[N][1][b0].get(c0, Fr(0)) * P.ev(q_, N) for N in Ns[40:]):
                found = (p_, q_); break
    if found: break
den = P.trim(found[1])
print('denominator from one entry: deg', P.deg(den), ':', P.fmt(den), flush=True)
# try den, then den times small factors, as common denominator
def try_common(den):
    fit = Ns[:45]; test = Ns[45:]
    rec = []
    for b in range(len(free)):
        vec = {}
        for c in range(ncol):
            ys = [data[N][1][b].get(c, Fr(0)) * P.ev(den, N) for N in fit]
            if all(y == 0 for y in ys) and all(data[N][1][b].get(c, 0) == 0 for N in test): continue
            p_ = P.interp(fit, ys)
            if P.deg(p_) > 30: return None
            if any(P.ev(p_, N) != data[N][1][b].get(c, Fr(0)) * P.ev(den, N) for N in test): return None
            vec[c] = p_
        rec.append(vec)
    return rec
rec = try_common(den)
extra = P.const(1)
if rec is None:
    for f in ([Fr(0), Fr(1)], [Fr(-1), Fr(1)], [Fr(-2), Fr(1)], [Fr(-3), Fr(1)]):
        rec = try_common(P.mul(den, f))
        if rec: den = P.mul(den, f); break
print('common denominator works:', rec is not None, ' D(N) =', P.fmt(den), flush=True)
maxdeg = max(P.deg(p_) for v in rec for p_ in v.values())
print('max numerator degree:', maxdeg, flush=True)
# symbolic verification
ok = True
for ra, rb in zip(rowsA, rowsB):
    cols = set(ra) | set(rb)
    for b in range(len(free)):
        s = []
        for c in cols:
            if c in rec[b]:
                s = P.add(s, P.mul(P.add(P.const(ra.get(c, 0)), P.scal(Fr(rb.get(c, 0)), P.X())), rec[b][c]))
        if s: ok = False
print('symbolic check (A + N B) w_b(N) = 0 identically for all b:', ok, flush=True)
pickle.dump({'crk': crk, 'free': free, 'den': den, 'rec': rec, 'rowsA': rowsA, 'rowsB': rowsB}, open(f'stepone_{"x"*k}.pkl', 'wb'))
