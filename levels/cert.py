# Certificates (C2) finiteness and (C3) closure, with N symbolic.
import pickle, itertools
from fractions import Fraction as Fr
import poly as P
S = pickle.load(open('step1sym.pkl', 'rb')); cr6, rec = S['cr6'], S['rec']
Dz = pickle.load(open('design.pkl', 'rb'))
data = {}
for f in ('step2sel_b.pkl', 'step2sel_c.pkl'): data.update(pickle.load(open(f, 'rb')))
Ns = sorted(data); FIT = [N for N in Ns if N <= 15]; CHK = [N for N in Ns if N > 15]
n = 5; sym = [(a, b) for a in range(n) for b in range(a, n)]
DEGB = 14                                   # a priori bound: 4 + 4 + 6 closed loops
assert len(FIT) >= DEGB + 1
# ---- 1. coefficient polynomials o[conf][P][i]
confs = sorted(data[Ns[0]])
O = {}
for cf in confs:
    keys = set().union(*[set(data[N][cf]) for N in Ns])
    O[cf] = {}
    for Pp in keys:
        vec = []
        for i in range(len(sym)):
            ys = [data[N][cf].get(Pp, [Fr(0)] * 15)[i] for N in FIT]
            p = P.interp(FIT, ys)
            assert P.deg(p) <= DEGB
            for N in CHK: assert P.ev(p, N) == data[N][cf].get(Pp, [Fr(0)] * 15)[i]
            vec.append(p)
        O[cf][Pp] = vec
maxd = max(P.deg(p) for cf in O for v in O[cf].values() for p in v)
print('coefficient polynomials interpolated and checked; max degree', maxd, flush=True)

def unit(i): return tuple(int(k == i) for k in range(n))
def madd(a, b): return tuple(x + y for x, y in zip(a, b))
def monos(d, nv): return sorted((m for m in itertools.product(range(d + 1), repeat=nv) if sum(m) == d), reverse=True)
def polyform(vec):                          # quadric: mono -> poly in N
    q = {}
    for p, (a, b) in zip(vec, sym):
        if p: m = madd(unit(a), unit(b)); q[m] = P.add(q.get(m, []), p)
    return {m: p for m, p in q.items() if p}

def det_exact(M):
    M = [r[:] for r in M]; k = len(M); D = Fr(1)
    for c in range(k):
        pv = next((r for r in range(c, k) if M[r][c] != 0), None)
        if pv is None: return Fr(0)
        if pv != c: M[c], M[pv] = M[pv], M[c]; D = -D
        D *= M[c][c]; inv = 1 / M[c][c]
        for r in range(c + 1, k):
            if M[r][c] != 0:
                f = M[r][c] * inv; M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return D
def choose_rows(rows, cols, N0=7):
    """rows: list of dict mono->poly; pick a maximal independent set at N=N0 (exact)."""
    ci = {m: i for i, m in enumerate(cols)}; basis = []; chosen = []
    for idx, r in enumerate(rows):
        v = [Fr(0)] * len(cols)
        for m, p in r.items(): v[ci[m]] = P.ev(p, N0)
        for (pc, bv) in basis:
            if v[pc] != 0:
                f = v[pc] / bv[pc]; v = [x - f * y for x, y in zip(v, bv)]
        nz = next((j for j, x in enumerate(v) if x != 0), None)
        if nz is not None: basis.append((nz, v)); chosen.append(idx)
    return chosen
def analyse(name, rows, cols):
    sel = choose_rows(rows, cols)
    assert len(sel) == len(cols), f'{name}: rank {len(sel)} < {len(cols)} at N=7'
    rows = [rows[i] for i in sel]; ci = {m: i for i, m in enumerate(cols)}
    rdeg = [max(P.deg(p) for p in r.values()) for r in rows]
    bound = sum(rdeg)
    xs = list(range(0, bound + 3)); ys = []
    for N in xs:
        M = [[Fr(0)] * len(cols) for _ in rows]
        for i, r in enumerate(rows):
            for m, p in r.items(): M[i][ci[m]] = P.ev(p, N)
        ys.append(det_exact(M))
    det = P.interp(xs[:-2], ys[:-2])
    assert all(P.ev(det, x) == y for x, y in zip(xs[-2:], ys[-2:]))
    print(f'{name}: {len(cols)}x{len(cols)} minor, degree bound {bound}, actual degree {P.deg(det)}', flush=True)
    rem = det; facs = {}
    for r in range(-30, 31):
        while rem and P.ev(rem, r) == 0:
            rem, z = P.divmod_(rem, [Fr(-r), Fr(1)]); assert not z; facs[r] = facs.get(r, 0) + 1
    print(f'  integer linear factors (N-r)^e: {facs}; remaining degree {P.deg(rem)}', flush=True)
    # Descartes on rem(6 + x): number of sign changes bounds the roots in (6, oo)
    sh = [Fr(0)]
    for c in reversed(rem): sh = P.add(P.mul(sh, [Fr(6), Fr(1)]), [c])
    signs = [x > 0 for x in sh if x != 0]
    ch = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
    print(f'  remaining factor: value at N=6 nonzero: {P.ev(rem, 6) != 0}; Descartes sign changes on (6,oo): {ch}', flush=True)
    return det, rem, facs, ch

# ---- (C2): quadrics restricted to c3 = 0, Macaulay degree 3 in c0,c1,c2,c4
h = 3; keep = [i for i in range(5) if i != h]
Q = [polyform(O[k[:5]][k[5]]) for k in Dz['Qsel']]
Qh = [{tuple(m[i] for i in keep): p for m, p in q.items() if m[h] == 0} for q in Q]
rows2 = [{madd(mm, m): p for m, p in q.items()} for q in Qh for mm in monos(1, 4)]
res2 = analyse('(C2) finiteness', rows2, monos(3, 4))

# ---- (C3): closure cubics o_Pa xi_Pb - o_Pb xi_Pa
cr_index = {Pp: a for a, Pp in enumerate(cr6)}
def xi_coef(Pp, e): return rec[e][cr_index[Pp]] if Pp in cr_index else []
cubics = []
for (cf, Pa, Pb) in Dz['C3keys']:
    cub = {}
    for oP, xP, sg in [(Pa, Pb, 1), (Pb, Pa, -1)]:
        if oP not in O[cf]: continue
        for p, (a, b) in zip(O[cf][oP], sym):
            if not p: continue
            for e in range(5):
                w = xi_coef(xP, e)
                if w:
                    m = madd(madd(unit(a), unit(b)), unit(e))
                    cub[m] = P.add(cub.get(m, []), P.scal(Fr(sg), P.mul(p, w)))
    cubics.append({m: p for m, p in cub.items() if p})
res3 = analyse('(C3) closure', cubics, monos(3, 5))
pickle.dump({'C2': res2, 'C3': res3, 'O': O}, open('cert.pkl', 'wb'))
