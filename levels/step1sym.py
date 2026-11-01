# Step 1 with N symbolic: K(N) = ker M(N), M(N) = A + N B on the 71 crossing columns of P(6).
import pickle, sys
from fractions import Fraction as Fr
from levlin import parts, noncrossing, delete_leg, merge
import poly as P

k = 6
P6 = parts(k); P5 = parts(k - 1)
cr6 = [p for p in P6 if not noncrossing(p)]; cr5 = [q for q in P5 if not noncrossing(q)]
i6 = {p: a for a, p in enumerate(cr6)}; i5 = {q: a for a, q in enumerate(cr5)}
A = []; B = []
for typ, i in [('d', i) for i in range(k)] + [('m', i) for i in range(k)]:
    RA = [[0] * len(cr6) for _ in cr5]; RB = [[0] * len(cr6) for _ in cr5]
    for a, p in enumerate(cr6):
        if typ == 'd':
            sing, q = delete_leg(p, i, k)
            if q in i5:
                (RB if sing else RA)[i5[q]][a] += 1
        else:
            q = merge(p, i, (i + 1) % k, k)
            if q in i5: RA[i5[q]][a] += 1
    A += RA; B += RB
# NC columns: check they are zero (moves preserve NC), so K = ker on crossing columns
for typ, i in [('d', i) for i in range(k)] + [('m', i) for i in range(k)]:
    for p in P6:
        if noncrossing(p):
            q = delete_leg(p, i, k)[1] if typ == 'd' else merge(p, i, (i + 1) % k, k)
            assert noncrossing(q)
print(f'M(N) = A + N B : {len(A)} x {len(cr6)}, NC columns vanish identically', flush=True)

def kernel_rref(N):
    rows = [[Fr(a + N * b) for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]
    M = [r[:] for r in rows]; piv = []; r = 0; n = len(cr6)
    for c in range(n):
        p = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if p is None: continue
        M[r], M[p] = M[p], M[r]; v = M[r][c]; M[r] = [x / v for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c]; M[i] = [x - f * y for x, y in zip(M[i], M[r])]
        piv.append(c); r += 1
    free = [c for c in range(n) if c not in piv]
    basis = []
    for f in free:
        v = [Fr(0)] * n; v[f] = Fr(1)
        for i, pc in enumerate(piv): v[pc] = -M[i][f]
        basis.append(v)
    return piv, free, basis

samples = list(range(6, 6 + 40))
data = {}
for N in samples:
    piv, free, basis = kernel_rref(N); data[N] = (tuple(free), basis)
frees = {data[N][0] for N in samples}
print('free columns at sampled N:', frees, flush=True)
free = data[samples[0]][0]
assert len(frees) == 1
# denominators observed
from math import lcm
dens = {N: lcm(*[x.denominator for v in data[N][1] for x in v]) for N in samples[:8]}
print('lcm of denominators at N=6..13:', dens, flush=True)

# reconstruct with candidate denominator den(N); fit on first 25 points, check on the rest
def try_den(den):
    fit = samples[:25]; test = samples[25:]
    rec = []
    for b in range(5):
        vec = []
        for a in range(len(cr6)):
            ys = [data[N][1][b][a] * P.ev(den, N) for N in fit]
            p = P.interp(fit, ys)
            if P.deg(p) > 15: return None
            if any(P.ev(p, N) != data[N][1][b][a] * P.ev(den, N) for N in test): return None
            vec.append(p)
        rec.append(vec)
    return rec
for name, den in [('1', P.const(1)), ('N', P.X()), ('N(N-2)', P.mul(P.X(), [Fr(-2), Fr(1)])),
                  ('N^2(N-2)^2', P.mul(P.mul(P.X(), P.X()), P.mul([Fr(-2), Fr(1)], [Fr(-2), Fr(1)])))]:
    rec = try_den(den)
    print(f'denominator {name}:', 'fits' if rec else 'no', flush=True)
    if rec: break
maxdeg = max(P.deg(p) for v in rec for p in v)
print('max numerator degree:', maxdeg, flush=True)

# symbolic verification: (A + N B) v(N) = 0 as polynomials (v = numerators / den)
Npoly = P.X(); ok = True
for row_a, row_b in zip(A, B):
    for b in range(5):
        s = []
        for a in range(len(cr6)):
            coef = P.add(P.const(row_a[a]), P.scal(Fr(row_b[a]), Npoly)) if (row_a[a] or row_b[a]) else []
            if coef: s = P.add(s, P.mul(coef, rec[b][a]))
        if s: ok = False
print('symbolic check (A + N B) v_b(N) = 0 identically, b = 1..5:', ok, flush=True)

# rank certificate: 66 x 66 minor on pivot columns, determinant as exact polynomial in N
piv = [c for c in range(len(cr6)) if c not in free]
# choose 66 rows independent at N = 7
def rref_rows(N):
    rows = [[Fr(a + N * b) for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]
    chosen = []; basis = []
    for i, r in enumerate(rows):
        v = [r[c] for c in piv]
        for (pc, bv) in basis:
            if v[pc] != 0:
                f = v[pc] / bv[pc]; v = [x - f * y for x, y in zip(v, bv)]
        nz = next((j for j, x in enumerate(v) if x != 0), None)
        if nz is not None: basis.append((nz, v)); chosen.append(i)
        if len(chosen) == len(piv): break
    return chosen
rows66 = rref_rows(7)
print('rows chosen for the minor:', len(rows66), flush=True)
def det(Mx):
    M = [r[:] for r in Mx]; n = len(M); D = Fr(1)
    for c in range(n):
        p = next((r for r in range(c, n) if M[r][c] != 0), None)
        if p is None: return Fr(0)
        if p != c: M[c], M[p] = M[p], M[c]; D = -D
        D *= M[c][c]; inv = 1 / M[c][c]
        for r in range(c + 1, n):
            if M[r][c] != 0:
                f = M[r][c] * inv; M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return D
xs = list(range(0, len(piv) + 3)); ys = []
for N in xs:
    ys.append(det([[Fr(A[i][c] + N * B[i][c]) for c in piv] for i in rows66]))
Delta = P.interp(xs[:-2], ys[:-2])
assert P.ev(Delta, xs[-1]) == ys[-1] and P.ev(Delta, xs[-2]) == ys[-2]
print('Delta(N) = det of 66x66 minor: degree', P.deg(Delta), flush=True)
# strip factors N - r for small r
rem = Delta; facs = {}
for r in range(-10, 11):
    while rem and P.ev(rem, r) == 0:
        rem, z = P.divmod_(rem, [Fr(-r), Fr(1)]); assert not z; facs[r] = facs.get(r, 0) + 1
print('linear factors (N - r)^e found:', facs, ' remaining degree', P.deg(rem), flush=True)
print('remaining factor:', P.fmt(rem)[:300], flush=True)
pickle.dump({'cr6': cr6, 'free': free, 'den': None, 'rec': rec, 'Delta': Delta, 'rows66': rows66,
             'A': A, 'B': B}, open('step1sym.pkl', 'wb'))
