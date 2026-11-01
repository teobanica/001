# Exact level-6 closure test (Step 3) in the quartic field K = Q[t]/(chi).
# The 4 lines are Galois conjugate, so one computation in K treats all of them.
import sys, pickle, io, contextlib
from fractions import Fraction as Fr
with contextlib.redirect_stdout(io.StringIO()):
    from levquad import Kbasis, contract
from levlin import noncrossing

N = int(sys.argv[1])
E = pickle.load(open(f'elim_N{N}.pkl', 'rb'))
Q, sym, vecs = pickle.load(open(f'quad_N{N}.pkl', 'rb'))
chi = [Fr(x) for x in E['charpoly']]            # monic, high -> low, degree 4
T = [[[Fr(x) for x in r] for r in t] for t in E['T']]
g = E['g']; n = 5; D = 4

# --- arithmetic in K: elements are lists of 4 Fractions, low -> high
mono = [-c for c in chi[1:]][::-1]               # t^4 = sum mono[k] t^k
def kmul(a, b):
    r = [Fr(0)] * 7
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                if y: r[i + j] += x * y
    for k in range(6, 3, -1):
        if r[k]:
            c = r[k]; r[k] = Fr(0)
            for j in range(4): r[k - 4 + j] += c * mono[j]
    return r[:4]
def kadd(a, b): return [x + y for x, y in zip(a, b)]
def kscal(s, a): return [s * x for x in a]
Z = [Fr(0)] * 4
def iszero(a): return all(x == 0 for x in a)

# --- express T_i as polynomials in T_g (T_g is nonderogatory: distinct eigenvalues)
def matmul(A, B): return [[sum(A[i][k] * B[k][j] for k in range(D)) for j in range(D)] for i in range(D)]
Tg = [[sum(g[i] * T[i][r][s] for i in range(n)) for s in range(D)] for r in range(D)]
pows = [[[Fr(int(i == j)) for j in range(D)] for i in range(D)]]
for _ in range(3): pows.append(matmul(pows[-1], Tg))
def solve_poly(M):              # find a_0..a_3 with sum a_k Tg^k = M (least: exact consistent system)
    rows = [[pows[k][r][s] for k in range(D)] + [M[r][s]] for r in range(D) for s in range(D)]
    piv = []; R = [r[:] for r in rows]; rk = 0
    for c in range(D):
        p = next((i for i in range(rk, len(R)) if R[i][c] != 0), None)
        if p is None: continue
        R[rk], R[p] = R[p], R[rk]; v = R[rk][c]; R[rk] = [x / v for x in R[rk]]
        for i in range(len(R)):
            if i != rk and R[i][c] != 0:
                f = R[i][c]; R[i] = [x - f * y for x, y in zip(R[i], R[rk])]
        piv.append(c); rk += 1
    assert rk == D and all(all(x == 0 for x in r) for r in R[rk:]), 'inconsistent'
    return [R[i][D] for i in range(D)]
# eigenvalue of T_g on a point is t; then c_i/ell = p_i(t)
c = [solve_poly(T[i]) for i in range(n)]
print(f'N={N}: point coordinates c_i = p_i(t) in Q[t]/(chi), chi irreducible of degree 4')

# --- sanity: all quadrics vanish exactly at the point
ok = True
for row in Q:
    s = Z
    for x, (a, b) in zip(row, sym):
        if x: s = kadd(s, kscal(Fr(x), kmul(c[a], c[b])))
    ok &= iszero(s)
print('  all 9 quadrics vanish exactly at the generic point:', ok)
# sanity: c0 = c2 and c1 = c4 exactly
print('  c0 = c2 exactly:', c[0] == c[2], '  c1 = c4 exactly:', c[1] == c[4])

# --- xi = sum_a c_a vecs[a]: crossing coordinates in K
crossP = sorted({P for v in vecs for P in v}, key=str)
def kvec_of_lin(coeffs_by_P): return coeffs_by_P
xi = {}
for a, v in enumerate(vecs):
    for P, x in v.items():
        xi[P] = kadd(xi.get(P, Z), kscal(Fr(x), c[a]))
xi = {P: w for P, w in xi.items() if not iszero(w)}

# --- contractions landing at level 6: (j,o1,o2) in {(3,0,0),(4,1,1)}
viol = 0; tests = 0; worst = None
for s in range(6):
    for t in range(6):
        for j, o1, o2 in [(3, 0, 0), (4, 1, 1)]:
            out = {}
            for a in range(n):
                for b in range(n):
                    L, r = contract(vecs[a], vecs[b], s, t, j, o1, o2, N)
                    cab = kmul(c[a], c[b])
                    for P, x in r.items():
                        if noncrossing(P): continue
                        out[P] = kadd(out.get(P, Z), kscal(Fr(x), cab))
            out = {P: w for P, w in out.items() if not iszero(w)}
            tests += 1
            # need out = lambda * xi on crossing coordinates: all 2x2 minors vanish in K
            keys = set(out) | set(xi)
            ref = next(iter(xi))                   # a coordinate where xi != 0
            x0 = xi[ref]; o0 = out.get(ref, Z)
            bad = False
            for P in keys:
                m = kadd(kmul(out.get(P, Z), x0), kscal(Fr(-1), kmul(o0, xi.get(P, Z))))
                if not iszero(m): bad = True; break
            if bad: viol += 1
print(f'  level-6 closure: {tests} contractions, {viol} leave NC(6) + C xi  (exact, in K)')
print('  => all 4 lines killed exactly' if viol else '  => some line SURVIVES closure')
