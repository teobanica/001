import pickle, sys, math
from fractions import Fraction as Fr
res = pickle.load(open('locus_x_zoo.pkl', 'rb'))
Ns = sorted(res)
def solve_mod(rows, p):
    # rows: augmented matrix; returns solution or None (unique solution required)
    M = [r[:] for r in rows]; n = len(M[0]) - 1; r = 0; piv = []
    for c in range(n):
        pr = next((i for i in range(r, len(M)) if M[i][c] % p), None)
        if pr is None: return None
        M[r], M[pr] = M[pr], M[r]; iv = pow(M[r][c], p - 2, p); M[r] = [x * iv % p for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] % p:
                f = M[i][c]; M[i] = [(x - f * y) % p for x, y in zip(M[i], M[r])]
        piv.append(c); r += 1
    if any(M[i][n] % p for i in range(r, len(M))): return None
    return [M[i][n] for i in range(n)]
def pade_mod(xs, vs, dp, dq, p):
    rows = []
    for x, v in zip(xs, vs):
        rows.append([pow(x, i, p) for i in range(dp + 1)] + [(-v * pow(x, i, p)) % p for i in range(dq)] + [v * pow(x, dq, p) % p])
    return solve_mod(rows, p)
p = 2147483647
for j in range(2):
    vs = [res[N][j].numerator * pow(res[N][j].denominator, -1, p) % p for N in Ns]
    # growth: deg num - deg den
    a, b = Ns[-20], Ns[-1]
    g = (math.log(abs(float(res[b][j]))) - math.log(abs(float(res[a][j])))) / (math.log(b) - math.log(a))
    print(f'Norm{j+1}: growth exponent ~ {g:.2f}', flush=True)
    found = None
    for tot in range(10, 186):
        for dq in range(0, tot + 1):
            dp = tot - dq
            if dp - dq < round(g) - 1 or dp - dq > round(g) + 1: continue
            npts = dp + dq + 1
            sol = pade_mod(Ns[:npts], vs[:npts], dp, dq, p)
            if sol is None: continue
            P_ = sol[:dp + 1]; Q_ = sol[dp + 1:] + [1]
            ok = all((sum(c * pow(N, i, p) for i, c in enumerate(P_)) - vs[k] * sum(c * pow(N, i, p) for i, c in enumerate(Q_))) % p == 0
                     for k, N in enumerate(Ns) if k >= npts)
            if ok: found = (dp, dq); break
        if found: break
    print(f'  Norm{j+1}: rational function of type (deg num, deg den) = {found}', flush=True)
