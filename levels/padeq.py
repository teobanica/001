import pickle, numpy as np
from fractions import Fraction as Fr
import poly as P
res = pickle.load(open('locus_x_zoo.pkl', 'rb')); Ns = sorted(res)
dp, dq = 55, 24
def pade_exact(xs, vs):
    n = dp + 1 + dq
    M = [[Fr(x) ** i for i in range(dp + 1)] + [-v * Fr(x) ** i for i in range(dq)] + [v * Fr(x) ** dq] for x, v in zip(xs, vs)]
    r = 0
    for c in range(n):
        pr = next(i for i in range(r, len(M)) if M[i][c] != 0)
        M[r], M[pr] = M[pr], M[r]; iv = 1 / M[r][c]; M[r] = [x * iv for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c]; M[i] = [x - f * y for x, y in zip(M[i], M[r])]
        r += 1
    sol = [M[i][n] for i in range(n)]
    return P.trim(sol[:dp + 1]), P.trim(sol[dp + 1:] + [Fr(1)])
out = {}
for j in range(2):
    vs = [res[N][j] for N in Ns]
    num, den = pade_exact(Ns[:80], vs[:80])
    ok = all(P.ev(num, N) == vs[k] * P.ev(den, N) for k, N in enumerate(Ns) if k >= 80)
    print(f'Norm{j+1}: exact reconstruction verified on {len(Ns)-80} extra points: {ok}', flush=True)
    out[j] = (num, den)
g = P.gcd(out[0][0], out[1][0])
print('gcd of numerators: degree', P.deg(g), flush=True)
gd = P.gcd(out[0][1], out[1][1]); print('gcd of denominators: degree', P.deg(gd))
def rootsinfo(pol, name):
    r = np.roots([float(x) for x in reversed(pol)])
    print(f'{name} roots:', np.round(np.sort_complex(r), 5))
rootsinfo(g, 'common numerator factor')
rootsinfo(out[0][1], 'denominator of Norm1')
# integer / rational roots of g
rem = g; facs = {}
for r in range(-20, 21):
    while P.deg(rem) > 0 and P.ev(rem, r) == 0:
        rem, z = P.divmod_(rem, [Fr(-r), Fr(1)]); facs[r] = facs.get(r, 0) + 1
print('integer root multiplicities in gcd:', facs, ' remaining factor:', P.fmt(rem)[:400])
pickle.dump({'norms': out, 'gcd': g}, open('locusrat.pkl', 'wb'))
