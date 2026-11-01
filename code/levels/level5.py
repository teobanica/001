# Level 5 with N symbolic: K_5(N) = ker of (A + N B) on crossing columns of P(5), into crossing part of P(4).
from fractions import Fraction as Fr
from levlin import parts, noncrossing, delete_leg, merge
import poly as P
k = 5
Pk = parts(k); Pl = parts(k - 1)
crk = [p for p in Pk if not noncrossing(p)]; crl = [q for q in Pl if not noncrossing(q)]
print('crossing partitions: level 5:', len(crk), ' level 4:', len(crl), crl)
il = {q: a for a, q in enumerate(crl)}
A = []; B = []
for typ, i in [('d', i) for i in range(k)] + [('m', i) for i in range(k)]:
    RA = [[0] * len(crk) for _ in crl]; RB = [[0] * len(crk) for _ in crl]
    for a, p in enumerate(crk):
        if typ == 'd':
            sing, q = delete_leg(p, i, k)
            if q in il: (RB if sing else RA)[il[q]][a] += 1
        else:
            q = merge(p, i, (i + 1) % k, k)
            if q in il: RA[il[q]][a] += 1
    A += RA; B += RB
for typ, i in [('d', i) for i in range(k)] + [('m', i) for i in range(k)]:
    for p in Pk:
        if noncrossing(p):
            q = delete_leg(p, i, k)[1] if typ == 'd' else merge(p, i, (i + 1) % k, k)
            assert noncrossing(q)
print(f'M5(N) = A + N B: {len(A)} x {len(crk)}; NC columns vanish identically')
def det(M):
    M = [r[:] for r in M]; n = len(M); D = Fr(1)
    for c in range(n):
        pv = next((r for r in range(c, n) if M[r][c] != 0), None)
        if pv is None: return Fr(0)
        if pv != c: M[c], M[pv] = M[pv], M[c]; D = -D
        D *= M[c][c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]; M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return D
n = len(crk); xs = list(range(n + 3))
ys = [det([[Fr(A[i][c] + N * B[i][c]) for c in range(n)] for i in range(len(A))]) for N in xs]
D5 = P.interp(xs[:-2], ys[:-2]); assert all(P.ev(D5, x) == y for x, y in zip(xs[-2:], ys[-2:]))
print('det M5(N) =', P.fmt(D5))
rem = D5; facs = {}
for r in range(-10, 11):
    while rem and P.ev(rem, r) == 0:
        rem, z = P.divmod_(rem, [Fr(-r), Fr(1)]); facs[r] = facs.get(r, 0) + 1
print('factors (N-r)^e:', facs, ' remaining:', P.fmt(rem))
