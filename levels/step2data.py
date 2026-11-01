# For a given integer N: quadric rows (levels <= 5) and level-6 closure outputs,
# in the coordinates c of the polynomial basis w_b(N) = N(N-2) v_b(N) from step1sym.
import sys, pickle, io, contextlib
from fractions import Fraction as Fr
with contextlib.redirect_stdout(io.StringIO()):
    from levquad import contract
from levlin import noncrossing
import poly as P
S = pickle.load(open('step1sym.pkl', 'rb'))
cr6, rec = S['cr6'], S['rec']
N = int(sys.argv[1])
vecs = [{cr6[a]: P.ev(p, N) for a, p in enumerate(v) if P.ev(p, N) != 0} for v in rec]
n = 5; sym = [(a, b) for a in range(n) for b in range(a, n)]; si = {p: i for i, p in enumerate(sym)}
quad = {}     # (s,t,j,o1,o2,P) -> 15-vector
clos = {}     # (s,t,j,o1,o2) -> {P: 15-vector}
for s in range(6):
    for t in range(6):
        for j in (3, 4, 5, 6):
            for o1 in (0, 1):
                for o2 in (0, 1):
                    L = 12 - 2 * j + o1 + o2
                    if L > 6: continue
                    if j == 6 and o2: continue
                    if L == 6 and (j, o1, o2) not in [(3, 0, 0), (4, 1, 1)]: continue
                    coeff = {}
                    for a in range(n):
                        for b in range(n):
                            _, r = contract(vecs[a], vecs[b], s, t, j, o1, o2, N)
                            for Pp, c in r.items():
                                if noncrossing(Pp): continue
                                v = coeff.setdefault(Pp, [Fr(0)] * len(sym))
                                v[si[(min(a, b), max(a, b))]] += c
                    if L <= 5:
                        for Pp, v in coeff.items(): quad[(s, t, j, o1, o2, Pp)] = v
                    else:
                        clos[(s, t, j, o1, o2)] = coeff
pickle.dump({'N': N, 'quad': quad, 'clos': clos, 'vecs': vecs}, open(f'data/d_{"".join("abcdefghij"[int(ch)] for ch in str(N))}.pkl', 'wb'))
print(N, len(quad), len(clos))
