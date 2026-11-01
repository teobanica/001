# Exact data at integer N, only for the configurations used by the certificates.
import sys, pickle, io, contextlib
from fractions import Fraction as Fr
with contextlib.redirect_stdout(io.StringIO()):
    from levquad import contract
from levlin import noncrossing
import poly as P
S = pickle.load(open('step1sym.pkl', 'rb')); cr6, rec = S['cr6'], S['rec']
Dz = pickle.load(open('design.pkl', 'rb'))
confs = sorted({k[:5] for k in Dz['Qsel']} | {k[0] for k in Dz['C3keys']})
n = 5; sym = [(a, b) for a in range(n) for b in range(a, n)]; si = {p: i for i, p in enumerate(sym)}
out = {}
for N in [int(x) for x in sys.argv[1:]]:
    vecs = [{cr6[a]: P.ev(p, N) for a, p in enumerate(v) if P.ev(p, N) != 0} for v in rec]
    res = {}
    for (s, t, j, o1, o2) in confs:
        coeff = {}
        for a in range(n):
            for b in range(n):
                _, r = contract(vecs[a], vecs[b], s, t, j, o1, o2, N)
                for Pp, c in r.items():
                    if noncrossing(Pp): continue
                    v = coeff.setdefault(Pp, [Fr(0)] * len(sym))
                    v[si[(min(a, b), max(a, b))]] += c
        res[(s, t, j, o1, o2)] = coeff
    out[N] = res
    print('done', N, flush=True)
pickle.dump(out, open(sys.argv[0].replace('.py', '') + '_' + '_'.join('abcdefghij'[int(c)] for x in sys.argv[1:2] for c in x) + '.pkl', 'wb'))
