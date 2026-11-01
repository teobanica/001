import sys, itertools, numpy as np
sys.argv = ['x', '6', '11', '4', '200']
src = open('triple.py').read().replace("print(f'triple-contraction cubics alone", "print(f'(cubics alone)")
exec(src)
from engine import labels, forms, config_outputs
L_, nb = labels(crk, k)
qrows = []
for conf in [(s, t, j, o1, o2) for s in range(k) for t in range(k) for j in range(4, k + 1) for o1 in (0, 1) for o2 in (0, 1)
             if 2 * k - 2 * j + o1 + o2 <= k - 1 and len(config_outputs(k, (s, t, j, o1, o2))[2]) >= 4]:
    acc, _ = forms(L_, nb, k, conf, W.astype(np.float64), N, p=p)
    for Q in acc.values():
        Q = Q.astype(np.int64)
        for e in range(d):
            row = np.zeros(len(mon3), dtype=np.int64)
            for a in range(d):
                for b in range(a, d):
                    cf = Q[a, b] if a == b else (2 * Q[a, b]) % p      # Q symmetric (already Q+Q^T)
                    if cf: row[m3[tuple(sorted((a, b, e)))]] = (row[m3[tuple(sorted((a, b, e)))]] + cf) % p
            qrows.append(row)
print('quadrics x linear alone: rank', rank_mod(np.array(qrows)))
print('quadrics x linear + triple cubics: rank', rank_mod(np.array(qrows + rows)), 'of', len(mon3))
